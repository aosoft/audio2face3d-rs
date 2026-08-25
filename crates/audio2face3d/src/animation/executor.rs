use crate::animation::{RegressionContract, RegressionFrameInput};
use crate::common::{AudioAccumulator, EmotionAccumulator, Error, Result};
use std::sync::{Condvar, Mutex};

pub const MAX_REGRESSION_TRACKS: usize = 32;

pub trait RegressionBackend {
    type Output;

    fn infer(&mut self, track: usize, input: &RegressionFrameInput) -> Result<Self::Output>;

    fn infer_batch(
        &mut self,
        inputs: &[(usize, RegressionFrameInput)],
    ) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|(track, input)| self.infer(*track, input))
            .collect()
    }
}

impl<T, F> RegressionBackend for F
where
    F: FnMut(usize, &RegressionFrameInput) -> Result<T>,
{
    type Output = T;

    fn infer(&mut self, track: usize, input: &RegressionFrameInput) -> Result<T> {
        self(track, input)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegressionCallbackMetadata {
    pub track: usize,
    pub frame: usize,
    pub timestamp: i64,
    pub next_timestamp: i64,
}

pub struct RegressionTrack<'a> {
    pub audio: &'a AudioAccumulator,
    pub emotions: &'a EmotionAccumulator,
    pub implicit_emotion: &'a [f32],
    pub input_strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PumpStatus {
    AwaitingInput,
    Complete,
    Interrupted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegressionExecutorState {
    pub running: bool,
    pub completed_tracks: usize,
    pub track_count: usize,
}

#[derive(Debug)]
struct State {
    running: bool,
    frames: Vec<usize>,
    stopped: Vec<bool>,
}

/// CPU orchestration for regression tracks. TensorRT and post-processing are
/// supplied by `RegressionBackend`, keeping scheduling independently testable.
#[derive(Debug)]
pub struct RegressionExecutor {
    contract: RegressionContract,
    state: Mutex<State>,
    idle: Condvar,
}

impl RegressionExecutor {
    pub fn new(contract: RegressionContract, track_count: usize) -> Result<Self> {
        if !(1..=MAX_REGRESSION_TRACKS).contains(&track_count) {
            return Err(Error::InvalidSchema(format!(
                "regression track count must be in 1..={MAX_REGRESSION_TRACKS}"
            )));
        }
        Ok(Self {
            contract,
            state: Mutex::new(State {
                running: false,
                frames: vec![0; track_count],
                stopped: vec![false; track_count],
            }),
            idle: Condvar::new(),
        })
    }

    pub fn pump<B, C>(
        &self,
        tracks: &[RegressionTrack<'_>],
        backend: &mut B,
        mut callback: C,
    ) -> Result<PumpStatus>
    where
        B: RegressionBackend,
        C: FnMut(RegressionCallbackMetadata, &B::Output) -> bool,
    {
        let mut state = self.lock()?;
        if tracks.len() != state.frames.len() {
            return Err(Error::InvalidSchema(format!(
                "received {} tracks, expected {}",
                tracks.len(),
                state.frames.len()
            )));
        }
        if state.running {
            return Err(Error::InvalidSchema(
                "regression executor is already running".into(),
            ));
        }
        state.running = true;
        drop(state);

        let result = self.pump_inner(tracks, backend, &mut callback);
        let mut state = self.lock()?;
        state.running = false;
        self.idle.notify_all();
        result
    }

    fn pump_inner<B, C>(
        &self,
        tracks: &[RegressionTrack<'_>],
        backend: &mut B,
        callback: &mut C,
    ) -> Result<PumpStatus>
    where
        B: RegressionBackend,
        C: FnMut(RegressionCallbackMetadata, &B::Output) -> bool,
    {
        loop {
            let mut pending = Vec::new();
            for (track_index, track) in tracks.iter().enumerate() {
                let frame = {
                    let state = self.lock()?;
                    if state.stopped[track_index] {
                        continue;
                    }
                    state.frames[track_index]
                };
                if self.track_finished(track, frame)? {
                    self.lock()?.stopped[track_index] = true;
                    continue;
                }
                if !self.track_ready(track, frame)? {
                    continue;
                }
                let input = self.contract.prepare_frame(
                    frame,
                    track.audio,
                    track.emotions,
                    track.implicit_emotion,
                    track.input_strength,
                )?;
                pending.push((track_index, frame, input));
            }
            if pending.is_empty() {
                let state = self.lock()?;
                if state.stopped.iter().all(|stopped| *stopped) {
                    return Ok(PumpStatus::Complete);
                }
                return Ok(PumpStatus::AwaitingInput);
            }
            let inference_inputs: Vec<_> = pending
                .iter()
                .map(|(track, _, input)| (*track, input.clone()))
                .collect();
            let outputs = backend.infer_batch(&inference_inputs)?;
            if outputs.len() != pending.len() {
                return Err(Error::InvalidSchema(
                    "regression backend returned the wrong batch size".into(),
                ));
            }
            for ((track_index, frame, input), output) in pending.into_iter().zip(outputs.iter()) {
                let metadata = RegressionCallbackMetadata {
                    track: track_index,
                    frame,
                    timestamp: input.timestamp,
                    next_timestamp: input.next_timestamp,
                };
                let keep_running = callback(metadata, output);
                {
                    let mut state = self.lock()?;
                    state.frames[track_index] += 1;
                    if !keep_running {
                        state.stopped.fill(true);
                    }
                }
                self.drop_consumed(&tracks[track_index], frame)?;
                if !keep_running {
                    return Ok(PumpStatus::Interrupted);
                }
            }
        }
    }

    fn track_ready(&self, track: &RegressionTrack<'_>, frame: usize) -> Result<bool> {
        let window = self.contract.progress.window(frame)?;
        let audio_end = window
            .start
            .checked_add(i64::try_from(self.contract.audio_size).map_err(|_| {
                Error::IntegerOverflow {
                    field: "regression_audio_size",
                    value: self.contract.audio_size,
                    target: "i64",
                }
            })?)
            .ok_or_else(|| Error::InvalidSchema("audio window overflow".into()))?;
        let audio_ready = track.audio.is_closed()
            || audio_end <= i64::try_from(track.audio.nb_accumulated_samples()).unwrap_or(i64::MAX);
        let emotion = track.emotions.state();
        let emotion_ready = emotion.key_count != 0
            && (emotion.closed || emotion.last_accumulated_timestamp >= window.target);
        Ok(audio_ready && emotion_ready)
    }

    fn track_finished(&self, track: &RegressionTrack<'_>, frame: usize) -> Result<bool> {
        if !track.audio.is_closed() {
            return Ok(false);
        }
        let window = self.contract.progress.window(frame)?;
        Ok(
            window.target
                >= i64::try_from(track.audio.nb_accumulated_samples()).unwrap_or(i64::MAX),
        )
    }

    fn drop_consumed(&self, track: &RegressionTrack<'_>, frame: usize) -> Result<()> {
        let current = self.contract.progress.window(frame)?;
        let next = self.contract.progress.window(frame + 1)?;
        track.audio.drop_samples_before(
            usize::try_from(next.start.max(0))
                .map_err(|_| Error::InvalidSchema("negative drop watermark".into()))?,
        )?;
        track
            .emotions
            .drop_before(current.target)
            .map_err(|error| Error::InvalidSchema(format!("emotion drop failed: {error}")))
    }

    pub fn wait(&self) -> Result<()> {
        let mut state = self.lock()?;
        while state.running {
            state = self
                .idle
                .wait(state)
                .map_err(|_| Error::InvalidSchema("executor mutex poisoned".into()))?;
        }
        Ok(())
    }

    pub fn state(&self) -> Result<RegressionExecutorState> {
        let state = self.lock()?;
        Ok(RegressionExecutorState {
            running: state.running,
            completed_tracks: state.stopped.iter().filter(|stopped| **stopped).count(),
            track_count: state.frames.len(),
        })
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| Error::InvalidSchema("executor mutex poisoned".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{RegressionAudioParameters, RegressionParameters};

    fn contract() -> RegressionContract {
        RegressionContract::new(
            &RegressionParameters {
                implicit_emotion_len: 1,
                explicit_emotions: vec!["joy".into()],
                default_emotion: vec![0.0],
                num_shapes_skin: 1,
                num_shapes_tongue: 0,
                num_verts_skin: 1,
                num_verts_tongue: 0,
                result_jaw_size: 0,
                result_eyes_size: 0,
            },
            &RegressionAudioParameters {
                buffer_len: 2,
                buffer_ofs: 0,
                samplerate: 2,
            },
            2,
            1,
        )
        .unwrap()
    }

    fn input() -> (AudioAccumulator, EmotionAccumulator) {
        let audio = AudioAccumulator::new(1, 0).unwrap();
        audio.accumulate(&[1.0, 2.0, 3.0]).unwrap();
        audio.close().unwrap();
        let emotion = EmotionAccumulator::new(1, 1).unwrap();
        emotion.accumulate(0, &[0.0]).unwrap();
        emotion.accumulate(3, &[1.0]).unwrap();
        emotion.close().unwrap();
        (audio, emotion)
    }

    #[test]
    fn rejects_zero_and_over_max_tracks() {
        assert!(RegressionExecutor::new(contract(), 0).is_err());
        assert!(RegressionExecutor::new(contract(), MAX_REGRESSION_TRACKS + 1).is_err());
        assert!(RegressionExecutor::new(contract(), 1).is_ok());
        assert!(RegressionExecutor::new(contract(), 2).is_ok());
        assert!(RegressionExecutor::new(contract(), MAX_REGRESSION_TRACKS).is_ok());
    }

    #[test]
    fn callback_interrupts_all_tracks_and_metadata_is_ordered() {
        let (audio0, emotion0) = input();
        let (audio1, emotion1) = input();
        let tracks = [
            RegressionTrack {
                audio: &audio0,
                emotions: &emotion0,
                implicit_emotion: &[0.5],
                input_strength: 1.0,
            },
            RegressionTrack {
                audio: &audio1,
                emotions: &emotion1,
                implicit_emotion: &[0.5],
                input_strength: 1.0,
            },
        ];
        let executor = RegressionExecutor::new(contract(), 2).unwrap();
        let mut seen = Vec::new();
        let mut backend = |track, input: &RegressionFrameInput| Ok((track, input.audio.clone()));
        let status = executor
            .pump(&tracks, &mut backend, |metadata, _| {
                seen.push(metadata);
                metadata.track != 0
            })
            .unwrap();
        assert_eq!(status, PumpStatus::Interrupted);
        assert_eq!(seen.iter().filter(|m| m.track == 0).count(), 1);
        assert_eq!(seen.iter().filter(|m| m.track == 1).count(), 0);
        assert_eq!(executor.state().unwrap().completed_tracks, 2);
        executor.wait().unwrap();
        assert!(audio0.nb_dropped_samples() > 0);
    }

    #[test]
    fn open_stream_waits_for_input_and_resumes() {
        let audio = AudioAccumulator::new(1, 0).unwrap();
        audio.accumulate(&[1.0]).unwrap();
        let emotion = EmotionAccumulator::new(1, 1).unwrap();
        emotion.accumulate(0, &[0.0]).unwrap();
        let track = RegressionTrack {
            audio: &audio,
            emotions: &emotion,
            implicit_emotion: &[0.0],
            input_strength: 1.0,
        };
        let executor = RegressionExecutor::new(contract(), 1).unwrap();
        let mut calls = 0;
        let mut backend = |_, _: &RegressionFrameInput| -> Result<()> {
            calls += 1;
            Ok(())
        };
        assert_eq!(
            executor.pump(&[track], &mut backend, |_, _| true).unwrap(),
            PumpStatus::AwaitingInput
        );
        assert_eq!(calls, 0);
    }

    #[test]
    fn executes_maximum_track_boundary() {
        let inputs: Vec<_> = (0..MAX_REGRESSION_TRACKS).map(|_| input()).collect();
        let tracks: Vec<_> = inputs
            .iter()
            .map(|(audio, emotions)| RegressionTrack {
                audio,
                emotions,
                implicit_emotion: &[0.0],
                input_strength: 1.0,
            })
            .collect();
        let executor = RegressionExecutor::new(contract(), MAX_REGRESSION_TRACKS).unwrap();
        let mut callbacks = 0;
        let mut backend = |_, _: &RegressionFrameInput| -> Result<()> { Ok(()) };
        assert_eq!(
            executor
                .pump(&tracks, &mut backend, |_, _| {
                    callbacks += 1;
                    true
                })
                .unwrap(),
            PumpStatus::Complete
        );
        assert_eq!(callbacks, MAX_REGRESSION_TRACKS * 3);
    }
}
