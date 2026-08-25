use crate::animation::{
    DiffusionContract, DiffusionFrameInput, DiffusionInferenceOutput, DiffusionState, PhiloxNoise,
};
use crate::common::{AudioAccumulator, EmotionAccumulator, Error, Result};
use std::sync::{Condvar, Mutex};

pub const MAX_DIFFUSION_TRACKS: usize = 32;

pub trait DiffusionBackend {
    fn infer_batch(
        &mut self,
        inputs: &[(usize, DiffusionFrameInput)],
    ) -> Result<Vec<DiffusionInferenceOutput>>;
}

impl<F> DiffusionBackend for F
where
    F: FnMut(&[(usize, DiffusionFrameInput)]) -> Result<Vec<DiffusionInferenceOutput>>,
{
    fn infer_batch(
        &mut self,
        inputs: &[(usize, DiffusionFrameInput)],
    ) -> Result<Vec<DiffusionInferenceOutput>> {
        self(inputs)
    }
}

pub struct DiffusionTrack<'a> {
    pub audio: &'a AudioAccumulator,
    pub emotions: &'a EmotionAccumulator,
    pub identity_index: usize,
    pub input_strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffusionCallbackMetadata {
    pub track: usize,
    pub inference: usize,
    pub frame: usize,
    pub timestamp: i64,
    pub next_timestamp: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffusionExecutionStatus {
    AwaitingInput,
    Executed { tracks: usize },
    Complete,
}

#[derive(Debug)]
struct ExecutorState {
    running: bool,
    inferences: Vec<usize>,
}

/// Stateful multi-track scheduler matching the SDK's frame-major callback order.
#[derive(Debug)]
pub struct DiffusionExecutor {
    contract: DiffusionContract,
    recurrent: Mutex<DiffusionState>,
    noise: Mutex<PhiloxNoise>,
    state: Mutex<ExecutorState>,
    idle: Condvar,
}

impl DiffusionExecutor {
    pub fn new(contract: DiffusionContract, track_count: usize, seed: u64) -> Result<Self> {
        if !(1..=MAX_DIFFUSION_TRACKS).contains(&track_count) {
            return Err(invalid(format!(
                "diffusion track count must be in 1..={MAX_DIFFUSION_TRACKS}"
            )));
        }
        let recurrent = DiffusionState::new(&contract, track_count)?;
        let noise = PhiloxNoise::new(track_count, contract.noise_size()?, seed)?;
        Ok(Self {
            contract,
            recurrent: Mutex::new(recurrent),
            noise: Mutex::new(noise),
            state: Mutex::new(ExecutorState {
                running: false,
                inferences: vec![0; track_count],
            }),
            idle: Condvar::new(),
        })
    }

    /// Runs one available inference for each ready track.
    ///
    /// A callback returning false suppresses the remaining frames of that track
    /// for this inference only. Other tracks and recurrent-state advancement are
    /// intentionally unaffected, matching the original executor.
    pub fn execute<B, C>(
        &self,
        tracks: &[DiffusionTrack<'_>],
        backend: &mut B,
        mut callback: C,
    ) -> Result<DiffusionExecutionStatus>
    where
        B: DiffusionBackend,
        C: FnMut(DiffusionCallbackMetadata, &[f32]) -> bool,
    {
        {
            let mut state = self.state.lock().map_err(|_| poisoned())?;
            if tracks.len() != state.inferences.len() {
                return Err(invalid("diffusion track count mismatch"));
            }
            if state.running {
                return Err(invalid("diffusion executor is already running"));
            }
            state.running = true;
        }
        let result = self.execute_inner(tracks, backend, &mut callback);
        let mut state = self.state.lock().map_err(|_| poisoned())?;
        state.running = false;
        self.idle.notify_all();
        result
    }

    fn execute_inner<B, C>(
        &self,
        tracks: &[DiffusionTrack<'_>],
        backend: &mut B,
        callback: &mut C,
    ) -> Result<DiffusionExecutionStatus>
    where
        B: DiffusionBackend,
        C: FnMut(DiffusionCallbackMetadata, &[f32]) -> bool,
    {
        let inference_indices = self
            .state
            .lock()
            .map_err(|_| poisoned())?
            .inferences
            .clone();
        let mut pending = Vec::new();
        for (track_index, track) in tracks.iter().enumerate() {
            let inference = inference_indices[track_index];
            if self.finished(track, inference)? || !self.ready(track, inference)? {
                continue;
            }
            pending.push((
                track_index,
                self.prepare_input(track_index, inference, track)?,
            ));
        }
        if pending.is_empty() {
            return Ok(
                if tracks
                    .iter()
                    .zip(inference_indices)
                    .all(|(track, inference)| self.finished(track, inference).unwrap_or(false))
                {
                    DiffusionExecutionStatus::Complete
                } else {
                    DiffusionExecutionStatus::AwaitingInput
                },
            );
        }
        let outputs = backend.infer_batch(&pending)?;
        if outputs.len() != pending.len() {
            return Err(invalid("diffusion backend returned the wrong batch size"));
        }
        let expected_prediction = self
            .contract
            .total_frames()
            .checked_mul(self.contract.result_layout.total()?)
            .ok_or_else(|| invalid("diffusion prediction size overflow"))?;
        let expected_state = self.contract.state_size()?;
        for output in &outputs {
            if output.prediction.len() != expected_prediction
                || output.output_latents.len() != expected_state
            {
                return Err(invalid("diffusion backend output dimensions mismatch"));
            }
        }
        {
            let updates: Vec<_> = pending
                .iter()
                .zip(&outputs)
                .map(|((track, _), output)| (*track, output.output_latents.as_slice()))
                .collect();
            self.recurrent
                .lock()
                .map_err(|_| poisoned())?
                .commit_tracks(&updates)?;
        }

        let result_size = self.contract.result_layout.total()?;
        let mut callback_active = vec![false; tracks.len()];
        for (track, _) in &pending {
            callback_active[*track] = true;
        }
        for frame in 0..self.contract.center_frames {
            for ((track, _), output) in pending.iter().zip(&outputs) {
                if !callback_active[*track] {
                    continue;
                }
                let inference = inference_indices[*track];
                let window = self
                    .contract
                    .frame_progress
                    .window(inference * self.contract.center_frames + frame)?;
                if window.target < 0 {
                    continue;
                }
                if window.target
                    >= i64::try_from(tracks[*track].audio.nb_accumulated_samples())
                        .unwrap_or(i64::MAX)
                {
                    callback_active[*track] = false;
                    continue;
                }
                let prediction_frame = self.contract.left_frames + frame;
                let offset = prediction_frame * result_size;
                let keep_going = callback(
                    DiffusionCallbackMetadata {
                        track: *track,
                        inference,
                        frame,
                        timestamp: window.target,
                        next_timestamp: self
                            .contract
                            .frame_progress
                            .window(inference * self.contract.center_frames + frame + 1)?
                            .target,
                    },
                    &output.prediction[offset..offset + result_size],
                );
                if !keep_going {
                    callback_active[*track] = false;
                }
            }
        }

        let mut state = self.state.lock().map_err(|_| poisoned())?;
        for (track, _) in &pending {
            state.inferences[*track] += 1;
            self.drop_consumed(&tracks[*track], state.inferences[*track])?;
        }
        Ok(DiffusionExecutionStatus::Executed {
            tracks: pending.len(),
        })
    }

    fn prepare_input(
        &self,
        track_index: usize,
        inference: usize,
        track: &DiffusionTrack<'_>,
    ) -> Result<DiffusionFrameInput> {
        if track.identity_index >= self.contract.identity_size {
            return Err(invalid("diffusion identity index is out of range"));
        }
        let window = self.contract.progress.window(inference)?;
        let audio =
            track
                .audio
                .read(window.start, self.contract.audio_size, track.input_strength)?;
        let mut emotions =
            Vec::with_capacity(self.contract.center_frames * self.contract.emotion_size);
        for frame in 0..self.contract.center_frames {
            let target = self
                .contract
                .frame_progress
                .window(inference * self.contract.center_frames + frame)?
                .target;
            emotions.extend(
                track
                    .emotions
                    .read(target)
                    .map_err(|error| invalid(format!("diffusion emotion read failed: {error}")))?,
            );
        }
        let mut identity = vec![0.0; self.contract.identity_size];
        identity[track.identity_index] = 1.0;
        let noise = self
            .noise
            .lock()
            .map_err(|_| poisoned())?
            .generate(track_index)?;
        let input_latents = self
            .recurrent
            .lock()
            .map_err(|_| poisoned())?
            .track_input(track_index)?;
        Ok(DiffusionFrameInput {
            audio,
            emotions,
            identity,
            noise,
            input_latents,
        })
    }

    fn ready(&self, track: &DiffusionTrack<'_>, inference: usize) -> Result<bool> {
        let window = self.contract.progress.window(inference)?;
        let audio_ready = track.audio.is_closed()
            || window.end
                <= i64::try_from(track.audio.nb_accumulated_samples()).unwrap_or(i64::MAX);
        let last_emotion = self
            .contract
            .frame_progress
            .window(inference * self.contract.center_frames + self.contract.center_frames - 1)?
            .target;
        let emotion = track.emotions.state();
        Ok(audio_ready
            && emotion.key_count != 0
            && (emotion.closed || emotion.last_accumulated_timestamp >= last_emotion))
    }

    fn finished(&self, track: &DiffusionTrack<'_>, inference: usize) -> Result<bool> {
        Ok(track.audio.is_closed()
            && self.contract.progress.window(inference)?.target
                >= i64::try_from(track.audio.nb_accumulated_samples()).unwrap_or(i64::MAX))
    }

    fn drop_consumed(&self, track: &DiffusionTrack<'_>, next_inference: usize) -> Result<()> {
        let next = self.contract.progress.window(next_inference)?;
        track
            .audio
            .drop_samples_before(usize::try_from(next.start.max(0)).unwrap_or(usize::MAX))?;
        let previous_frame = next_inference
            .checked_mul(self.contract.center_frames)
            .and_then(|value| value.checked_sub(1));
        if let Some(frame) = previous_frame {
            let target = self.contract.frame_progress.window(frame)?.target;
            track
                .emotions
                .drop_before(target)
                .map_err(|error| invalid(format!("diffusion emotion drop failed: {error}")))?;
        }
        Ok(())
    }

    pub fn reset(&self, track: usize) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| poisoned())?;
        let inference = state
            .inferences
            .get_mut(track)
            .ok_or_else(|| invalid("diffusion reset track is out of range"))?;
        *inference = 0;
        self.recurrent
            .lock()
            .map_err(|_| poisoned())?
            .reset(track)?;
        self.noise.lock().map_err(|_| poisoned())?.reset(track, 0)
    }

    pub fn wait(&self) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| poisoned())?;
        while state.running {
            state = self.idle.wait(state).map_err(|_| poisoned())?;
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

fn poisoned() -> Error {
    invalid("diffusion executor mutex poisoned")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{DiffusionAudioParameters, DiffusionParameters};

    fn contract() -> DiffusionContract {
        DiffusionContract::new(
            &DiffusionParameters {
                emotions: vec!["joy".into()],
                default_emotion: vec![0.0],
                identities: vec!["actor".into()],
                skin_size: 3,
                tongue_size: 3,
                jaw_size: 3,
                eyes_size: 4,
                num_diffusion_steps: 1,
                num_gru_layers: 1,
                gru_latent_dim: 2,
                num_frames_left_truncate: 1,
                num_frames_right_truncate: 1,
                num_frames_center: 2,
            },
            &DiffusionAudioParameters {
                buffer_len: 8,
                padding_left: 8,
                padding_right: 8,
                samplerate: 8,
            },
        )
        .unwrap()
    }

    fn accumulators() -> (AudioAccumulator, EmotionAccumulator) {
        let audio = AudioAccumulator::new(4, 0).unwrap();
        audio.accumulate(&[1.0; 12]).unwrap();
        audio.close().unwrap();
        let emotions = EmotionAccumulator::new(1, 2).unwrap();
        emotions.accumulate(-8, &[0.0]).unwrap();
        emotions.accumulate(12, &[1.0]).unwrap();
        emotions.close().unwrap();
        (audio, emotions)
    }

    #[test]
    fn callback_stops_only_one_track_and_state_advances() {
        let contract = contract();
        let (audio0, emotion0) = accumulators();
        let (audio1, emotion1) = accumulators();
        let tracks = [
            DiffusionTrack {
                audio: &audio0,
                emotions: &emotion0,
                identity_index: 0,
                input_strength: 1.0,
            },
            DiffusionTrack {
                audio: &audio1,
                emotions: &emotion1,
                identity_index: 0,
                input_strength: 1.0,
            },
        ];
        let executor = DiffusionExecutor::new(contract.clone(), 2, 9).unwrap();
        let result_size = contract.result_layout.total().unwrap();
        let mut backend = |inputs: &[(usize, DiffusionFrameInput)]| {
            Ok(inputs
                .iter()
                .map(|(track, input)| DiffusionInferenceOutput {
                    output_latents: vec![*track as f32 + 1.0; input.input_latents.len()],
                    prediction: vec![*track as f32; contract.total_frames() * result_size],
                })
                .collect())
        };
        for _ in 0..2 {
            executor
                .execute(&tracks, &mut backend, |_, _| true)
                .unwrap();
        }
        let mut seen = Vec::new();
        executor
            .execute(&tracks, &mut backend, |metadata, _| {
                seen.push(metadata);
                metadata.track != 0
            })
            .unwrap();
        assert_eq!(seen.iter().filter(|item| item.track == 0).count(), 1);
        assert!(seen.iter().filter(|item| item.track == 1).count() > 1);
        executor.wait().unwrap();
        executor.reset(0).unwrap();
    }

    #[test]
    fn open_input_waits_without_consuming_rng() {
        let contract = contract();
        let audio = AudioAccumulator::new(4, 0).unwrap();
        let emotions = EmotionAccumulator::new(1, 1).unwrap();
        emotions.accumulate(0, &[0.0]).unwrap();
        let track = DiffusionTrack {
            audio: &audio,
            emotions: &emotions,
            identity_index: 0,
            input_strength: 1.0,
        };
        let executor = DiffusionExecutor::new(contract.clone(), 1, 1).unwrap();
        let result_size = contract.result_layout.total().unwrap();
        let total_frames = contract.total_frames();
        let mut backend = move |inputs: &[(usize, DiffusionFrameInput)]| {
            Ok(inputs
                .iter()
                .map(|(_, input)| DiffusionInferenceOutput {
                    output_latents: vec![0.0; input.input_latents.len()],
                    prediction: vec![0.0; total_frames * result_size],
                })
                .collect())
        };
        assert_eq!(
            executor
                .execute(&[track], &mut backend, |_, _| true)
                .unwrap(),
            DiffusionExecutionStatus::Executed { tracks: 1 }
        );
        let track = DiffusionTrack {
            audio: &audio,
            emotions: &emotions,
            identity_index: 0,
            input_strength: 1.0,
        };
        assert_eq!(
            executor
                .execute(&[track], &mut backend, |_, _| true)
                .unwrap(),
            DiffusionExecutionStatus::AwaitingInput
        );
    }
}
