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
struct RegressionSchedulerState {
    running: bool,
    frames: Vec<usize>,
    completed: Vec<bool>,
}

/// Runtime state for the regression scheduler.
///
/// This name is intentionally implementation-oriented. The Step 3 owning
/// facade will keep this state together with its concrete backend and
/// post-processor in [`RegressionGeometryExecution`].
#[derive(Debug)]
struct RegressionExecutionState {
    contract: RegressionContract,
    state: Mutex<RegressionSchedulerState>,
    idle: Condvar,
}

struct RegressionRunningGuard<'a> {
    execution: &'a RegressionExecutionState,
}

impl Drop for RegressionRunningGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.execution.state.lock() {
            state.running = false;
            self.execution.idle.notify_all();
        }
    }
}

impl RegressionExecutionState {
    fn new(contract: RegressionContract, track_count: usize) -> Result<Self> {
        if !(1..=MAX_REGRESSION_TRACKS).contains(&track_count) {
            return Err(Error::InvalidSchema(format!(
                "regression track count must be in 1..={MAX_REGRESSION_TRACKS}"
            )));
        }
        Ok(Self {
            contract,
            state: Mutex::new(RegressionSchedulerState {
                running: false,
                frames: vec![0; track_count],
                completed: vec![false; track_count],
            }),
            idle: Condvar::new(),
        })
    }

    fn pump<B, C>(
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
        let _running = RegressionRunningGuard { execution: self };
        self.pump_inner(tracks, backend, &mut callback)
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
        let mut callback_active = vec![true; tracks.len()];
        let mut interrupted = false;
        loop {
            let mut pending = Vec::new();
            for (track_index, track) in tracks.iter().enumerate() {
                let frame = {
                    let state = self.lock()?;
                    if state.completed[track_index] {
                        continue;
                    }
                    state.frames[track_index]
                };
                if self.track_finished(track, frame)? {
                    self.lock()?.completed[track_index] = true;
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
                if interrupted {
                    return Ok(PumpStatus::Interrupted);
                }
                if state.completed.iter().all(|completed| *completed) {
                    return Ok(PumpStatus::Complete);
                }
                return Ok(PumpStatus::AwaitingInput);
            }
            let inference_inputs = (0..tracks.len())
                .map(|track| {
                    pending
                        .iter()
                        .find(|(pending_track, _, _)| *pending_track == track)
                        .map_or_else(
                            || {
                                (
                                    track,
                                    RegressionFrameInput {
                                        timestamp: 0,
                                        next_timestamp: 0,
                                        audio: vec![0.0; self.contract.audio_size],
                                        emotion: vec![0.0; self.contract.emotion_size],
                                    },
                                )
                            },
                            |(_, _, input)| (track, input.clone()),
                        )
                })
                .collect::<Vec<_>>();
            let outputs = backend.infer_batch(&inference_inputs)?;
            if outputs.len() != tracks.len() {
                return Err(Error::InvalidSchema(
                    "regression backend returned a non-fixed batch size".into(),
                ));
            }
            for (track_index, frame, input) in pending {
                let metadata = RegressionCallbackMetadata {
                    track: track_index,
                    frame,
                    timestamp: input.timestamp,
                    next_timestamp: input.next_timestamp,
                };
                if callback_active[track_index] && !callback(metadata, &outputs[track_index]) {
                    callback_active[track_index] = false;
                    interrupted = true;
                }
                {
                    let mut state = self.lock()?;
                    state.frames[track_index] += 1;
                }
                self.drop_consumed(&tracks[track_index], frame)?;
            }
        }
    }

    #[cfg(feature = "tensorrt")]
    #[allow(clippy::too_many_arguments)]
    fn pump_device<C>(
        &self,
        tracks: &[RegressionTrack<'_>],
        backend: &mut crate::animation::TensorRtRegressionBackend,
        postprocessor: &mut crate::animation::GpuRegressionPcaPostprocessor,
        skin: &mut crate::cuda::DeviceBuffer<f32>,
        tongue: &mut crate::cuda::DeviceBuffer<f32>,
        jaw: &mut crate::cuda::DeviceBuffer<f32>,
        eyes: &mut crate::cuda::DeviceBuffer<f32>,
        mut callback: C,
    ) -> Result<PumpStatus>
    where
        C: for<'a> FnMut(
            RegressionCallbackMetadata,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::CudaStreamRef<'a>,
        ) -> bool,
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
        let _running = RegressionRunningGuard { execution: self };
        let mut callback_active = vec![true; tracks.len()];
        let mut interrupted = false;
        loop {
            let mut pending = Vec::new();
            for (track_index, track) in tracks.iter().enumerate() {
                let frame = {
                    let state = self.lock()?;
                    if state.completed[track_index] {
                        continue;
                    }
                    state.frames[track_index]
                };
                if self.track_finished(track, frame)? {
                    self.lock()?.completed[track_index] = true;
                    continue;
                }
                if !self.track_ready(track, frame)? {
                    continue;
                }
                pending.push((
                    track_index,
                    frame,
                    self.contract.prepare_frame(
                        frame,
                        track.audio,
                        track.emotions,
                        track.implicit_emotion,
                        track.input_strength,
                    )?,
                ));
            }
            if pending.is_empty() {
                let state = self.lock()?;
                if interrupted {
                    return Ok(PumpStatus::Interrupted);
                }
                if state.completed.iter().all(|completed| *completed) {
                    return Ok(PumpStatus::Complete);
                }
                return Ok(PumpStatus::AwaitingInput);
            }
            let inference_inputs = (0..tracks.len())
                .map(|track| {
                    pending
                        .iter()
                        .find(|(pending_track, _, _)| *pending_track == track)
                        .map_or_else(
                            || RegressionFrameInput {
                                timestamp: 0,
                                next_timestamp: 0,
                                audio: vec![0.0; self.contract.audio_size],
                                emotion: vec![0.0; self.contract.emotion_size],
                            },
                            |(_, _, input)| input.clone(),
                        )
                })
                .collect::<Vec<_>>();
            let raw = backend.run_device_batch(&inference_inputs)?;
            let active = pending
                .iter()
                .map(|(track, _, _)| *track)
                .filter(|track| callback_active[*track])
                .collect::<Vec<_>>();
            if !active.is_empty() {
                let fence = postprocessor.enqueue(
                    raw.result_tensor(),
                    &active,
                    crate::animation::GpuRegressionOutputs {
                        skin,
                        tongue,
                        jaw_transforms: jaw,
                        eyes_rotations: eyes,
                    },
                    backend.stream(),
                )?;
                fence.synchronize()?;
                drop(fence);
            }
            for (track, frame, input) in pending {
                let metadata = RegressionCallbackMetadata {
                    track,
                    frame,
                    timestamp: input.timestamp,
                    next_timestamp: input.next_timestamp,
                };
                if callback_active[track] {
                    let keep_going = callback(
                        metadata,
                        skin.view().slice(
                            track * self.contract.result_skin_size,
                            self.contract.result_skin_size,
                        )?,
                        tongue.view().slice(
                            track * self.contract.result_tongue_size,
                            self.contract.result_tongue_size,
                        )?,
                        jaw.view().slice(track * 16, 16)?,
                        eyes.view().slice(track * 6, 6)?,
                        backend.stream().as_ref(),
                    );
                    if !keep_going {
                        callback_active[track] = false;
                        interrupted = true;
                    }
                }
                self.lock()?.frames[track] += 1;
                self.drop_consumed(&tracks[track], frame)?;
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

    fn wait(&self) -> Result<()> {
        let mut state = self.lock()?;
        while state.running {
            state = self
                .idle
                .wait(state)
                .map_err(|_| Error::InvalidSchema("executor mutex poisoned".into()))?;
        }
        Ok(())
    }

    fn state(&self) -> Result<RegressionExecutorState> {
        let state = self.lock()?;
        Ok(RegressionExecutorState {
            running: state.running,
            completed_tracks: state
                .completed
                .iter()
                .filter(|completed| **completed)
                .count(),
            track_count: state.frames.len(),
        })
    }

    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    fn next_frame_index(&self, track: usize) -> Result<usize> {
        let state = self.lock()?;
        state.frames.get(track).copied().ok_or(Error::OutOfBounds {
            field: "track",
            index: track,
            len: state.frames.len(),
        })
    }

    fn reset(&self, track: usize) -> Result<()> {
        let mut state = self.lock()?;
        if state.running {
            return Err(Error::InvalidState {
                operation: "reset regression track",
                state: "execution is running",
            });
        }
        let len = state.frames.len();
        let frame = state.frames.get_mut(track).ok_or(Error::OutOfBounds {
            field: "track",
            index: track,
            len,
        })?;
        *frame = 0;
        state.completed[track] = false;
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, RegressionSchedulerState>> {
        self.state
            .lock()
            .map_err(|_| Error::InvalidSchema("executor mutex poisoned".into()))
    }
}

/// Internal static-dispatch execution used by the non-generic facade.
///
/// `B` and `P` are implementation details and this concrete name is not
/// re-exported from [`crate::animation`]. The legacy [`RegressionExecutor`]
/// alias remains temporarily available until Step 7.
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct RegressionGeometryExecution<B, P> {
    state: RegressionExecutionState,
    backend: B,
    postprocessor: P,
}

/// Legacy low-level scheduler retained until the Step 7 API removal.
#[derive(Debug)]
pub struct RegressionExecutor {
    inner: RegressionGeometryExecution<(), ()>,
}

impl RegressionExecutor {
    pub fn new(contract: RegressionContract, track_count: usize) -> Result<Self> {
        Ok(Self {
            inner: RegressionGeometryExecution::with_dependencies(contract, track_count, (), ())?,
        })
    }

    pub fn pump<B, C>(
        &self,
        tracks: &[RegressionTrack<'_>],
        backend: &mut B,
        callback: C,
    ) -> Result<PumpStatus>
    where
        B: RegressionBackend,
        C: FnMut(RegressionCallbackMetadata, &B::Output) -> bool,
    {
        self.inner.state.pump(tracks, backend, callback)
    }

    #[cfg(feature = "tensorrt")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pump_device<C>(
        &self,
        tracks: &[RegressionTrack<'_>],
        backend: &mut crate::animation::TensorRtRegressionBackend,
        postprocessor: &mut crate::animation::GpuRegressionPcaPostprocessor,
        skin: &mut crate::cuda::DeviceBuffer<f32>,
        tongue: &mut crate::cuda::DeviceBuffer<f32>,
        jaw: &mut crate::cuda::DeviceBuffer<f32>,
        eyes: &mut crate::cuda::DeviceBuffer<f32>,
        callback: C,
    ) -> Result<PumpStatus>
    where
        C: for<'a> FnMut(
            RegressionCallbackMetadata,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::CudaStreamRef<'a>,
        ) -> bool,
    {
        self.inner.state.pump_device(
            tracks,
            backend,
            postprocessor,
            skin,
            tongue,
            jaw,
            eyes,
            callback,
        )
    }

    pub fn wait(&self) -> Result<()> {
        self.inner.wait()
    }

    pub fn state(&self) -> Result<RegressionExecutorState> {
        self.inner.state()
    }

    pub fn reset(&self, track: usize) -> Result<()> {
        self.inner.reset(track)
    }

    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    pub(crate) fn next_frame_index(&self, track: usize) -> Result<usize> {
        self.inner.state.next_frame_index(track)
    }
}

impl<B, P> RegressionGeometryExecution<B, P> {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_dependencies(
        contract: RegressionContract,
        track_count: usize,
        backend: B,
        postprocessor: P,
    ) -> Result<Self> {
        Ok(Self {
            state: RegressionExecutionState::new(contract, track_count)?,
            backend,
            postprocessor,
        })
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.state.wait()
    }

    pub(crate) fn state(&self) -> Result<RegressionExecutorState> {
        self.state.state()
    }

    pub(crate) fn reset(&self, track: usize) -> Result<()> {
        self.state.reset(track)
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn backend(&self) -> &B {
        &self.backend
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn postprocessor(&self) -> &P {
        &self.postprocessor
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn postprocessor_mut(&mut self) -> &mut P {
        &mut self.postprocessor
    }
}

impl<B, P> RegressionGeometryExecution<B, P>
where
    B: RegressionBackend,
{
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn execute<C>(
        &mut self,
        tracks: &[RegressionTrack<'_>],
        callback: C,
    ) -> Result<PumpStatus>
    where
        C: FnMut(RegressionCallbackMetadata, &B::Output) -> bool,
    {
        self.state.pump(tracks, &mut self.backend, callback)
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

    #[derive(Default)]
    struct FixedBatchTrace {
        batches: Vec<Vec<(usize, RegressionFrameInput)>>,
    }

    impl RegressionBackend for FixedBatchTrace {
        type Output = usize;

        fn infer(&mut self, track: usize, _: &RegressionFrameInput) -> Result<Self::Output> {
            Ok(track)
        }

        fn infer_batch(
            &mut self,
            inputs: &[(usize, RegressionFrameInput)],
        ) -> Result<Vec<Self::Output>> {
            self.batches.push(inputs.to_vec());
            Ok(inputs.iter().map(|(track, _)| *track).collect())
        }
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
    fn callback_interrupts_only_its_track_for_the_current_call() {
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
        let backend = |track, input: &RegressionFrameInput| Ok((track, input.audio.clone()));
        let mut execution =
            RegressionGeometryExecution::with_dependencies(contract(), 2, backend, Vec::new())
                .unwrap();
        assert!(execution.postprocessor().is_empty());
        execution.postprocessor_mut().push("trace");
        let _ = execution.backend();
        let _ = execution.backend_mut();
        let mut seen = Vec::new();
        let status = execution
            .execute(&tracks, |metadata, _| {
                seen.push(metadata);
                metadata.track != 0
            })
            .unwrap();
        assert_eq!(status, PumpStatus::Interrupted);
        assert_eq!(seen.iter().filter(|m| m.track == 0).count(), 1);
        assert_eq!(seen.iter().filter(|m| m.track == 1).count(), 3);
        assert_eq!(execution.state().unwrap().completed_tracks, 2);
        execution.wait().unwrap();
        assert!(audio0.nb_dropped_samples() > 0);
    }

    #[test]
    fn fixed_batch_keeps_inactive_track_slots_without_advancing_them() {
        let (audio0, emotion0) = input();
        let audio1 = AudioAccumulator::new(1, 0).unwrap();
        let emotion1 = EmotionAccumulator::new(1, 1).unwrap();
        emotion1.accumulate(0, &[0.0]).unwrap();
        emotion1.accumulate(3, &[1.0]).unwrap();
        let (audio2, emotion2) = input();
        let audio3 = AudioAccumulator::new(1, 0).unwrap();
        let emotion3 = EmotionAccumulator::new(1, 1).unwrap();
        emotion3.accumulate(0, &[0.0]).unwrap();
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
            RegressionTrack {
                audio: &audio2,
                emotions: &emotion2,
                implicit_emotion: &[0.5],
                input_strength: 1.0,
            },
            RegressionTrack {
                audio: &audio3,
                emotions: &emotion3,
                implicit_emotion: &[0.5],
                input_strength: 1.0,
            },
        ];
        let mut backend = FixedBatchTrace::default();
        let mut first_seen = Vec::new();
        let executor = RegressionExecutor::new(contract(), 4).unwrap();

        assert_eq!(
            executor
                .pump(&tracks, &mut backend, |metadata, output| {
                    first_seen.push((metadata.track, metadata.frame, *output));
                    true
                })
                .unwrap(),
            PumpStatus::AwaitingInput
        );
        assert!(
            first_seen
                .iter()
                .all(|(track, _, output)| { matches!(*track, 0 | 2) && track == output })
        );
        assert!(backend.batches.iter().all(|batch| batch.len() == 4));
        assert!(backend.batches.iter().all(|batch| {
            batch[1].1.audio.iter().all(|sample| *sample == 0.0)
                && batch[3].1.audio.iter().all(|sample| *sample == 0.0)
        }));
        assert_eq!(executor.state().unwrap().completed_tracks, 2);

        audio1.accumulate(&[1.0, 2.0, 3.0]).unwrap();
        audio1.close().unwrap();
        emotion1.close().unwrap();
        let mut resumed_frames = Vec::new();
        assert_eq!(
            executor
                .pump(&tracks, &mut backend, |metadata, _| {
                    resumed_frames.push((metadata.track, metadata.frame));
                    true
                })
                .unwrap(),
            PumpStatus::AwaitingInput
        );
        assert_eq!(resumed_frames.first(), Some(&(1, 0)));
        assert!(resumed_frames.iter().all(|(track, _)| *track == 1));
    }

    #[test]
    fn callback_panic_releases_running_guard() {
        let (audio, emotions) = input();
        let track = RegressionTrack {
            audio: &audio,
            emotions: &emotions,
            implicit_emotion: &[0.5],
            input_strength: 1.0,
        };
        let executor = RegressionExecutor::new(contract(), 1).unwrap();
        let mut backend = |track, _: &RegressionFrameInput| Ok(track);
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = executor.pump(&[track], &mut backend, |_, _| -> bool {
                panic!("callback panic")
            });
        }));
        assert!(panic.is_err());
        assert!(!executor.state().unwrap().running);
        executor.reset(0).unwrap();
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
