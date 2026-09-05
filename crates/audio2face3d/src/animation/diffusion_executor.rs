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
struct DiffusionSchedulerState {
    running: bool,
    inferences: Vec<usize>,
}

/// Stateful multi-track scheduler matching the SDK's frame-major callback order.
#[derive(Debug)]
struct DiffusionExecutionState {
    contract: DiffusionContract,
    recurrent: Mutex<DiffusionState>,
    noise: Mutex<PhiloxNoise>,
    state: Mutex<DiffusionSchedulerState>,
    idle: Condvar,
}

struct DiffusionRunningGuard<'a> {
    execution: &'a DiffusionExecutionState,
}

impl Drop for DiffusionRunningGuard<'_> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.execution.state.lock() {
            state.running = false;
            self.execution.idle.notify_all();
        }
    }
}

impl DiffusionExecutionState {
    fn new(contract: DiffusionContract, track_count: usize, seed: u64) -> Result<Self> {
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
            state: Mutex::new(DiffusionSchedulerState {
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
    fn execute<B, C>(
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
        let _running = DiffusionRunningGuard { execution: self };
        self.execute_inner(tracks, backend, &mut callback)
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
        let noise_size = self.contract.noise_size()?;
        let expected_state = self.contract.state_size()?;
        let full_batch = (0..tracks.len())
            .map(|track| {
                pending
                    .iter()
                    .find(|(pending_track, _)| *pending_track == track)
                    .map_or_else(
                        || {
                            (
                                track,
                                DiffusionFrameInput {
                                    audio: vec![0.0; self.contract.audio_size],
                                    emotions: vec![
                                        0.0;
                                        self.contract.center_frames
                                            * self.contract.emotion_size
                                    ],
                                    identity: vec![0.0; self.contract.identity_size],
                                    noise: vec![0.0; noise_size],
                                    input_latents: vec![0.0; expected_state],
                                },
                            )
                        },
                        |(_, input)| (track, input.clone()),
                    )
            })
            .collect::<Vec<_>>();
        let outputs = backend.infer_batch(&full_batch)?;
        if outputs.len() != tracks.len() {
            return Err(invalid("diffusion backend returned a non-fixed batch size"));
        }
        let expected_prediction = self
            .contract
            .total_frames()
            .checked_mul(self.contract.result_layout.total()?)
            .ok_or_else(|| invalid("diffusion prediction size overflow"))?;
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
                .map(|(track, _)| (*track, outputs[*track].output_latents.as_slice()))
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
            for (track, _) in &pending {
                if !callback_active[*track] {
                    continue;
                }
                let output = &outputs[*track];
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
                        frame: inference * self.contract.center_frames + frame,
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

    #[cfg(feature = "tensorrt")]
    // The device path keeps each independently-owned output allocation explicit
    // so callback lifetimes cannot be hidden behind an untracked aggregate.
    #[allow(clippy::too_many_arguments)]
    fn execute_device<C>(
        &self,
        tracks: &[DiffusionTrack<'_>],
        backend: &mut crate::animation::TensorRtDiffusionBackend,
        postprocessor: &mut crate::animation::GpuRegressionPostprocessor,
        skin: &mut crate::cuda::DeviceBuffer<f32>,
        tongue: &mut crate::cuda::DeviceBuffer<f32>,
        jaw: &mut crate::cuda::DeviceBuffer<f32>,
        eyes: &mut crate::cuda::DeviceBuffer<f32>,
        mut callback: C,
    ) -> Result<DiffusionExecutionStatus>
    where
        C: for<'a> FnMut(
            DiffusionCallbackMetadata,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::CudaStreamRef<'a>,
        ) -> bool,
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
        let _running = DiffusionRunningGuard { execution: self };
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

        let noise_size = self.contract.noise_size()?;
        let expected_state = self.contract.state_size()?;
        let full_batch = (0..tracks.len())
            .map(|track| {
                pending
                    .iter()
                    .find(|(pending_track, _)| *pending_track == track)
                    .map_or_else(
                        || {
                            (
                                track,
                                DiffusionFrameInput {
                                    audio: vec![0.0; self.contract.audio_size],
                                    emotions: vec![
                                        0.0;
                                        self.contract.center_frames
                                            * self.contract.emotion_size
                                    ],
                                    identity: vec![0.0; self.contract.identity_size],
                                    noise: vec![0.0; noise_size],
                                    input_latents: vec![0.0; expected_state],
                                },
                            )
                        },
                        |(_, input)| (track, input.clone()),
                    )
            })
            .collect::<Vec<_>>();
        let device_batch = backend.run_device_batch(&full_batch)?;
        if device_batch.state_output.len() != tracks.len()
            || device_batch
                .state_output
                .iter()
                .any(|state| state.len() != expected_state)
        {
            return Err(invalid("diffusion backend output dimensions mismatch"));
        }
        {
            let updates = pending
                .iter()
                .map(|(track, _)| (*track, device_batch.state_output[*track].as_slice()))
                .collect::<Vec<_>>();
            self.recurrent
                .lock()
                .map_err(|_| poisoned())?
                .commit_tracks(&updates)?;
        }

        let result_size = self.contract.result_layout.total()?;
        let input_stride = self
            .contract
            .total_frames()
            .checked_mul(result_size)
            .ok_or_else(|| invalid("diffusion prediction size overflow"))?;
        let stream = backend.stream();
        let mut callback_active = vec![false; tracks.len()];
        for (track, _) in &pending {
            callback_active[*track] = true;
        }
        for frame_offset in 0..self.contract.center_frames {
            for (track, _) in &pending {
                if !callback_active[*track] {
                    continue;
                }
                let inference = inference_indices[*track];
                let frame = inference * self.contract.center_frames + frame_offset;
                let window = self.contract.frame_progress.window(frame)?;
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
                let prediction_frame = self.contract.left_frames + frame_offset;
                let fence = postprocessor.enqueue_view(
                    device_batch.prediction.result_tensor(),
                    input_stride,
                    prediction_frame * result_size,
                    std::slice::from_ref(track),
                    crate::animation::GpuRegressionOutputs {
                        skin,
                        tongue,
                        jaw_transforms: jaw,
                        eyes_rotations: eyes,
                    },
                    stream,
                )?;
                fence.synchronize()?;
                drop(fence);
                let skin_view = skin.view().slice(
                    *track * self.contract.result_layout.skin,
                    self.contract.result_layout.skin,
                )?;
                let tongue_view = tongue.view().slice(
                    *track * self.contract.result_layout.tongue,
                    self.contract.result_layout.tongue,
                )?;
                let jaw_view = jaw.view().slice(*track * 16, 16)?;
                let eyes_view = eyes.view().slice(*track * 6, 6)?;
                if !callback(
                    DiffusionCallbackMetadata {
                        track: *track,
                        inference,
                        frame,
                        timestamp: window.target,
                        next_timestamp: self.contract.frame_progress.window(frame + 1)?.target,
                    },
                    skin_view,
                    tongue_view,
                    jaw_view,
                    eyes_view,
                    stream.as_ref(),
                ) {
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

    fn reset(&self, track: usize) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| poisoned())?;
        if state.running {
            return Err(Error::InvalidState {
                operation: "reset diffusion track",
                state: "execution is running",
            });
        }
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

    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    fn next_inference_index(&self, track: usize) -> Result<usize> {
        let state = self.state.lock().map_err(|_| poisoned())?;
        state
            .inferences
            .get(track)
            .copied()
            .ok_or_else(|| invalid("diffusion track is out of range"))
    }

    fn wait(&self) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| poisoned())?;
        while state.running {
            state = self.idle.wait(state).map_err(|_| poisoned())?;
        }
        Ok(())
    }
}

/// Internal static-dispatch Diffusion execution.
///
/// The concrete backend and post-processor stay behind the non-generic
/// facade. The legacy [`DiffusionExecutor`] alias is retained until Step 7.
#[derive(Debug)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct DiffusionGeometryExecution<B, P> {
    state: DiffusionExecutionState,
    backend: B,
    postprocessor: P,
}

/// Legacy low-level scheduler retained until the Step 7 API removal.
#[derive(Debug)]
pub struct DiffusionExecutor {
    inner: DiffusionGeometryExecution<(), ()>,
}

impl DiffusionExecutor {
    pub fn new(contract: DiffusionContract, track_count: usize, seed: u64) -> Result<Self> {
        Ok(Self {
            inner: DiffusionGeometryExecution::with_dependencies(
                contract,
                track_count,
                seed,
                (),
                (),
            )?,
        })
    }

    pub fn execute<B, C>(
        &self,
        tracks: &[DiffusionTrack<'_>],
        backend: &mut B,
        callback: C,
    ) -> Result<DiffusionExecutionStatus>
    where
        B: DiffusionBackend,
        C: FnMut(DiffusionCallbackMetadata, &[f32]) -> bool,
    {
        self.inner.state.execute(tracks, backend, callback)
    }

    #[cfg(feature = "tensorrt")]
    // Mirrors the explicit allocation ownership of the state implementation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_device<C>(
        &self,
        tracks: &[DiffusionTrack<'_>],
        backend: &mut crate::animation::TensorRtDiffusionBackend,
        postprocessor: &mut crate::animation::GpuRegressionPostprocessor,
        skin: &mut crate::cuda::DeviceBuffer<f32>,
        tongue: &mut crate::cuda::DeviceBuffer<f32>,
        jaw: &mut crate::cuda::DeviceBuffer<f32>,
        eyes: &mut crate::cuda::DeviceBuffer<f32>,
        callback: C,
    ) -> Result<DiffusionExecutionStatus>
    where
        C: for<'a> FnMut(
            DiffusionCallbackMetadata,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::CudaStreamRef<'a>,
        ) -> bool,
    {
        self.inner.state.execute_device(
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

    pub fn reset(&self, track: usize) -> Result<()> {
        self.inner.reset(track)
    }

    pub fn wait(&self) -> Result<()> {
        self.inner.wait()
    }

    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    pub(crate) fn next_inference_index(&self, track: usize) -> Result<usize> {
        self.inner.state.next_inference_index(track)
    }
}

impl<B, P> DiffusionGeometryExecution<B, P> {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_dependencies(
        contract: DiffusionContract,
        track_count: usize,
        seed: u64,
        backend: B,
        postprocessor: P,
    ) -> Result<Self> {
        Ok(Self {
            state: DiffusionExecutionState::new(contract, track_count, seed)?,
            backend,
            postprocessor,
        })
    }

    pub(crate) fn reset(&self, track: usize) -> Result<()> {
        self.state.reset(track)
    }

    pub(crate) fn wait(&self) -> Result<()> {
        self.state.wait()
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

impl<B, P> DiffusionGeometryExecution<B, P>
where
    B: DiffusionBackend,
{
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn execute_internal<C>(
        &mut self,
        tracks: &[DiffusionTrack<'_>],
        callback: C,
    ) -> Result<DiffusionExecutionStatus>
    where
        C: FnMut(DiffusionCallbackMetadata, &[f32]) -> bool,
    {
        self.state.execute(tracks, &mut self.backend, callback)
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

    struct FixedBatchTrace {
        total_frames: usize,
        result_size: usize,
        calls: usize,
        batches: Vec<Vec<(usize, DiffusionFrameInput)>>,
    }

    impl FixedBatchTrace {
        fn new(contract: &DiffusionContract) -> Self {
            Self {
                total_frames: contract.total_frames(),
                result_size: contract.result_layout.total().unwrap(),
                calls: 0,
                batches: Vec::new(),
            }
        }
    }

    impl DiffusionBackend for FixedBatchTrace {
        fn infer_batch(
            &mut self,
            inputs: &[(usize, DiffusionFrameInput)],
        ) -> Result<Vec<DiffusionInferenceOutput>> {
            self.calls += 1;
            self.batches.push(inputs.to_vec());
            Ok(inputs
                .iter()
                .map(|(_, input)| DiffusionInferenceOutput {
                    output_latents: vec![self.calls as f32; input.input_latents.len()],
                    prediction: vec![0.0; self.total_frames * self.result_size],
                })
                .collect())
        }
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
        let result_size = contract.result_layout.total().unwrap();
        let backend = |inputs: &[(usize, DiffusionFrameInput)]| {
            Ok(inputs
                .iter()
                .map(|(track, input)| DiffusionInferenceOutput {
                    output_latents: vec![*track as f32 + 1.0; input.input_latents.len()],
                    prediction: vec![*track as f32; contract.total_frames() * result_size],
                })
                .collect())
        };
        let mut execution = DiffusionGeometryExecution::with_dependencies(
            contract.clone(),
            2,
            9,
            backend,
            Vec::new(),
        )
        .unwrap();
        assert!(execution.postprocessor().is_empty());
        execution.postprocessor_mut().push("trace");
        let _ = execution.backend();
        let _ = execution.backend_mut();
        for _ in 0..2 {
            execution.execute_internal(&tracks, |_, _| true).unwrap();
        }
        let mut seen = Vec::new();
        execution
            .execute_internal(&tracks, |metadata, _| {
                seen.push(metadata);
                metadata.track != 0
            })
            .unwrap();
        assert_eq!(seen.iter().filter(|item| item.track == 0).count(), 1);
        assert!(seen.iter().filter(|item| item.track == 1).count() > 1);
        execution.wait().unwrap();
        execution.reset(0).unwrap();
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

    #[test]
    fn fixed_batch_preserves_inactive_recurrent_noise_and_cursor() {
        let contract = contract();
        let (audio0, emotion0) = accumulators();
        let audio1 = AudioAccumulator::new(4, 0).unwrap();
        let emotion1 = EmotionAccumulator::new(1, 2).unwrap();
        emotion1.accumulate(-8, &[0.0]).unwrap();
        emotion1.accumulate(12, &[1.0]).unwrap();
        let (audio2, emotion2) = accumulators();
        let audio3 = AudioAccumulator::new(4, 0).unwrap();
        let emotion3 = EmotionAccumulator::new(1, 2).unwrap();
        emotion3.accumulate(-8, &[0.0]).unwrap();
        emotion3.accumulate(12, &[1.0]).unwrap();
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
            DiffusionTrack {
                audio: &audio2,
                emotions: &emotion2,
                identity_index: 0,
                input_strength: 1.0,
            },
            DiffusionTrack {
                audio: &audio3,
                emotions: &emotion3,
                identity_index: 0,
                input_strength: 1.0,
            },
        ];
        let mut backend = FixedBatchTrace::new(&contract);
        let executor = DiffusionExecutor::new(contract.clone(), 4, 17).unwrap();

        for _ in 0..16 {
            let status = executor
                .execute(&tracks, &mut backend, |_, _| true)
                .unwrap();
            if status == DiffusionExecutionStatus::AwaitingInput {
                break;
            }
        }
        assert!(backend.batches.len() > 1);
        assert!(backend.batches.iter().all(|batch| batch.len() == 4));
        assert!(
            backend.batches[1][1]
                .1
                .input_latents
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            backend.batches[1][3]
                .1
                .input_latents
                .iter()
                .all(|value| *value == 0.0)
        );

        let control_inputs: Vec<_> = (0..4).map(|_| accumulators()).collect();
        let control_tracks: Vec<_> = control_inputs
            .iter()
            .map(|(audio, emotions)| DiffusionTrack {
                audio,
                emotions,
                identity_index: 0,
                input_strength: 1.0,
            })
            .collect();
        let mut control_backend = FixedBatchTrace::new(&contract);
        let control = DiffusionExecutor::new(contract, 4, 17).unwrap();
        control
            .execute(&control_tracks, &mut control_backend, |_, _| true)
            .unwrap();
        control
            .execute(&control_tracks, &mut control_backend, |_, _| true)
            .unwrap();
        let expected_second_noise = control_backend.batches[1][1].1.noise.clone();

        audio1.accumulate(&[1.0; 12]).unwrap();
        audio1.close().unwrap();
        emotion1.close().unwrap();
        let mut resumed = Vec::new();
        assert_eq!(
            executor
                .execute(&tracks, &mut backend, |metadata, _| {
                    resumed.push(metadata);
                    true
                })
                .unwrap(),
            DiffusionExecutionStatus::Executed { tracks: 1 }
        );
        let resumed_input = &backend.batches.last().unwrap()[1].1;
        assert!(
            resumed_input
                .input_latents
                .iter()
                .all(|value| *value == 1.0)
        );
        assert_eq!(resumed_input.noise, expected_second_noise);
        assert!(
            resumed
                .iter()
                .all(|metadata| { metadata.track == 1 && metadata.inference == 1 })
        );
    }

    #[test]
    fn callback_panic_releases_running_guard() {
        let contract = contract();
        let (audio, emotions) = accumulators();
        let track = DiffusionTrack {
            audio: &audio,
            emotions: &emotions,
            identity_index: 0,
            input_strength: 1.0,
        };
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
        let executor = DiffusionExecutor::new(contract, 1, 7).unwrap();
        let tracks = [track];
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            for _ in 0..8 {
                let _ = executor.execute(&tracks, &mut backend, |_, _| -> bool {
                    panic!("callback panic")
                });
            }
        }));
        assert!(panic.is_err());
        executor.reset(0).unwrap();
    }
}
