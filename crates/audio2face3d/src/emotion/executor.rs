#![cfg_attr(not(feature = "tensorrt"), allow(dead_code))]

use crate::common::{
    AudioAccumulator, EmotionAccumulator, Error, Result, WindowProgress, WindowProgressParameters,
};
use crate::emotion::{EmotionPostProcessData, EmotionPostProcessParameters, EmotionPostProcessor};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClassifierContract {
    pub buffer_length: usize,
    pub sample_rate: usize,
    pub emotion_length: usize,
    pub frame_rate_numerator: usize,
    pub frame_rate_denominator: usize,
    pub inferences_to_skip: usize,
    pub inference_progress: WindowProgress,
    pub frame_progress: WindowProgress,
}

impl ClassifierContract {
    pub fn new(
        buffer_length: usize,
        sample_rate: usize,
        emotion_length: usize,
        frame_rate_numerator: usize,
        frame_rate_denominator: usize,
        inferences_to_skip: usize,
    ) -> Result<Self> {
        if buffer_length == 0
            || sample_rate == 0
            || emotion_length == 0
            || frame_rate_numerator == 0
            || frame_rate_denominator == 0
        {
            return Err(invalid("classifier contract dimensions must be non-zero"));
        }
        let frames_per_inference = inferences_to_skip
            .checked_add(1)
            .ok_or_else(|| invalid("classifier inference skip overflow"))?;
        let frame_stride = sample_rate
            .checked_mul(frame_rate_denominator)
            .ok_or_else(|| invalid("classifier frame stride overflow"))?;
        let inference_stride = frame_stride
            .checked_mul(frames_per_inference)
            .ok_or_else(|| invalid("classifier inference stride overflow"))?;
        if inference_stride
            > buffer_length
                .checked_mul(frame_rate_numerator)
                .ok_or_else(|| invalid("classifier window limit overflow"))?
        {
            return Err(invalid(
                "classifier stride including skipped inferences exceeds window size",
            ));
        }
        let target_offset = i64::try_from(buffer_length / 2)
            .map_err(|_| invalid("classifier target offset overflow"))?;
        let base = WindowProgressParameters {
            window_size: buffer_length,
            start_offset: -target_offset,
            target_offset,
            stride_numerator: frame_stride,
            stride_denominator: frame_rate_numerator,
        };
        let inference_progress = WindowProgress::new(WindowProgressParameters {
            stride_numerator: inference_stride,
            ..base
        })?;
        Ok(Self {
            buffer_length,
            sample_rate,
            emotion_length,
            frame_rate_numerator,
            frame_rate_denominator,
            inferences_to_skip,
            inference_progress,
            frame_progress: WindowProgress::new(base)?,
        })
    }

    pub fn frames_per_inference(&self) -> usize {
        self.inferences_to_skip + 1
    }

    pub fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        Ok(self.frame_progress.window(frame)?.target)
    }
}

pub(crate) trait ClassifierBackend {
    /// Maximum batch accepted by the backend, when it has a finite profile.
    fn max_batch_size(&self) -> Option<usize> {
        None
    }

    fn infer(&mut self, track: usize, audio: &[f32]) -> Result<Vec<f32>>;

    fn infer_batch(&mut self, inputs: &[(usize, Vec<f32>)]) -> Result<Vec<Vec<f32>>> {
        inputs
            .iter()
            .map(|(track, audio)| self.infer(*track, audio))
            .collect()
    }
}

impl<F> ClassifierBackend for F
where
    F: FnMut(usize, &[f32]) -> Result<Vec<f32>>,
{
    fn infer(&mut self, track: usize, audio: &[f32]) -> Result<Vec<f32>> {
        self(track, audio)
    }
}

pub(crate) struct EmotionTrack<'a> {
    pub audio: &'a AudioAccumulator,
    pub preferred_emotions: Option<&'a EmotionAccumulator>,
    pub input_strength: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmotionCallbackMetadata {
    pub track: usize,
    pub frame: usize,
    pub timestamp: i64,
    pub next_timestamp: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmotionExecutionStatus {
    AwaitingInput,
    Executed { tracks: usize, frames: usize },
    Complete,
}

struct ClassifierExecutionState {
    contract: ClassifierContract,
    processors: Vec<EmotionPostProcessor>,
    inference_indices: Vec<usize>,
}

impl ClassifierExecutionState {
    fn new(
        contract: ClassifierContract,
        data: EmotionPostProcessData,
        parameters: EmotionPostProcessParameters,
        track_count: usize,
    ) -> Result<Self> {
        if track_count == 0 {
            return Err(invalid("emotion track count must be non-zero"));
        }
        if contract.emotion_length != data.inference_emotion_length {
            return Err(invalid(
                "classifier output and post-process input dimensions differ",
            ));
        }
        let processor = EmotionPostProcessor::new(data, parameters)?;
        Ok(Self {
            contract,
            processors: vec![processor; track_count],
            inference_indices: vec![0; track_count],
        })
    }

    fn output_emotion_length(&self) -> usize {
        self.processors[0].data().output_emotion_length
    }

    fn has_execution_started(&self, track: usize) -> bool {
        self.inference_indices
            .get(track)
            .is_some_and(|index| *index != 0)
    }

    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    fn next_inference_index(&self, track: usize) -> Result<usize> {
        self.inference_indices
            .get(track)
            .copied()
            .ok_or_else(|| invalid("emotion track is out of range"))
    }

    fn reset(&mut self, track: usize) -> Result<()> {
        let index = self
            .inference_indices
            .get_mut(track)
            .ok_or_else(|| invalid("emotion reset track is out of range"))?;
        *index = 0;
        self.processors[track].reset();
        Ok(())
    }

    fn execute<B, C>(
        &mut self,
        tracks: &[EmotionTrack<'_>],
        backend: &mut B,
        mut callback: C,
    ) -> Result<EmotionExecutionStatus>
    where
        B: ClassifierBackend,
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        if tracks.len() != self.processors.len() {
            return Err(invalid("emotion executor track count mismatch"));
        }
        if let Some(maximum) = backend.max_batch_size()
            && self.processors.len() > maximum
        {
            return Err(invalid(format!(
                "emotion track count {} exceeds classifier engine maximum batch size {maximum}",
                self.processors.len()
            )));
        }
        let mut pending = Vec::new();
        let mut incomplete = false;
        for (track_index, track) in tracks.iter().enumerate() {
            if !track.input_strength.is_finite() {
                return Err(invalid("emotion input strength must be finite"));
            }
            let inference_index = self.inference_indices[track_index];
            let window = self.contract.inference_progress.window(inference_index)?;
            let audio_end = i64::try_from(track.audio.nb_accumulated_samples()).unwrap_or(i64::MAX);
            if track.audio.is_closed() && window.target >= audio_end {
                continue;
            }
            incomplete = true;
            let available = self
                .contract
                .inference_progress
                .available_windows(audio_end, track.audio.is_closed())?;
            if inference_index >= available {
                continue;
            }
            let audio = track.audio.read(
                window.start,
                self.contract.buffer_length,
                track.input_strength,
            )?;
            pending.push((track_index, inference_index, audio));
        }
        if pending.is_empty() {
            return Ok(if incomplete {
                EmotionExecutionStatus::AwaitingInput
            } else {
                EmotionExecutionStatus::Complete
            });
        }
        let backend_inputs = pending
            .iter()
            .map(|(track, _, audio)| (*track, audio.clone()))
            .collect::<Vec<_>>();
        let logits = backend.infer_batch(&backend_inputs)?;
        if logits.len() != pending.len()
            || logits
                .iter()
                .any(|output| output.len() != self.contract.emotion_length)
        {
            return Err(invalid(
                "classifier backend returned invalid output dimensions",
            ));
        }

        let pending = pending
            .into_iter()
            .zip(logits)
            .map(|((track, inference, _), logits)| (track, inference, logits))
            .collect::<Vec<_>>();
        let mut callback_frames = 0;
        let mut callback_active = vec![true; pending.len()];
        for offset in 0..self.contract.frames_per_inference() {
            for (pending_index, (track, inference, logits)) in pending.iter().enumerate() {
                if !callback_active[pending_index] {
                    continue;
                }
                let first_frame = inference * self.contract.frames_per_inference();
                let frame = first_frame + offset;
                let timestamp = self.contract.frame_timestamp(frame)?;
                if timestamp
                    >= i64::try_from(tracks[*track].audio.nb_accumulated_samples())
                        .unwrap_or(i64::MAX)
                {
                    callback_active[pending_index] = false;
                    continue;
                }
                let preferred = match tracks[*track].preferred_emotions {
                    Some(accumulator)
                        if self.processors[*track]
                            .parameters()
                            .enable_preferred_emotion =>
                    {
                        Some(accumulator.read(timestamp).map_err(|error| {
                            invalid(format!("preferred emotion read failed: {error}"))
                        })?)
                    }
                    _ => None,
                };
                let output =
                    self.processors[*track].process_with_preferred(logits, preferred.as_deref())?;
                callback_frames += 1;
                if !callback(
                    EmotionCallbackMetadata {
                        track: *track,
                        frame,
                        timestamp,
                        next_timestamp: self.contract.frame_timestamp(frame + 1)?,
                    },
                    &output,
                ) {
                    callback_active[pending_index] = false;
                }
            }
        }
        for (track, _, _) in &pending {
            self.inference_indices[*track] += 1;
        }
        Ok(EmotionExecutionStatus::Executed {
            tracks: backend_inputs.len(),
            frames: callback_frames,
        })
    }
}

/// Internal classifier execution with an owned backend.
///
/// The concrete `B` remains an implementation detail.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct ClassifierExecution<B> {
    state: ClassifierExecutionState,
    backend: B,
}

pub(crate) struct ClassifierScheduler {
    inner: ClassifierExecution<()>,
}

impl ClassifierScheduler {
    pub fn new(
        contract: ClassifierContract,
        data: EmotionPostProcessData,
        parameters: EmotionPostProcessParameters,
        track_count: usize,
    ) -> Result<Self> {
        Ok(Self {
            inner: ClassifierExecution::with_backend(contract, data, parameters, track_count, ())?,
        })
    }

    #[cfg(test)]
    pub fn execute<B, C>(
        &mut self,
        tracks: &[EmotionTrack<'_>],
        backend: &mut B,
        callback: C,
    ) -> Result<EmotionExecutionStatus>
    where
        B: ClassifierBackend,
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        self.inner.state.execute(tracks, backend, callback)
    }

    #[cfg(feature = "tensorrt")]
    pub(crate) fn execute_device<C>(
        &mut self,
        tracks: &[EmotionTrack<'_>],
        backend: &mut crate::emotion::TensorRtClassifierBackend,
        processor: &mut crate::emotion::GpuEmotionPostProcessor,
        output: &mut crate::cuda::DeviceBuffer<f32>,
        mut callback: C,
    ) -> Result<EmotionExecutionStatus>
    where
        C: for<'a> FnMut(
            EmotionCallbackMetadata,
            crate::cuda::DeviceView<'a, f32>,
            crate::cuda::CudaStreamRef<'a>,
        ) -> bool,
    {
        let state = &mut self.inner.state;
        if tracks.len() != state.processors.len() || processor.track_count() != tracks.len() {
            return Err(invalid("emotion executor track count mismatch"));
        }
        backend.validate_track_count(tracks.len())?;
        let output_stride = state.output_emotion_length();
        let expected_output = output_stride
            .checked_mul(tracks.len())
            .ok_or_else(|| invalid("emotion device output size overflow"))?;
        if output.len() != expected_output {
            return Err(invalid("emotion device output dimensions mismatch"));
        }

        let mut pending = Vec::new();
        let mut incomplete = false;
        for (track_index, track) in tracks.iter().enumerate() {
            if !track.input_strength.is_finite() {
                return Err(invalid("emotion input strength must be finite"));
            }
            let inference_index = state.inference_indices[track_index];
            let window = state.contract.inference_progress.window(inference_index)?;
            let audio_end = i64::try_from(track.audio.nb_accumulated_samples()).unwrap_or(i64::MAX);
            if track.audio.is_closed() && window.target >= audio_end {
                continue;
            }
            incomplete = true;
            let available = state
                .contract
                .inference_progress
                .available_windows(audio_end, track.audio.is_closed())?;
            if inference_index >= available {
                continue;
            }
            let audio = track.audio.read(
                window.start,
                state.contract.buffer_length,
                track.input_strength,
            )?;
            pending.push((track_index, inference_index, audio));
        }
        if pending.is_empty() {
            return Ok(if incomplete {
                EmotionExecutionStatus::AwaitingInput
            } else {
                EmotionExecutionStatus::Complete
            });
        }

        let backend_inputs = pending
            .iter()
            .map(|(track, _, audio)| (*track, audio.clone()))
            .collect::<Vec<_>>();
        let logits = backend.run_device_batch(&backend_inputs)?;
        let input_stride = state.contract.emotion_length;
        if logits.len() != input_stride.saturating_mul(pending.len()) {
            return Err(invalid(
                "classifier backend returned invalid device output dimensions",
            ));
        }

        let mut callback_frames = 0;
        let mut callback_active = vec![true; pending.len()];
        for offset in 0..state.contract.frames_per_inference() {
            for (pending_index, (track, inference, _)) in pending.iter().enumerate() {
                if !callback_active[pending_index] {
                    continue;
                }
                let first_frame = inference * state.contract.frames_per_inference();
                let frame = first_frame + offset;
                let timestamp = state.contract.frame_timestamp(frame)?;
                if timestamp
                    >= i64::try_from(tracks[*track].audio.nb_accumulated_samples())
                        .unwrap_or(i64::MAX)
                {
                    callback_active[pending_index] = false;
                    continue;
                }
                if processor.preferred_enabled(*track)?
                    && let Some(accumulator) = tracks[*track].preferred_emotions
                {
                    let preferred = accumulator.read(timestamp).map_err(|error| {
                        invalid(format!("preferred emotion read failed: {error}"))
                    })?;
                    processor.set_preferred(*track, &preferred, backend.stream())?;
                }
                let input = logits
                    .view()
                    .slice(pending_index * input_stride, input_stride)?;
                let fence = processor.enqueue_view(
                    input,
                    input_stride,
                    output,
                    output_stride,
                    std::slice::from_ref(track),
                    backend.stream(),
                )?;
                fence.synchronize()?;
                drop(fence);

                let values = output.view().slice(*track * output_stride, output_stride)?;
                callback_frames += 1;
                if !callback(
                    EmotionCallbackMetadata {
                        track: *track,
                        frame,
                        timestamp,
                        next_timestamp: state.contract.frame_timestamp(frame + 1)?,
                    },
                    values,
                    backend.stream().as_ref(),
                ) {
                    callback_active[pending_index] = false;
                }
            }
        }
        for (track, _, _) in &pending {
            state.inference_indices[*track] += 1;
        }
        Ok(EmotionExecutionStatus::Executed {
            tracks: backend_inputs.len(),
            frames: callback_frames,
        })
    }

    pub fn output_emotion_length(&self) -> usize {
        self.inner.output_emotion_length()
    }

    pub fn has_execution_started(&self, track: usize) -> bool {
        self.inner.has_execution_started(track)
    }

    pub fn reset(&mut self, track: usize) -> Result<()> {
        self.inner.reset(track)
    }

    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    pub(crate) fn next_inference_index(&self, track: usize) -> Result<usize> {
        self.inner.state.next_inference_index(track)
    }
}

impl<B> ClassifierExecution<B> {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_backend(
        contract: ClassifierContract,
        data: EmotionPostProcessData,
        parameters: EmotionPostProcessParameters,
        track_count: usize,
        backend: B,
    ) -> Result<Self> {
        Ok(Self {
            state: ClassifierExecutionState::new(contract, data, parameters, track_count)?,
            backend,
        })
    }

    pub(crate) fn output_emotion_length(&self) -> usize {
        self.state.output_emotion_length()
    }

    pub(crate) fn has_execution_started(&self, track: usize) -> bool {
        self.state.has_execution_started(track)
    }

    pub(crate) fn reset(&mut self, track: usize) -> Result<()> {
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
}

impl<B> ClassifierExecution<B>
where
    B: ClassifierBackend,
{
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn execute_internal<C>(
        &mut self,
        tracks: &[EmotionTrack<'_>],
        callback: C,
    ) -> Result<EmotionExecutionStatus>
    where
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        self.state.execute(tracks, &mut self.backend, callback)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract(skip: usize) -> ClassifierContract {
        ClassifierContract::new(8, 8, 3, 4, 1, skip).unwrap()
    }

    fn processor_data() -> (EmotionPostProcessData, EmotionPostProcessParameters) {
        (
            EmotionPostProcessData {
                inference_emotion_length: 3,
                output_emotion_length: 3,
                emotion_correspondence: vec![0, 1, 2],
            },
            EmotionPostProcessParameters {
                max_emotions: 3,
                beginning_emotion: vec![0.0; 3],
                preferred_emotion: vec![0.0; 3],
                live_blend_coefficient: 0.0,
                live_transition_time: 0.0,
                fixed_dt: 0.25,
                emotion_strength: 1.0,
                ..EmotionPostProcessParameters::default()
            },
        )
    }

    fn audio() -> AudioAccumulator {
        let audio = AudioAccumulator::new(4, 0).unwrap();
        audio.accumulate(&[1.0; 16]).unwrap();
        audio.close().unwrap();
        audio
    }

    #[test]
    fn contract_preserves_dynamic_audio_window_and_skip_stride() {
        let contract = contract(1);
        assert_eq!(contract.inference_progress.window(0).unwrap().start, -4);
        assert_eq!(contract.inference_progress.window(1).unwrap().target, 4);
        assert_eq!(contract.frame_timestamp(1).unwrap(), 2);
        assert_eq!(contract.frames_per_inference(), 2);
        assert!(ClassifierContract::new(2, 16, 3, 1, 1, 1).is_err());
    }

    #[test]
    fn executor_track_count_is_not_artificially_capped() {
        let (data, parameters) = processor_data();
        assert!(
            ClassifierScheduler::new(contract(0), data.clone(), parameters.clone(), 0).is_err()
        );
        assert!(ClassifierScheduler::new(contract(0), data, parameters, 128).is_ok());
    }

    #[test]
    fn executor_obeys_backend_batch_profile() {
        struct LimitedBackend;

        impl ClassifierBackend for LimitedBackend {
            fn max_batch_size(&self) -> Option<usize> {
                Some(1)
            }

            fn infer(&mut self, _track: usize, _audio: &[f32]) -> Result<Vec<f32>> {
                Ok(vec![0.0; 3])
            }
        }

        let (data, parameters) = processor_data();
        let mut executor = ClassifierScheduler::new(contract(0), data, parameters, 2).unwrap();
        let first = audio();
        let second = audio();
        let tracks = [
            EmotionTrack {
                audio: &first,
                preferred_emotions: None,
                input_strength: 1.0,
            },
            EmotionTrack {
                audio: &second,
                preferred_emotions: None,
                input_strength: 1.0,
            },
        ];
        assert!(
            executor
                .execute(&tracks, &mut LimitedBackend, |_, _| true)
                .unwrap_err()
                .to_string()
                .contains("maximum batch size 1")
        );
    }

    #[test]
    fn executor_batches_tracks_and_generates_skipped_frames() {
        #[derive(Default)]
        struct BackendTrace {
            calls: Vec<(usize, f32)>,
        }

        impl ClassifierBackend for BackendTrace {
            fn infer(&mut self, track: usize, input: &[f32]) -> Result<Vec<f32>> {
                self.calls.push((track, input[4]));
                Ok(vec![1.0, 0.0, -1.0])
            }
        }

        let (data, parameters) = processor_data();
        let mut execution = ClassifierExecution::with_backend(
            contract(1),
            data,
            parameters,
            2,
            BackendTrace::default(),
        )
        .unwrap();
        let first = audio();
        let second = audio();
        let tracks = [
            EmotionTrack {
                audio: &first,
                preferred_emotions: None,
                input_strength: 1.0,
            },
            EmotionTrack {
                audio: &second,
                preferred_emotions: None,
                input_strength: 0.5,
            },
        ];
        assert!(execution.backend().calls.is_empty());
        execution.backend_mut().calls.reserve(2);
        let mut metadata = Vec::new();
        let status = execution
            .execute_internal(&tracks, |frame, output| {
                assert_eq!(output.len(), 3);
                metadata.push(frame);
                true
            })
            .unwrap();
        assert_eq!(
            status,
            EmotionExecutionStatus::Executed {
                tracks: 2,
                frames: 4
            }
        );
        assert_eq!(execution.backend().calls, [(0, 1.0), (1, 0.5)]);
        assert_eq!(
            metadata
                .iter()
                .map(|value| (value.frame, value.track))
                .collect::<Vec<_>>(),
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        );
    }

    #[test]
    fn callback_stop_is_track_local_to_one_execution() {
        let (data, parameters) = processor_data();
        let mut executor = ClassifierScheduler::new(contract(1), data, parameters, 2).unwrap();
        let first = audio();
        let second = audio();
        let tracks = [
            EmotionTrack {
                audio: &first,
                preferred_emotions: None,
                input_strength: 1.0,
            },
            EmotionTrack {
                audio: &second,
                preferred_emotions: None,
                input_strength: 1.0,
            },
        ];
        let mut backend = |_track: usize, _input: &[f32]| Ok(vec![0.0; 3]);
        let mut callbacks = vec![0; 2];
        executor
            .execute(&tracks, &mut backend, |metadata, _| {
                callbacks[metadata.track] += 1;
                metadata.track != 0
            })
            .unwrap();
        assert_eq!(callbacks, [1, 2]);
    }
}
