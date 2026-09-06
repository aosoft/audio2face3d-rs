#![cfg_attr(not(feature = "tensorrt"), allow(dead_code))]

use crate::common::{AudioAccumulator, EmotionAccumulator, Error, Result};
use crate::emotion::{
    ClassifierBackend, ClassifierContract, EmotionCallbackMetadata, EmotionPostProcessData,
    EmotionPostProcessParameters, EmotionPostProcessor,
};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractiveEmotionStatus {
    Complete { frames: usize },
    Interrupted { frames: usize },
}

/// Internal classifier execution with separate inference and post-process caches.
pub(crate) struct ClassifierInteractiveExecution<B> {
    backend: B,
    contract: ClassifierContract,
    data: EmotionPostProcessData,
    parameters: EmotionPostProcessParameters,
    inference_cache: Vec<Vec<f32>>,
    inference_valid: bool,
    input_strength: f32,
    cached_audio_length: Option<usize>,
    batch_size: usize,
}

impl<B: ClassifierBackend> ClassifierInteractiveExecution<B> {
    pub fn new(
        backend: B,
        contract: ClassifierContract,
        data: EmotionPostProcessData,
        parameters: EmotionPostProcessParameters,
    ) -> Result<Self> {
        if contract.emotion_length != data.inference_emotion_length {
            return Err(invalid(
                "interactive classifier and post-process dimensions differ",
            ));
        }
        EmotionPostProcessor::new(data.clone(), parameters.clone())?;
        Ok(Self {
            backend,
            contract,
            data,
            parameters,
            inference_cache: Vec::new(),
            inference_valid: false,
            input_strength: 1.0,
            cached_audio_length: None,
            batch_size: 1,
        })
    }

    pub fn inference_cache_is_valid(&self) -> bool {
        self.inference_valid
    }

    pub fn invalidate_audio(&mut self) {
        self.inference_valid = false;
        self.cached_audio_length = None;
    }

    pub fn set_input_strength(&mut self, strength: f32) -> Result<()> {
        if !strength.is_finite() {
            return Err(invalid("interactive input strength must be finite"));
        }
        if self.input_strength != strength {
            self.input_strength = strength;
            self.invalidate_audio();
        }
        Ok(())
    }

    pub(crate) fn set_batch_size(&mut self, batch_size: usize) -> Result<()> {
        if batch_size == 0 {
            return Err(invalid(
                "interactive classifier batch size must be non-zero",
            ));
        }
        if self
            .backend
            .max_batch_size()
            .is_some_and(|max| batch_size > max)
        {
            return Err(invalid(
                "interactive classifier batch size exceeds backend maximum",
            ));
        }
        if self.batch_size != batch_size {
            self.batch_size = batch_size;
            self.invalidate_audio();
        }
        Ok(())
    }

    /// Changes only the replay stage; cached classifier outputs remain valid.
    #[cfg(test)]
    pub fn set_parameters(&mut self, parameters: EmotionPostProcessParameters) -> Result<()> {
        EmotionPostProcessor::new(self.data.clone(), parameters.clone())?;
        self.parameters = parameters;
        Ok(())
    }

    #[cfg(test)]
    pub fn compute_all<C>(
        &mut self,
        audio: &AudioAccumulator,
        preferred: Option<&EmotionAccumulator>,
        callback: C,
    ) -> Result<InteractiveEmotionStatus>
    where
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        let frame_count = self.frame_count(audio)?;
        self.compute_range(audio, preferred, frame_count, None, callback)
    }

    pub fn compute_frame<C>(
        &mut self,
        frame: usize,
        audio: &AudioAccumulator,
        preferred: Option<&EmotionAccumulator>,
        callback: C,
    ) -> Result<InteractiveEmotionStatus>
    where
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        let frame_count = self.frame_count(audio)?;
        if frame >= frame_count {
            return Err(invalid("interactive emotion frame is out of range"));
        }
        self.compute_range(audio, preferred, frame + 1, Some(frame), callback)
    }

    pub(crate) fn emotion_count(&self) -> usize {
        self.data.output_emotion_length
    }

    pub(crate) fn frame_count(&self, audio: &AudioAccumulator) -> Result<usize> {
        validate_closed_audio(audio)?;
        self.contract.frame_progress.available_windows(
            i64::try_from(audio.nb_accumulated_samples()).unwrap_or(i64::MAX),
            true,
        )
    }

    fn compute_range<C>(
        &mut self,
        audio: &AudioAccumulator,
        preferred: Option<&EmotionAccumulator>,
        end_frame: usize,
        callback_frame: Option<usize>,
        mut callback: C,
    ) -> Result<InteractiveEmotionStatus>
    where
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        validate_preferred(preferred, self.data.output_emotion_length)?;
        let frames_per_inference = self.contract.frames_per_inference();
        let required_inferences = end_frame
            .checked_add(frames_per_inference - 1)
            .ok_or_else(|| invalid("interactive inference prefix length overflow"))?
            / frames_per_inference;
        self.ensure_inference_cache(audio, required_inferences)?;
        let mut processor = EmotionPostProcessor::new(self.data.clone(), self.parameters.clone())?;
        let mut emitted = 0;
        for frame in 0..end_frame {
            let inference = frame / self.contract.frames_per_inference();
            let logits = self
                .inference_cache
                .get(inference)
                .ok_or_else(|| invalid("interactive inference cache is incomplete"))?;
            let timestamp = self.contract.frame_timestamp(frame)?;
            let selected_preferred = match preferred {
                Some(accumulator) if self.parameters.enable_preferred_emotion => {
                    Some(accumulator.read(timestamp).map_err(|error| {
                        invalid(format!(
                            "interactive preferred emotion read failed: {error}"
                        ))
                    })?)
                }
                _ => None,
            };
            let output = processor.process_with_preferred(logits, selected_preferred.as_deref())?;
            if callback_frame.is_none_or(|selected| selected == frame) {
                emitted += 1;
                if !callback(
                    EmotionCallbackMetadata {
                        track: 0,
                        frame,
                        timestamp,
                        next_timestamp: self.contract.frame_timestamp(frame + 1)?,
                    },
                    &output,
                ) {
                    return Ok(InteractiveEmotionStatus::Interrupted { frames: emitted });
                }
            }
        }
        Ok(InteractiveEmotionStatus::Complete { frames: emitted })
    }

    fn ensure_inference_cache(
        &mut self,
        audio: &AudioAccumulator,
        required_inferences: usize,
    ) -> Result<()> {
        let audio_length = audio.nb_accumulated_samples();
        if self.cached_audio_length != Some(audio_length) {
            self.inference_cache.clear();
            self.inference_valid = false;
            self.cached_audio_length = Some(audio_length);
        }
        let available_inferences = self
            .contract
            .inference_progress
            .available_windows(i64::try_from(audio_length).unwrap_or(i64::MAX), true)?;
        let required_inferences = required_inferences.min(available_inferences);
        if self.inference_cache.len() >= required_inferences {
            self.inference_valid = self.inference_cache.len() == available_inferences;
            return Ok(());
        }
        self.inference_cache
            .reserve(required_inferences - self.inference_cache.len());
        for chunk_start in
            (self.inference_cache.len()..required_inferences).step_by(self.batch_size)
        {
            let chunk_end = (chunk_start + self.batch_size).min(required_inferences);
            let mut inputs = Vec::with_capacity(chunk_end - chunk_start);
            for inference in chunk_start..chunk_end {
                let window = self.contract.inference_progress.window(inference)?;
                inputs.push((
                    0,
                    audio.read(
                        window.start,
                        self.contract.buffer_length,
                        self.input_strength,
                    )?,
                ));
            }
            let results = self.backend.infer_batch(&inputs)?;
            if results.len() != inputs.len()
                || results
                    .iter()
                    .any(|result| result.len() != self.contract.emotion_length)
            {
                return Err(invalid("interactive classifier output dimensions differ"));
            }
            self.inference_cache.extend(results);
        }
        self.inference_valid = self.inference_cache.len() == available_inferences;
        Ok(())
    }
}

fn validate_closed_audio(audio: &AudioAccumulator) -> Result<()> {
    if !audio.is_closed() || audio.nb_dropped_samples() != 0 {
        Err(invalid(
            "interactive audio must be closed and retain every accumulated sample",
        ))
    } else {
        Ok(())
    }
}

fn validate_preferred(preferred: Option<&EmotionAccumulator>, length: usize) -> Result<()> {
    let Some(preferred) = preferred else {
        return Ok(());
    };
    let state = preferred.state();
    if state.emotion_size != length || !state.closed || state.dropped_emotions != 0 {
        return Err(invalid(
            "interactive preferred emotions must be closed, complete, and dimensionally valid",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    fn setup() -> (
        ClassifierInteractiveExecution<impl ClassifierBackend>,
        AudioAccumulator,
        Rc<Cell<usize>>,
    ) {
        let calls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&calls);
        let backend = move |_track: usize, input: &[f32]| {
            observed.set(observed.get() + 1);
            Ok(vec![input[input.len() / 2], 0.0])
        };
        let contract = ClassifierContract::new(8, 8, 2, 4, 1, 1).unwrap();
        let data = EmotionPostProcessData {
            inference_emotion_length: 2,
            output_emotion_length: 2,
            emotion_correspondence: vec![0, 1],
        };
        let parameters = EmotionPostProcessParameters {
            max_emotions: 2,
            beginning_emotion: vec![0.0; 2],
            preferred_emotion: vec![0.0; 2],
            live_blend_coefficient: 0.0,
            live_transition_time: 0.0,
            emotion_strength: 1.0,
            ..EmotionPostProcessParameters::default()
        };
        let audio = AudioAccumulator::new(4, 0).unwrap();
        audio.accumulate(&[1.0; 16]).unwrap();
        audio.close().unwrap();
        (
            ClassifierInteractiveExecution::new(backend, contract, data, parameters).unwrap(),
            audio,
            calls,
        )
    }

    #[test]
    fn frame_replay_reuses_inference_and_parameter_change_only_replays_postprocess() {
        let (mut executor, audio, calls) = setup();
        let mut selected = Vec::new();
        executor
            .compute_frame(2, &audio, None, |metadata, _| {
                selected.push(metadata.frame);
                true
            })
            .unwrap();
        assert_eq!(selected, [2]);
        let initial_calls = calls.get();
        let mut parameters = executor.parameters.clone();
        parameters.emotion_strength = 0.5;
        executor.set_parameters(parameters).unwrap();
        executor
            .compute_frame(2, &audio, None, |_, _| true)
            .unwrap();
        assert_eq!(calls.get(), initial_calls);
        executor.set_input_strength(0.5).unwrap();
        executor
            .compute_frame(2, &audio, None, |_, _| true)
            .unwrap();
        assert!(calls.get() > initial_calls);
    }

    #[test]
    fn callback_false_interrupts_only_requested_computation() {
        let (mut executor, audio, _) = setup();
        assert!(matches!(
            executor.compute_all(&audio, None, |_, _| false).unwrap(),
            InteractiveEmotionStatus::Interrupted { frames: 1 }
        ));
        assert!(matches!(
            executor
                .compute_frame(0, &audio, None, |_, _| true)
                .unwrap(),
            InteractiveEmotionStatus::Complete { frames: 1 }
        ));
    }

    #[test]
    fn frame_replay_builds_only_the_required_inference_prefix() {
        let (mut executor, audio, calls) = setup();
        executor
            .compute_frame(0, &audio, None, |_, _| true)
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert!(!executor.inference_cache_is_valid());

        // Frame 1 reuses inference 0. Frame 2 requires inference 1 and appends
        // only that missing suffix; asking for frame 0 never runs inference 1.
        executor
            .compute_frame(1, &audio, None, |_, _| true)
            .unwrap();
        assert_eq!(calls.get(), 1);
        executor
            .compute_frame(2, &audio, None, |_, _| true)
            .unwrap();
        assert_eq!(calls.get(), 2);
        executor
            .compute_frame(0, &audio, None, |_, _| true)
            .unwrap();
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn batched_and_scalar_prefix_caches_have_identical_outputs() {
        let (mut scalar, audio, scalar_calls) = setup();
        let (mut batched, _, batched_calls) = setup();
        batched.set_batch_size(2).unwrap();
        let mut scalar_output = Vec::new();
        let mut batched_output = Vec::new();
        scalar
            .compute_all(&audio, None, |metadata, output| {
                scalar_output.push((metadata.frame, output.to_vec()));
                true
            })
            .unwrap();
        batched
            .compute_all(&audio, None, |metadata, output| {
                batched_output.push((metadata.frame, output.to_vec()));
                true
            })
            .unwrap();
        assert_eq!(scalar_output, batched_output);
        assert_eq!(scalar_calls.get(), batched_calls.get());
        assert!(scalar.inference_cache_is_valid());
        assert!(batched.inference_cache_is_valid());
    }
}
