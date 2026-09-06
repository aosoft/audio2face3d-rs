#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::common::{
    AudioAccumulator, EmotionAccumulator, Error, Result, WindowProgress, WindowProgressParameters,
};
use crate::emotion::{
    EmotionCallbackMetadata, EmotionExecutionStatus, EmotionPostProcessData,
    EmotionPostProcessParameters, EmotionPostProcessor, InteractiveEmotionStatus,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

/// Frame scheduling contract for the inference-free Audio2Emotion executor.
///
/// Timestamps are audio sample indices. As in the original SDK, one frame is
/// generated every `sample_rate * frame_rate_denominator / frame_rate_numerator`
/// samples and the post-process input is an all-zero emotion vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PostProcessEmotionContract {
    pub sample_rate: usize,
    pub frame_rate_numerator: usize,
    pub frame_rate_denominator: usize,
    frame_progress: WindowProgress,
}

impl PostProcessEmotionContract {
    pub fn new(
        sample_rate: usize,
        mut frame_rate_numerator: usize,
        mut frame_rate_denominator: usize,
    ) -> Result<Self> {
        if sample_rate == 0 || frame_rate_numerator == 0 || frame_rate_denominator == 0 {
            return Err(invalid(
                "post-process sample rate and frame rate must be non-zero",
            ));
        }
        let divisor = gcd(frame_rate_numerator, frame_rate_denominator);
        frame_rate_numerator /= divisor;
        frame_rate_denominator /= divisor;
        let stride_numerator = sample_rate
            .checked_mul(frame_rate_denominator)
            .ok_or_else(|| invalid("post-process frame stride overflow"))?;
        let frame_progress = WindowProgress::new(WindowProgressParameters {
            window_size: 1,
            start_offset: 0,
            target_offset: 0,
            stride_numerator,
            stride_denominator: frame_rate_numerator,
        })?;
        Ok(Self {
            sample_rate,
            frame_rate_numerator,
            frame_rate_denominator,
            frame_progress,
        })
    }

    pub fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        Ok(self.frame_progress.window(frame)?.target)
    }

    pub fn frame_count(&self, audio: &AudioAccumulator) -> Result<usize> {
        if !audio.is_closed() {
            return Ok(0);
        }
        self.frame_progress.available_windows(
            i64::try_from(audio.nb_accumulated_samples()).unwrap_or(i64::MAX),
            true,
        )
    }

    pub fn fixed_dt(&self) -> f32 {
        self.frame_rate_denominator as f32 / self.frame_rate_numerator as f32
    }
}

const fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

/// Inputs for one post-process-only track.
pub(crate) struct PostProcessEmotionTrack<'a> {
    /// Audio is used only for availability and duration; its samples are never read.
    pub audio: &'a AudioAccumulator,
    /// Optional time-varying preferred emotion values.
    pub preferred_emotions: Option<&'a EmotionAccumulator>,
}

/// Streaming Audio2Emotion executor that performs no classifier inference.
///
/// The executor advances one frame per ready track on each call to
/// [`execute`](Self::execute). It feeds an all-zero inference vector into the
/// existing post-processor and therefore supports manual/preferred-emotion
/// animation without a TensorRT engine.
pub(crate) struct PostProcessEmotionExecutor {
    contract: PostProcessEmotionContract,
    processors: Vec<EmotionPostProcessor>,
    frame_indices: Vec<usize>,
    zero_inference: Vec<f32>,
    input_strength: f32,
}

impl PostProcessEmotionExecutor {
    pub fn new(
        contract: PostProcessEmotionContract,
        data: EmotionPostProcessData,
        mut parameters: EmotionPostProcessParameters,
        track_count: usize,
    ) -> Result<Self> {
        if track_count == 0 {
            return Err(invalid("post-process emotion track count must be non-zero"));
        }
        // The original executor derives fixedDt from its runtime frame rate.
        parameters.fixed_dt = contract.fixed_dt();
        let processor = EmotionPostProcessor::new(data.clone(), parameters)?;
        Ok(Self {
            contract,
            processors: vec![processor; track_count],
            frame_indices: vec![0; track_count],
            zero_inference: vec![0.0; data.inference_emotion_length],
            input_strength: 1.0,
        })
    }

    pub fn output_emotion_length(&self) -> usize {
        self.processors[0].data().output_emotion_length
    }

    pub fn set_input_strength(&mut self, input_strength: f32) -> Result<()> {
        if !input_strength.is_finite() {
            return Err(invalid("post-process input strength must be finite"));
        }
        if self.frame_indices.iter().any(|frame| *frame != 0) {
            return Err(invalid(
                "post-process input strength cannot change after execution starts",
            ));
        }
        self.input_strength = input_strength;
        Ok(())
    }

    pub fn has_execution_started(&self, track: usize) -> bool {
        self.frame_indices
            .get(track)
            .is_some_and(|frame| *frame != 0)
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn next_frame_index(&self, track: usize) -> Result<usize> {
        self.frame_indices
            .get(track)
            .copied()
            .ok_or_else(|| invalid("post-process frame track is out of range"))
    }

    #[cfg(test)]
    pub fn parameters(&self, track: usize) -> Result<&EmotionPostProcessParameters> {
        self.processors
            .get(track)
            .map(EmotionPostProcessor::parameters)
            .ok_or_else(|| invalid("post-process parameter track is out of range"))
    }

    #[cfg(test)]
    pub fn set_parameters(
        &mut self,
        track: usize,
        parameters: EmotionPostProcessParameters,
    ) -> Result<()> {
        // The original post-process executor permits live updates. Existing
        // temporal state is retained; call reset first for a clean replay.
        self.processors
            .get_mut(track)
            .ok_or_else(|| invalid("post-process parameter track is out of range"))?
            .set_parameters(parameters)
    }

    pub fn reset(&mut self, track: usize) -> Result<()> {
        let frame = self
            .frame_indices
            .get_mut(track)
            .ok_or_else(|| invalid("post-process reset track is out of range"))?;
        *frame = 0;
        self.processors[track].reset();
        Ok(())
    }

    pub fn execute<C>(
        &mut self,
        tracks: &[PostProcessEmotionTrack<'_>],
        mut callback: C,
    ) -> Result<EmotionExecutionStatus>
    where
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        if tracks.len() != self.processors.len() {
            return Err(invalid(
                "post-process emotion executor track count mismatch",
            ));
        }

        let mut pending = Vec::new();
        let mut incomplete = false;
        for (track, inputs) in tracks.iter().enumerate() {
            let frame = self.frame_indices[track];
            let timestamp = self.contract.frame_timestamp(frame)?;
            let audio_end =
                i64::try_from(inputs.audio.nb_accumulated_samples()).unwrap_or(i64::MAX);
            if timestamp >= audio_end {
                if !inputs.audio.is_closed() {
                    incomplete = true;
                }
                continue;
            }
            incomplete = true;

            let preferred = if self.processors[track].parameters().enable_preferred_emotion {
                validate_preferred_dimension(
                    inputs.preferred_emotions,
                    self.output_emotion_length(),
                )?;
                match inputs.preferred_emotions {
                    Some(accumulator) => match preferred_at(accumulator, timestamp)? {
                        Some(value) => Some(value),
                        None => continue,
                    },
                    None => None,
                }
            } else {
                None
            };
            pending.push((track, frame, timestamp, preferred));
        }

        if pending.is_empty() {
            return Ok(if incomplete {
                EmotionExecutionStatus::AwaitingInput
            } else {
                EmotionExecutionStatus::Complete
            });
        }

        let mut frames = 0;
        for (track, frame, timestamp, preferred) in &pending {
            let next_frame = frame
                .checked_add(1)
                .ok_or_else(|| invalid("post-process frame index overflow"))?;
            let output = self.processors[*track]
                .process_with_preferred(&self.zero_inference, preferred.as_deref())?;
            frames += 1;
            // The original post-process executor produces one frame per
            // execution. A false return is therefore track-local but has no
            // later frame in this call to suppress; other ready tracks still
            // receive their frame.
            let _ = callback(
                EmotionCallbackMetadata {
                    track: *track,
                    frame: *frame,
                    timestamp: *timestamp,
                    next_timestamp: self.contract.frame_timestamp(next_frame)?,
                },
                &output,
            );
            self.frame_indices[*track] = next_frame;
        }
        Ok(EmotionExecutionStatus::Executed {
            tracks: pending.len(),
            frames,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PostProcessEmotionLayer {
    None,
    Inference,
    PostProcessing,
    All,
}

/// Cloneable interrupt signal for an interactive computation.
#[derive(Clone, Debug)]
pub(crate) struct PostProcessEmotionInterrupt {
    requested: Arc<AtomicBool>,
}

impl PostProcessEmotionInterrupt {
    pub fn interrupt(&self) {
        self.requested.store(true, Ordering::Release);
    }
}

/// Single-track random-access post-process-only executor.
///
/// There is no neural inference cache: the inference layer represents a
/// deterministic all-zero input. Frame replay still starts at frame zero so
/// temporal smoothing state exactly matches sequential execution.
pub(crate) struct InteractivePostProcessEmotionExecutor {
    contract: PostProcessEmotionContract,
    data: EmotionPostProcessData,
    parameters: EmotionPostProcessParameters,
    inference_valid: bool,
    post_processing_valid: bool,
    cached_audio_length: Option<usize>,
    interrupt: PostProcessEmotionInterrupt,
}

impl InteractivePostProcessEmotionExecutor {
    pub fn new(
        contract: PostProcessEmotionContract,
        data: EmotionPostProcessData,
        mut parameters: EmotionPostProcessParameters,
    ) -> Result<Self> {
        parameters.fixed_dt = contract.fixed_dt();
        EmotionPostProcessor::new(data.clone(), parameters.clone())?;
        Ok(Self {
            contract,
            data,
            parameters,
            inference_valid: false,
            post_processing_valid: false,
            cached_audio_length: None,
            interrupt: PostProcessEmotionInterrupt {
                requested: Arc::new(AtomicBool::new(false)),
            },
        })
    }

    #[cfg(test)]
    pub fn parameters(&self) -> &EmotionPostProcessParameters {
        &self.parameters
    }

    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn output_emotion_length(&self) -> usize {
        self.data.output_emotion_length
    }

    #[cfg(test)]
    pub fn set_parameters(&mut self, parameters: EmotionPostProcessParameters) -> Result<()> {
        EmotionPostProcessor::new(self.data.clone(), parameters.clone())?;
        if self.parameters != parameters {
            self.parameters = parameters;
            self.invalidate(PostProcessEmotionLayer::PostProcessing);
        }
        Ok(())
    }

    pub fn invalidate(&mut self, layer: PostProcessEmotionLayer) {
        match layer {
            PostProcessEmotionLayer::None => {}
            PostProcessEmotionLayer::Inference => {
                self.inference_valid = false;
                self.post_processing_valid = false;
                self.cached_audio_length = None;
            }
            PostProcessEmotionLayer::PostProcessing => {
                self.post_processing_valid = false;
            }
            PostProcessEmotionLayer::All => {
                self.inference_valid = false;
                self.post_processing_valid = false;
                self.cached_audio_length = None;
            }
        }
    }

    pub fn is_valid(&self, layer: PostProcessEmotionLayer) -> bool {
        match layer {
            PostProcessEmotionLayer::None => true,
            PostProcessEmotionLayer::Inference => self.inference_valid,
            PostProcessEmotionLayer::PostProcessing => self.post_processing_valid,
            PostProcessEmotionLayer::All => self.inference_valid && self.post_processing_valid,
        }
    }

    pub fn interrupt_handle(&self) -> PostProcessEmotionInterrupt {
        self.interrupt.clone()
    }

    pub fn frame_count(&self, audio: &AudioAccumulator) -> Result<usize> {
        validate_interactive_audio(audio)?;
        self.contract.frame_count(audio)
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
        let frame_count = self.prepare(audio, preferred)?;
        self.compute_range(preferred, frame_count, None, callback)
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
        let frame_count = self.prepare(audio, preferred)?;
        if frame >= frame_count {
            return Err(invalid(
                "interactive post-process emotion frame is out of range",
            ));
        }
        self.compute_range(
            preferred,
            frame
                .checked_add(1)
                .ok_or_else(|| invalid("interactive post-process frame index overflow"))?,
            Some(frame),
            callback,
        )
    }

    fn prepare(
        &mut self,
        audio: &AudioAccumulator,
        preferred: Option<&EmotionAccumulator>,
    ) -> Result<usize> {
        validate_interactive_audio(audio)?;
        if self.parameters.enable_preferred_emotion {
            validate_interactive_preferred(preferred, self.data.output_emotion_length)?;
        }
        let audio_length = audio.nb_accumulated_samples();
        if self.cached_audio_length != Some(audio_length) {
            self.invalidate(PostProcessEmotionLayer::Inference);
            self.cached_audio_length = Some(audio_length);
        }
        // The inference result is a deterministic zero vector.
        self.inference_valid = true;
        self.contract.frame_count(audio)
    }

    fn compute_range<C>(
        &mut self,
        preferred: Option<&EmotionAccumulator>,
        end_frame: usize,
        callback_frame: Option<usize>,
        mut callback: C,
    ) -> Result<InteractiveEmotionStatus>
    where
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        self.interrupt.requested.store(false, Ordering::Release);
        let mut processor = EmotionPostProcessor::new(self.data.clone(), self.parameters.clone())?;
        let zero_inference = vec![0.0; self.data.inference_emotion_length];
        let mut emitted = 0;
        for frame in 0..end_frame {
            if self.interrupt.requested.load(Ordering::Acquire) {
                self.post_processing_valid = false;
                return Ok(InteractiveEmotionStatus::Interrupted { frames: emitted });
            }
            let timestamp = self.contract.frame_timestamp(frame)?;
            let selected_preferred = if self.parameters.enable_preferred_emotion {
                match preferred {
                    Some(accumulator) => Some(accumulator.read(timestamp).map_err(|error| {
                        invalid(format!(
                            "interactive preferred emotion read failed: {error}"
                        ))
                    })?),
                    None => None,
                }
            } else {
                None
            };
            let output =
                processor.process_with_preferred(&zero_inference, selected_preferred.as_deref())?;
            if callback_frame.is_none_or(|selected| selected == frame) {
                emitted += 1;
                if !callback(
                    EmotionCallbackMetadata {
                        track: 0,
                        frame,
                        timestamp,
                        next_timestamp: self.contract.frame_timestamp(
                            frame.checked_add(1).ok_or_else(|| {
                                invalid("interactive post-process frame index overflow")
                            })?,
                        )?,
                    },
                    &output,
                ) {
                    self.post_processing_valid = false;
                    return Ok(InteractiveEmotionStatus::Interrupted { frames: emitted });
                }
            }
        }
        self.post_processing_valid = callback_frame.is_none();
        Ok(InteractiveEmotionStatus::Complete { frames: emitted })
    }
}

fn validate_preferred_dimension(
    preferred: Option<&EmotionAccumulator>,
    output_length: usize,
) -> Result<()> {
    if preferred.is_some_and(|value| value.state().emotion_size != output_length) {
        Err(invalid("preferred emotion dimensions differ"))
    } else {
        Ok(())
    }
}

fn preferred_at(accumulator: &EmotionAccumulator, timestamp: i64) -> Result<Option<Vec<f32>>> {
    let state = accumulator.state();
    if state.key_count == 0 || (!state.closed && timestamp > state.last_accumulated_timestamp) {
        return Ok(None);
    }
    if timestamp < state.last_dropped_timestamp {
        return Err(invalid(
            "preferred emotions required for execution were dropped",
        ));
    }
    accumulator
        .read(timestamp)
        .map(Some)
        .map_err(|error| invalid(format!("preferred emotion read failed: {error}")))
}

fn validate_interactive_audio(audio: &AudioAccumulator) -> Result<()> {
    if !audio.is_closed() || audio.nb_dropped_samples() != 0 {
        Err(invalid(
            "interactive post-process audio must be closed and complete",
        ))
    } else {
        Ok(())
    }
}

fn validate_interactive_preferred(
    preferred: Option<&EmotionAccumulator>,
    output_length: usize,
) -> Result<()> {
    let Some(preferred) = preferred else {
        return Ok(());
    };
    let state = preferred.state();
    if state.emotion_size != output_length
        || !state.closed
        || state.dropped_emotions != 0
        || state.last_dropped_timestamp != i64::MIN
    {
        return Err(invalid(
            "interactive preferred emotions must be closed, complete, and dimensionally valid",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (
        PostProcessEmotionContract,
        EmotionPostProcessData,
        EmotionPostProcessParameters,
        AudioAccumulator,
        EmotionAccumulator,
    ) {
        let contract = PostProcessEmotionContract::new(8, 4, 1).unwrap();
        let data = EmotionPostProcessData {
            inference_emotion_length: 2,
            output_emotion_length: 2,
            emotion_correspondence: vec![0, 1],
        };
        let parameters = EmotionPostProcessParameters {
            emotion_contrast: 1.0,
            max_emotions: 2,
            beginning_emotion: vec![0.0; 2],
            preferred_emotion: vec![0.0; 2],
            live_blend_coefficient: 0.0,
            enable_preferred_emotion: true,
            preferred_emotion_strength: 0.5,
            live_transition_time: 0.0,
            fixed_dt: 99.0,
            emotion_strength: 1.0,
        };
        let audio = AudioAccumulator::new(8, 0).unwrap();
        audio.accumulate(&[0.0; 16]).unwrap();
        audio.close().unwrap();
        let preferred = EmotionAccumulator::new(2, 8).unwrap();
        preferred.accumulate(0, &[1.0, 0.0]).unwrap();
        preferred.accumulate(14, &[0.0, 1.0]).unwrap();
        preferred.close().unwrap();
        (contract, data, parameters, audio, preferred)
    }

    #[test]
    fn normal_and_interactive_outputs_have_frame_parity() {
        let (contract, data, parameters, audio, preferred) = setup();
        let mut normal =
            PostProcessEmotionExecutor::new(contract.clone(), data.clone(), parameters.clone(), 1)
                .unwrap();
        assert_eq!(normal.parameters(0).unwrap().fixed_dt, 0.25);
        let tracks = [PostProcessEmotionTrack {
            audio: &audio,
            preferred_emotions: Some(&preferred),
        }];
        let mut normal_frames = Vec::new();
        loop {
            match normal
                .execute(&tracks, |metadata, output| {
                    normal_frames.push((metadata, output.to_vec()));
                    true
                })
                .unwrap()
            {
                EmotionExecutionStatus::Complete => break,
                EmotionExecutionStatus::Executed {
                    tracks: 1,
                    frames: 1,
                } => {}
                status => panic!("unexpected status: {status:?}"),
            }
        }

        let mut interactive =
            InteractivePostProcessEmotionExecutor::new(contract, data, parameters).unwrap();
        let mut interactive_frames = Vec::new();
        assert_eq!(
            interactive
                .compute_all(&audio, Some(&preferred), |metadata, output| {
                    interactive_frames.push((metadata, output.to_vec()));
                    true
                })
                .unwrap(),
            InteractiveEmotionStatus::Complete { frames: 8 }
        );
        assert_eq!(normal_frames, interactive_frames);
        assert!(interactive.is_valid(PostProcessEmotionLayer::Inference));
        assert!(interactive.is_valid(PostProcessEmotionLayer::PostProcessing));
    }

    #[test]
    fn compute_frame_replays_state_and_only_emits_selected_frame() {
        let (contract, data, parameters, audio, preferred) = setup();
        let mut executor =
            InteractivePostProcessEmotionExecutor::new(contract, data, parameters).unwrap();
        let mut frame = Vec::new();
        assert_eq!(
            executor
                .compute_frame(3, &audio, Some(&preferred), |metadata, output| {
                    frame.push((metadata.frame, metadata.timestamp, output.to_vec()));
                    true
                })
                .unwrap(),
            InteractiveEmotionStatus::Complete { frames: 1 }
        );
        assert_eq!(frame.len(), 1);
        assert_eq!((frame[0].0, frame[0].1), (3, 6));
        assert!(executor.is_valid(PostProcessEmotionLayer::Inference));
        assert!(!executor.is_valid(PostProcessEmotionLayer::PostProcessing));
    }

    #[test]
    fn rational_frame_rate_uses_audio_sample_timestamps() {
        let contract = PostProcessEmotionContract::new(16_000, 30_000, 1_001).unwrap();
        assert_eq!(contract.frame_timestamp(0).unwrap(), 0);
        assert_eq!(contract.frame_timestamp(1).unwrap(), 533);
        assert_eq!(contract.frame_timestamp(2).unwrap(), 1_067);
        assert!((contract.fixed_dt() - 1_001.0 / 30_000.0).abs() < f32::EPSILON);
    }

    #[test]
    fn multi_track_waits_for_each_tracks_preferred_emotion() {
        let (contract, data, parameters, audio, preferred) = setup();
        let waiting = EmotionAccumulator::new(2, 8).unwrap();
        let mut executor = PostProcessEmotionExecutor::new(contract, data, parameters, 2).unwrap();
        let tracks = [
            PostProcessEmotionTrack {
                audio: &audio,
                preferred_emotions: Some(&preferred),
            },
            PostProcessEmotionTrack {
                audio: &audio,
                preferred_emotions: Some(&waiting),
            },
        ];
        let mut called = Vec::new();
        assert_eq!(
            executor
                .execute(&tracks, |metadata, _| {
                    called.push(metadata.track);
                    true
                })
                .unwrap(),
            EmotionExecutionStatus::Executed {
                tracks: 1,
                frames: 1
            }
        );
        assert_eq!(called, [0]);

        waiting.accumulate(0, &[0.0, 1.0]).unwrap();
        waiting.close().unwrap();
        let mut called_frames = Vec::new();
        assert!(matches!(
            executor
                .execute(&tracks, |metadata, _| {
                    called_frames.push((metadata.track, metadata.frame));
                    true
                })
                .unwrap(),
            EmotionExecutionStatus::Executed {
                tracks: 2,
                frames: 2
            }
        ));
        assert_eq!(called_frames, [(0, 1), (1, 0)]);
    }

    #[test]
    fn arbitrary_fps_normal_and_interactive_metadata_match() {
        let contract = PostProcessEmotionContract::new(16_000, 30_000, 1_001).unwrap();
        let data = EmotionPostProcessData {
            inference_emotion_length: 1,
            output_emotion_length: 1,
            emotion_correspondence: vec![0],
        };
        let parameters = EmotionPostProcessParameters {
            max_emotions: 1,
            beginning_emotion: vec![0.0],
            preferred_emotion: vec![0.0],
            live_blend_coefficient: 0.0,
            live_transition_time: 0.0,
            emotion_strength: 1.0,
            ..EmotionPostProcessParameters::default()
        };
        let audio = AudioAccumulator::new(16_000, 0).unwrap();
        audio.accumulate(&[0.0; 1_602]).unwrap();
        audio.close().unwrap();
        let tracks = [PostProcessEmotionTrack {
            audio: &audio,
            preferred_emotions: None,
        }];
        let mut normal =
            PostProcessEmotionExecutor::new(contract.clone(), data.clone(), parameters.clone(), 1)
                .unwrap();
        let mut normal_metadata = Vec::new();
        loop {
            let status = normal
                .execute(&tracks, |metadata, _| {
                    normal_metadata.push(metadata);
                    true
                })
                .unwrap();
            if status == EmotionExecutionStatus::Complete {
                break;
            }
        }
        let mut interactive =
            InteractivePostProcessEmotionExecutor::new(contract, data, parameters).unwrap();
        let mut interactive_metadata = Vec::new();
        assert_eq!(
            interactive
                .compute_all(&audio, None, |metadata, _| {
                    interactive_metadata.push(metadata);
                    true
                })
                .unwrap(),
            InteractiveEmotionStatus::Complete { frames: 4 }
        );
        assert_eq!(normal_metadata, interactive_metadata);
        assert_eq!(
            normal_metadata
                .iter()
                .map(|metadata| metadata.timestamp)
                .collect::<Vec<_>>(),
            [0, 533, 1_067, 1_601]
        );
    }

    #[test]
    fn parameter_update_and_reset_are_track_local() {
        let contract = PostProcessEmotionContract::new(1, 1, 1).unwrap();
        let data = EmotionPostProcessData {
            inference_emotion_length: 1,
            output_emotion_length: 1,
            emotion_correspondence: vec![0],
        };
        let parameters = EmotionPostProcessParameters {
            max_emotions: 1,
            beginning_emotion: vec![0.0],
            preferred_emotion: vec![0.0],
            live_blend_coefficient: 0.0,
            live_transition_time: 0.0,
            emotion_strength: 1.0,
            ..EmotionPostProcessParameters::default()
        };
        let audio = AudioAccumulator::new(1, 0).unwrap();
        audio.accumulate(&[0.0; 3]).unwrap();
        audio.close().unwrap();
        let tracks = [
            PostProcessEmotionTrack {
                audio: &audio,
                preferred_emotions: None,
            },
            PostProcessEmotionTrack {
                audio: &audio,
                preferred_emotions: None,
            },
        ];
        let mut executor =
            PostProcessEmotionExecutor::new(contract, data, parameters.clone(), 2).unwrap();
        executor.execute(&tracks, |_, _| true).unwrap();
        let mut changed = parameters;
        changed.emotion_strength = 0.5;
        executor.set_parameters(0, changed).unwrap();
        let mut results = Vec::new();
        executor
            .execute(&tracks, |metadata, output| {
                results.push((metadata.track, metadata.frame, output[0]));
                false
            })
            .unwrap();
        assert_eq!(results, [(0, 1, 0.5), (1, 1, 1.0)]);

        executor.reset(0).unwrap();
        results.clear();
        executor
            .execute(&tracks, |metadata, output| {
                results.push((metadata.track, metadata.frame, output[0]));
                true
            })
            .unwrap();
        assert_eq!(results, [(0, 0, 0.5), (1, 2, 1.0)]);
    }

    #[test]
    fn interactive_parameter_invalidation_callback_stop_and_interrupt() {
        let (contract, data, parameters, audio, preferred) = setup();
        let mut executor =
            InteractivePostProcessEmotionExecutor::new(contract, data, parameters).unwrap();
        executor
            .compute_all(&audio, Some(&preferred), |_, _| true)
            .unwrap();
        let mut changed = executor.parameters().clone();
        changed.emotion_strength = 0.5;
        executor.set_parameters(changed.clone()).unwrap();
        assert!(executor.is_valid(PostProcessEmotionLayer::Inference));
        assert!(!executor.is_valid(PostProcessEmotionLayer::PostProcessing));

        let mut invalid = changed;
        invalid.preferred_emotion.clear();
        assert!(executor.set_parameters(invalid).is_err());
        assert_eq!(executor.parameters().emotion_strength, 0.5);

        assert_eq!(
            executor
                .compute_all(&audio, Some(&preferred), |_, _| false)
                .unwrap(),
            InteractiveEmotionStatus::Interrupted { frames: 1 }
        );
        assert!(!executor.is_valid(PostProcessEmotionLayer::PostProcessing));

        let interrupt = executor.interrupt_handle();
        assert_eq!(
            executor
                .compute_all(&audio, Some(&preferred), |_, _| {
                    interrupt.interrupt();
                    true
                })
                .unwrap(),
            InteractiveEmotionStatus::Interrupted { frames: 1 }
        );
        assert_eq!(
            executor
                .compute_frame(0, &audio, Some(&preferred), |_, _| true)
                .unwrap(),
            InteractiveEmotionStatus::Complete { frames: 1 }
        );
        assert!(executor.is_valid(PostProcessEmotionLayer::None));
        executor.invalidate(PostProcessEmotionLayer::All);
        assert!(!executor.is_valid(PostProcessEmotionLayer::Inference));
        assert!(!executor.is_valid(PostProcessEmotionLayer::PostProcessing));
    }

    #[test]
    fn interactive_rejects_any_preferred_emotion_drop_marker() {
        let (contract, data, parameters, audio, preferred) = setup();
        preferred.drop_before(0).unwrap();
        assert_eq!(preferred.state().dropped_emotions, 0);
        let mut executor =
            InteractivePostProcessEmotionExecutor::new(contract, data, parameters).unwrap();
        assert!(
            executor
                .compute_all(&audio, Some(&preferred), |_, _| true)
                .is_err()
        );
    }
}
