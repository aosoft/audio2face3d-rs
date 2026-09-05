//! Post-process-only Audio2Emotion facade declarations.

use std::sync::Arc;

use std::convert::Infallible;
#[cfg(feature = "cuda")]
use std::ops::ControlFlow;

use crate::audio2emotion::{
    EmotionExecutorCreationParameters, EmotionInteractiveExecutorCreationParameters,
};
use crate::audio2x::{EmotionAccumulator, FrameRate};

/// Immutable dimensions and correspondence used by emotion post-processing.
///
/// Corresponds to `nva2e::PostProcessData` in `audio2emotion/postprocess.h`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PostProcessData {
    pub inference_emotion_length: usize,
    pub output_emotion_length: usize,
    pub emotion_correspondence: Vec<i32>,
}

/// Runtime-tunable emotion post-processing parameters.
///
/// Corresponds to `nva2e::PostProcessParams` in
/// `audio2emotion/postprocess.h`.
#[derive(Clone, Debug, PartialEq)]
pub struct PostProcessParams {
    pub emotion_contrast: f32,
    pub max_emotions: usize,
    pub beginning_emotion: Vec<f32>,
    pub preferred_emotion: Vec<f32>,
    pub live_blend_coefficient: f32,
    pub enable_preferred_emotion: bool,
    pub preferred_emotion_strength: f32,
    pub live_transition_time: f32,
    pub fixed_dt: f32,
    pub emotion_strength: f32,
}

impl Default for PostProcessParams {
    fn default() -> Self {
        Self {
            emotion_contrast: 1.0,
            max_emotions: 0,
            beginning_emotion: Vec::new(),
            preferred_emotion: Vec::new(),
            live_blend_coefficient: 0.7,
            enable_preferred_emotion: false,
            preferred_emotion_strength: 0.5,
            live_transition_time: 0.5,
            fixed_dt: 0.033,
            emotion_strength: 0.6,
        }
    }
}

/// Parameters for loading a post-process-only emotion executor.
///
/// Corresponds to `nva2e::IPostProcessModel::EmotionExecutorCreationParameters`
/// in `audio2emotion-sdk/include/audio2emotion/executor_postprocess.h`, combined
/// with the owned common resources from
/// `nva2e::EmotionExecutorCreationParameters`.
pub struct PostProcessEmotionExecutorCreationParameters {
    pub common: EmotionExecutorCreationParameters,
    pub sample_rate: usize,
    pub input_strength: f32,
    pub frame_rate: FrameRate,
    pub post_process_data: PostProcessData,
    pub post_process_params: PostProcessParams,
    pub preferred_emotions: Vec<Arc<EmotionAccumulator>>,
}

/// Parameters for an interactive post-process-only emotion executor.
///
/// The common input timeline must be closed and retain all required history.
/// `batch_size` fixes the maximum frame batch used by interactive computation.
///
/// Corresponds to the arguments of
/// `nva2e::CreatePostProcessEmotionInteractiveExecutor` in
/// `audio2emotion-sdk/include/audio2emotion/audio2emotion.h`.
pub struct PostProcessEmotionInteractiveExecutorCreationParameters {
    pub common: EmotionInteractiveExecutorCreationParameters,
    pub sample_rate: usize,
    pub input_strength: f32,
    pub frame_rate: FrameRate,
    pub batch_size: usize,
    pub post_process_data: PostProcessData,
    pub post_process_params: PostProcessParams,
}

/// Opaque host emotion post-processor component.
///
/// Corresponds to `nva2e::IPostProcessor` in
/// `audio2emotion-sdk/include/audio2emotion/postprocess.h` and replaces
/// `crate::emotion::EmotionPostProcessor` at the SDK-facing module boundary.
pub struct PostProcessor {
    _opaque: Infallible,
    _not_sync: std::cell::Cell<()>,
}

/// Non-generic owning post-process-only emotion executor facade.
///
/// Corresponds to `nva2e::IPostProcessModel::EmotionExecutor` and implements
/// `nva2e::IEmotionExecutor` from
/// `audio2emotion-sdk/include/audio2emotion/executor.h`. It is the SDK-facing
/// completed replacement for `crate::emotion::PostProcessEmotionExecutor`.
#[cfg(feature = "cuda")]
pub struct PostProcessEmotionExecutor {
    execution: crate::emotion::PostProcessEmotionExecutor,
    contract: crate::emotion::PostProcessEmotionContract,
    tracks: Vec<crate::audio2emotion::EmotionTrackResources>,
    preferred_emotions: Vec<Arc<EmotionAccumulator>>,
    frame_rate: FrameRate,
    output: crate::cuda::DeviceBuffer<f32>,
    stream: crate::cuda::CudaStream,
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}

/// Non-generic owning interactive post-process-only emotion executor facade.
///
/// Corresponds to `nva2e::IPostProcessModel::EmotionInteractiveExecutor` and
/// `nva2e::IEmotionInteractiveExecutor` from
/// `audio2emotion-sdk/include/audio2emotion/interactive_executor.h`. It replaces
/// `crate::emotion::InteractivePostProcessEmotionExecutor` at the facade
/// boundary.
#[cfg(feature = "cuda")]
pub struct PostProcessEmotionInteractiveExecutor {
    _opaque: Infallible,
    _not_sync: std::cell::Cell<()>,
}

#[cfg(feature = "cuda")]
impl PostProcessEmotionExecutor {
    /// Creates an owning post-process-only executor and its device result stream.
    pub fn load(parameters: PostProcessEmotionExecutorCreationParameters) -> crate::Result<Self> {
        let track_count = parameters.common.tracks.len();
        if track_count == 0 {
            return Err(crate::Error::InvalidArgument {
                field: "tracks",
                reason: "at least one track is required".into(),
            });
        }
        if !parameters.preferred_emotions.is_empty()
            && parameters.preferred_emotions.len() != track_count
        {
            return Err(crate::Error::SizeMismatch {
                field: "preferred_emotions",
                expected: track_count,
                actual: parameters.preferred_emotions.len(),
            });
        }
        let contract = crate::emotion::PostProcessEmotionContract::new(
            parameters.sample_rate,
            parameters.frame_rate.numerator(),
            parameters.frame_rate.denominator(),
        )?;
        let data = into_internal_data(parameters.post_process_data);
        let output_len = data.output_emotion_length;
        let mut execution = crate::emotion::PostProcessEmotionExecutor::new(
            contract.clone(),
            data,
            into_internal_params(parameters.post_process_params),
            track_count,
        )?;
        execution.set_input_strength(parameters.input_strength)?;
        let device = crate::cuda::GpuDevice::new(parameters.common.device_ordinal)?;
        let stream = device.create_stream()?;
        let output = device.allocate(output_len)?;
        Ok(Self {
            execution,
            contract,
            tracks: parameters.common.tracks,
            preferred_emotions: parameters.preferred_emotions,
            frame_rate: parameters.frame_rate,
            output,
            stream,
            _not_sync: std::marker::PhantomData,
        })
    }

    fn audio(&self, track: usize) -> crate::Result<&Arc<crate::audio2x::AudioAccumulator>> {
        self.tracks
            .get(track)
            .map(|resources| &resources.audio)
            .ok_or(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            })
    }
}

#[cfg(feature = "cuda")]
impl crate::audio2x::Executor for PostProcessEmotionExecutor {
    fn track_count(&self) -> usize {
        self.tracks.len()
    }

    fn reset_track(&mut self, track: usize) -> crate::Result<()> {
        if self.audio(track)?.nb_dropped_samples() != 0
            || self
                .preferred_emotions
                .get(track)
                .is_some_and(|emotions| emotions.state().dropped_emotions != 0)
        {
            return Err(crate::Error::InputHistoryUnavailable { track });
        }
        self.execution.reset(track)
    }

    fn has_execution_started(&self, track: usize) -> crate::Result<bool> {
        if track >= self.tracks.len() {
            return Err(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            });
        }
        Ok(self.execution.has_execution_started(track))
    }

    fn available_execution_count(&self, track: usize) -> crate::Result<usize> {
        let audio = self.audio(track)?;
        if !audio.is_closed() {
            return Ok(0);
        }
        Ok(self
            .contract
            .frame_count(audio)?
            .saturating_sub(self.execution.next_frame_index(track)?))
    }

    fn ready_track_count(&self) -> usize {
        self.tracks
            .iter()
            .enumerate()
            .filter(|(track, resources)| {
                self.execution
                    .next_frame_index(*track)
                    .and_then(|frame| self.contract.frame_timestamp(frame))
                    .is_ok_and(|timestamp| {
                        timestamp
                            < i64::try_from(resources.audio.nb_accumulated_samples())
                                .unwrap_or(i64::MAX)
                    })
            })
            .count()
    }

    fn total_frame_count(&self, track: usize) -> crate::Result<Option<usize>> {
        let audio = self.audio(track)?;
        if audio.is_closed() {
            self.contract.frame_count(audio).map(Some)
        } else {
            Ok(None)
        }
    }

    fn sample_rate(&self) -> usize {
        self.contract.sample_rate
    }

    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    fn frame_timestamp(&self, frame: usize) -> crate::Result<i64> {
        self.contract.frame_timestamp(frame)
    }
}

#[cfg(feature = "cuda")]
impl crate::audio2emotion::EmotionExecutor for PostProcessEmotionExecutor {
    fn emotion_count(&self) -> usize {
        self.execution.output_emotion_length()
    }

    fn next_audio_sample_to_read(&self, track: usize) -> crate::Result<usize> {
        self.audio(track)?;
        Ok(self
            .contract
            .frame_timestamp(self.execution.next_frame_index(track)?)?
            .max(0) as usize)
    }

    fn execute(
        &mut self,
        callback: &mut dyn for<'r> FnMut(
            crate::audio2emotion::EmotionResults<'r>,
        ) -> ControlFlow<()>,
    ) -> crate::Result<crate::audio2x::Execution> {
        let tracks = self
            .tracks
            .iter()
            .enumerate()
            .map(
                |(track, resources)| crate::emotion::PostProcessEmotionTrack {
                    audio: &resources.audio,
                    preferred_emotions: self.preferred_emotions.get(track).map(Arc::as_ref),
                },
            )
            .collect::<Vec<_>>();
        let output = &mut self.output;
        let stream = &self.stream;
        let mut emitted_frames = 0;
        let mut callback_error = None;
        let status = self.execution.execute(&tracks, |metadata, values| {
            if callback_error.is_some() {
                return false;
            }
            if let Err(error) = output.copy_from(values, stream) {
                callback_error = Some(error);
                return false;
            }
            emitted_frames += 1;
            matches!(
                callback(crate::audio2emotion::EmotionResults {
                    metadata: crate::audio2x::CallbackMetadata {
                        track_index: metadata.track,
                        frame_index: metadata.frame,
                        timestamp: metadata.timestamp,
                        next_timestamp: metadata.next_timestamp,
                    },
                    emotions: crate::audio2x::DeviceComponentResults {
                        values: output.view(),
                        stream: stream.as_ref(),
                    },
                }),
                ControlFlow::Continue(())
            )
        })?;
        if let Some(error) = callback_error {
            return Err(error);
        }
        let (state, executed_tracks) = match status {
            crate::emotion::EmotionExecutionStatus::AwaitingInput => {
                (crate::audio2x::ExecutionState::AwaitingInput, 0)
            }
            crate::emotion::EmotionExecutionStatus::Complete => {
                (crate::audio2x::ExecutionState::Complete, 0)
            }
            crate::emotion::EmotionExecutionStatus::Executed { tracks, .. } => {
                (crate::audio2x::ExecutionState::Progress, tracks)
            }
        };
        Ok(crate::audio2x::Execution::ready(
            crate::audio2x::ExecutionReport {
                state,
                executed_tracks,
                emitted_frames,
            },
        ))
    }
}

#[cfg(feature = "cuda")]
fn into_internal_data(data: PostProcessData) -> crate::emotion::EmotionPostProcessData {
    crate::emotion::EmotionPostProcessData {
        inference_emotion_length: data.inference_emotion_length,
        output_emotion_length: data.output_emotion_length,
        emotion_correspondence: data
            .emotion_correspondence
            .into_iter()
            .map(i64::from)
            .collect(),
    }
}

#[cfg(feature = "cuda")]
fn into_internal_params(params: PostProcessParams) -> crate::emotion::EmotionPostProcessParameters {
    crate::emotion::EmotionPostProcessParameters {
        emotion_contrast: params.emotion_contrast,
        max_emotions: params.max_emotions,
        beginning_emotion: params.beginning_emotion,
        preferred_emotion: params.preferred_emotion,
        live_blend_coefficient: params.live_blend_coefficient,
        enable_preferred_emotion: params.enable_preferred_emotion,
        preferred_emotion_strength: params.preferred_emotion_strength,
        live_transition_time: params.live_transition_time,
        fixed_dt: params.fixed_dt,
        emotion_strength: params.emotion_strength,
    }
}
