//! Classifier-model Audio2Emotion facade declarations.

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "tensorrt")]
use std::cell::Cell;
#[cfg(feature = "tensorrt")]
use std::convert::Infallible;
#[cfg(feature = "tensorrt")]
use std::ops::ControlFlow;

use crate::audio2emotion::{
    EmotionExecutorCreationParameters, EmotionInteractiveExecutorCreationParameters,
    PostProcessData, PostProcessParams,
};
use crate::audio2x::{EmotionAccumulator, FrameRate};

#[cfg(feature = "tensorrt")]
use crate::common::NetworkDocument;
#[cfg(feature = "tensorrt")]
use crate::cuda::GpuDevice;
#[cfg(feature = "tensorrt")]
use crate::emotion::{
    ClassifierContract, EmotionCallbackMetadata, EmotionExecutor, EmotionTrack,
    TensorRtClassifierBackend,
};
#[cfg(feature = "tensorrt")]
use crate::{Model, ModelKind, ModelParameters};

/// Network dimensions read from an Audio2Emotion classifier model.
///
/// Corresponds to `nva2e::IClassifierModel::NetworkInfo` in
/// `audio2emotion-sdk/include/audio2emotion/model.h`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NetworkInfo {
    pub audio_buffer_length: usize,
    pub sample_rate: usize,
    pub emotion_length: usize,
}

/// Parameters for loading an owning classifier emotion executor.
///
/// Corresponds to `nva2e::IClassifierModel::EmotionExecutorCreationParameters`
/// in `audio2emotion-sdk/include/audio2emotion/executor_classifier.h`, combined
/// with the owned common resources from
/// `nva2e::EmotionExecutorCreationParameters`.
pub struct ClassifierEmotionExecutorCreationParameters {
    pub model_path: PathBuf,
    pub common: EmotionExecutorCreationParameters,
    pub input_strength: f32,
    /// Number of samples in one classifier inference window.
    pub buffer_length: usize,
    pub frame_rate: FrameRate,
    pub inferences_to_skip: usize,
    pub post_process_data: PostProcessData,
    pub post_process_params: PostProcessParams,
    pub preferred_emotions: Vec<Arc<EmotionAccumulator>>,
}

/// Parameters for loading an owning interactive classifier emotion executor.
///
/// `batch_size` controls the number of ready frames packed into an inference;
/// it is not a standard-executor track count. The interactive input timeline
/// must be closed and preserve all history.
///
/// Corresponds to the arguments of
/// `nva2e::CreateClassifierEmotionInteractiveExecutor` in
/// `audio2emotion-sdk/include/audio2emotion/audio2emotion.h`.
pub struct ClassifierEmotionInteractiveExecutorCreationParameters {
    pub model_path: PathBuf,
    pub common: EmotionInteractiveExecutorCreationParameters,
    pub input_strength: f32,
    pub frame_rate: FrameRate,
    pub inferences_to_skip: usize,
    pub batch_size: usize,
    pub post_process_data: PostProcessData,
    pub post_process_params: PostProcessParams,
}

/// Non-generic owning classifier emotion executor facade.
///
/// Corresponds to `nva2e::IClassifierModel::EmotionExecutor` and implements
/// `nva2e::IEmotionExecutor` from
/// `audio2emotion-sdk/include/audio2emotion/executor.h`. It replaces the current
/// generic `crate::emotion::EmotionExecutor<B>` with an owning facade whose
/// classifier backend is private.
#[cfg(feature = "tensorrt")]
pub struct ClassifierEmotionExecutor {
    execution: EmotionExecutor,
    backend: TensorRtClassifierBackend,
    contract: ClassifierContract,
    tracks: Vec<crate::audio2emotion::EmotionTrackResources>,
    preferred_emotions: Vec<Arc<EmotionAccumulator>>,
    input_strength: f32,
    frame_rate: FrameRate,
    sample_rate: usize,
    device_output: crate::cuda::DeviceBuffer<f32>,
    result_stream: crate::cuda::CudaStream,
}

/// Non-generic owning interactive classifier emotion executor facade.
///
/// Corresponds to `nva2e::IClassifierModel::EmotionInteractiveExecutor` and
/// `nva2e::IEmotionInteractiveExecutor` from
/// `audio2emotion-sdk/include/audio2emotion/interactive_executor.h`. It replaces
/// the current generic `crate::emotion::InteractiveEmotionExecutor<B>`.
#[cfg(feature = "tensorrt")]
pub struct ClassifierEmotionInteractiveExecutor {
    _opaque: Infallible,
    _not_sync: Cell<()>,
}

#[cfg(feature = "tensorrt")]
impl ClassifierEmotionExecutor {
    /// Loads an owning classifier executor. The model, TensorRT backend, and
    /// post-processing state are all moved into this value; no `Model` borrow
    /// is retained after construction.
    pub fn load(parameters: ClassifierEmotionExecutorCreationParameters) -> crate::Result<Self> {
        let model = Model::load(&parameters.model_path)?;
        if model.kind() != ModelKind::Emotion {
            return Err(crate::Error::InvalidSchema(
                "classifier executor requires an emotion model".into(),
            ));
        }
        let NetworkDocument::Emotion(network) = model.network() else {
            return Err(crate::Error::InvalidSchema(
                "emotion network is missing".into(),
            ));
        };
        let config = match model.parameters(0)? {
            ModelParameters::Emotion(value) => value,
            _ => {
                return Err(crate::Error::InvalidSchema(
                    "emotion post-process config is missing".into(),
                ));
            }
        };
        let (data, model_parameters) =
            crate::emotion::EmotionPostProcessData::from_model(network, config)?;
        if parameters.buffer_length == 0 {
            return Err(crate::Error::InvalidArgument {
                field: "buffer_length",
                reason: "must be non-zero".into(),
            });
        }
        let contract = ClassifierContract::new(
            parameters.buffer_length,
            network.audio_params.samplerate,
            network.emotions.len(),
            parameters.frame_rate.numerator(),
            parameters.frame_rate.denominator(),
            parameters.inferences_to_skip,
        )?;
        let device = GpuDevice::new(parameters.common.device_ordinal)?;
        let backend = TensorRtClassifierBackend::load(
            Arc::clone(&device),
            model.engine_path(),
            contract.clone(),
        )?;
        backend.validate_track_count(parameters.common.tracks.len())?;
        let execution = EmotionExecutor::new(
            contract.clone(),
            data,
            model_parameters,
            parameters.common.tracks.len(),
        )?;
        let device_output = device.allocate(execution.output_emotion_length())?;
        let result_stream = device.create_stream()?;
        let sample_rate = network.audio_params.samplerate;
        let preferred_emotions = parameters.preferred_emotions;
        Ok(Self {
            execution,
            backend,
            contract,
            tracks: parameters.common.tracks,
            preferred_emotions,
            input_strength: parameters.input_strength,
            frame_rate: parameters.frame_rate,
            sample_rate,
            device_output,
            result_stream,
        })
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }
    pub fn audio(&self, track: usize) -> crate::Result<&Arc<crate::audio2x::AudioAccumulator>> {
        self.tracks
            .get(track)
            .map(|value| &value.audio)
            .ok_or(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            })
    }
    pub fn emotion_count(&self) -> usize {
        self.execution.output_emotion_length()
    }

    /// Runs the owned classifier scheduler and invokes a host callback with
    /// post-processed emotion values. The native device callback adapter is
    /// intentionally separate from this host convenience path.
    pub fn execute_host<C>(
        &mut self,
        mut callback: C,
    ) -> crate::Result<crate::emotion::EmotionExecutionStatus>
    where
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> ControlFlow<()>,
    {
        let tracks = self
            .tracks
            .iter()
            .enumerate()
            .map(|(index, track)| EmotionTrack {
                audio: &track.audio,
                preferred_emotions: self.preferred_emotions.get(index).map(Arc::as_ref),
                input_strength: self.input_strength,
            })
            .collect::<Vec<_>>();
        self.execution
            .execute(&tracks, &mut self.backend, |metadata, output| {
                matches!(callback(metadata, output), ControlFlow::Continue(()))
            })
    }

    pub fn reset_track(&mut self, track: usize) -> crate::Result<()> {
        <Self as crate::audio2x::Executor>::reset_track(self, track)
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2x::Executor for ClassifierEmotionExecutor {
    fn track_count(&self) -> usize {
        self.tracks.len()
    }

    fn reset_track(&mut self, track: usize) -> crate::Result<()> {
        if track >= self.tracks.len() {
            return Err(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            });
        }
        if self.tracks[track].audio.nb_dropped_samples() != 0
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
        Ok(self
            .contract
            .inference_progress
            .available_windows(
                i64::try_from(audio.nb_accumulated_samples()).unwrap_or(i64::MAX),
                audio.is_closed(),
            )?
            .saturating_sub(self.execution.next_inference_index(track)?))
    }

    fn ready_track_count(&self) -> usize {
        self.tracks
            .iter()
            .enumerate()
            .filter(|(track, _)| {
                self.available_execution_count(*track)
                    .is_ok_and(|count| count != 0)
            })
            .count()
    }

    fn total_frame_count(&self, track: usize) -> crate::Result<Option<usize>> {
        let audio = self.audio(track)?;
        if !audio.is_closed() {
            return Ok(None);
        }
        Ok(Some(self.contract.frame_progress.available_windows(
            i64::try_from(audio.nb_accumulated_samples()).unwrap_or(i64::MAX),
            true,
        )?))
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }
    fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }
    fn frame_timestamp(&self, frame: usize) -> crate::Result<i64> {
        self.contract.frame_timestamp(frame)
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2emotion::EmotionExecutor for ClassifierEmotionExecutor {
    fn emotion_count(&self) -> usize {
        self.execution.output_emotion_length()
    }

    fn next_audio_sample_to_read(&self, track: usize) -> crate::Result<usize> {
        self.audio(track)?;
        let inference = self.execution.next_inference_index(track)?;
        Ok(self
            .contract
            .inference_progress
            .window(inference)?
            .start
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
            .map(|(index, track)| EmotionTrack {
                audio: &track.audio,
                preferred_emotions: self.preferred_emotions.get(index).map(Arc::as_ref),
                input_strength: self.input_strength,
            })
            .collect::<Vec<_>>();
        let output = &mut self.device_output;
        let stream = &self.result_stream;
        let mut emitted_frames = 0;
        let mut callback_error = None;
        let status = self
            .execution
            .execute(&tracks, &mut self.backend, |metadata, values| {
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
