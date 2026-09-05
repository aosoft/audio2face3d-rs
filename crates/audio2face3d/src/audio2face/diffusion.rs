//! Diffusion-model Audio2Face facade declarations.

use std::path::PathBuf;

#[cfg(feature = "tensorrt")]
use std::cell::Cell;
#[cfg(feature = "tensorrt")]
use std::convert::Infallible;
#[cfg(feature = "tensorrt")]
use std::ops::ControlFlow;
#[cfg(feature = "tensorrt")]
use std::sync::Arc;

use crate::audio2face::{
    GeometryExecutorCreationParameters, GeometryInteractiveExecutorCreationParameters,
};
use crate::audio2x::FrameRate;

#[cfg(feature = "tensorrt")]
use crate::animation::{
    DiffusionCallbackMetadata, DiffusionContract, DiffusionExecutor, DiffusionPostprocessor,
    DiffusionTrack, GeometryModelData, RegressionGeometry, TensorRtDiffusionBackend,
};
#[cfg(feature = "tensorrt")]
use crate::common::{GeometryAudioParameters, GeometryParameters, NetworkDocument};
#[cfg(feature = "tensorrt")]
use crate::cuda::GpuDevice;
#[cfg(feature = "tensorrt")]
use crate::{Model, ModelKind, ModelParameters};

/// Network dimensions read from an Audio2Face Diffusion model.
///
/// Corresponds to `nva2f::IDiffusionModel::NetworkInfo` in
/// `audio2face-sdk/include/audio2face/model_diffusion.h`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NetworkInfo {
    pub emotion_length: usize,
    pub identity_count: usize,
    pub skin_dimension: usize,
    pub tongue_dimension: usize,
    pub jaw_dimension: usize,
    pub eyes_dimension: usize,
    pub diffusion_step_count: usize,
    pub gru_layer_count: usize,
    pub gru_latent_dimension: usize,
    pub left_truncate_frame_count: usize,
    pub right_truncate_frame_count: usize,
    pub center_frame_count: usize,
    pub audio_buffer_length: usize,
    pub audio_padding_left: usize,
    pub audio_padding_right: usize,
    pub sample_rate: usize,
}

/// Parameters for loading an owning Diffusion geometry executor.
///
/// Corresponds to `nva2f::IDiffusionModel::GeometryExecutorCreationParameters`
/// in `audio2face-sdk/include/audio2face/executor_diffusion.h`, combined with
/// the owned common resources from `nva2f::GeometryExecutorCreationParameters`.
pub struct DiffusionGeometryExecutorCreationParameters {
    pub model_path: PathBuf,
    pub common: GeometryExecutorCreationParameters,
    pub input_strength: f32,
    pub frame_rate: FrameRate,
    pub identity_index: usize,
    pub constant_noise: bool,
}

/// Parameters for loading an owning interactive Diffusion geometry executor.
///
/// `preview_inference_count` corresponds to the original interactive preview
/// count. The common interactive inputs must be closed and preserve the history
/// required to rebuild recurrent state for random access.
///
/// Corresponds to the arguments of
/// `nva2f::CreateDiffusionGeometryInteractiveExecutor` in
/// `audio2face-sdk/include/audio2face/audio2face.h`.
pub struct DiffusionGeometryInteractiveExecutorCreationParameters {
    pub model_path: PathBuf,
    pub common: GeometryInteractiveExecutorCreationParameters,
    pub input_strength: f32,
    pub identity_index: usize,
    pub constant_noise: bool,
    pub preview_inference_count: usize,
}

/// Non-generic owning Diffusion geometry executor facade.
///
/// Corresponds to `nva2f::IDiffusionModel::GeometryExecutor` in the model
/// namespace and implements the `nva2f::IGeometryExecutor` contract from
/// `audio2face-sdk/include/audio2face/executor.h`. It replaces the current
/// `crate::animation::DiffusionExecutor` without exposing backend or
/// post-processor type parameters.
#[cfg(feature = "tensorrt")]
pub struct DiffusionGeometryExecutor {
    execution: DiffusionExecutor,
    backend: TensorRtDiffusionBackend,
    postprocessors: Vec<DiffusionPostprocessor>,
    gpu_postprocessor: crate::animation::GpuRegressionPostprocessor,
    contract: DiffusionContract,
    tracks: Vec<crate::audio2face::GeometryTrackResources>,
    input_strength: f32,
    identity_index: usize,
    constant_noise: bool,
    frame_rate: FrameRate,
    sample_rate: usize,
    execution_option: crate::audio2face::GeometryExecutionOption,
    skin_output: crate::cuda::DeviceBuffer<f32>,
    tongue_output: crate::cuda::DeviceBuffer<f32>,
    jaw_output: crate::cuda::DeviceBuffer<f32>,
    eyes_output: crate::cuda::DeviceBuffer<f32>,
    emotion_output: crate::cuda::DeviceBuffer<f32>,
    result_stream: crate::cuda::CudaStream,
    device: Arc<GpuDevice>,
}

/// Non-generic owning interactive Diffusion geometry executor facade.
///
/// Corresponds to `nva2f::IDiffusionModel::GeometryInteractiveExecutor` and
/// the `nva2f::IGeometryInteractiveExecutor` contract in
/// `audio2face-sdk/include/audio2face/interactive_executor.h`. It replaces the
/// current generic `crate::animation::InteractiveDiffusionExecutor<B, P>`.
#[cfg(feature = "tensorrt")]
pub struct DiffusionGeometryInteractiveExecutor {
    _opaque: Infallible,
    _not_sync: Cell<()>,
}

#[cfg(feature = "tensorrt")]
impl DiffusionGeometryExecutor {
    pub(crate) fn device_arc(&self) -> Arc<GpuDevice> {
        Arc::clone(&self.device)
    }

    #[allow(clippy::result_large_err)]
    pub fn try_into_host_blendshape(
        self,
        parameters: crate::audio2face::HostBlendshapeSolveExecutorCreationParameters<'_>,
    ) -> std::result::Result<
        crate::audio2face::HostBlendshapeSolveExecutor,
        crate::audio2x::TransferError<Self>,
    > {
        match crate::audio2face::HostBlendshapeSolveExecutor::from_source(
            crate::audio2face::GeometrySource::Diffusion(self),
            parameters,
        ) {
            Ok(executor) => Ok(executor),
            Err((error, crate::audio2face::GeometrySource::Diffusion(original))) => {
                Err(crate::audio2x::TransferError { error, original })
            }
            Err(_) => unreachable!("geometry source variant is preserved"),
        }
    }

    #[allow(clippy::result_large_err)]
    pub fn try_into_device_blendshape(
        self,
        parameters: crate::audio2face::DeviceBlendshapeSolveExecutorCreationParameters<'_>,
    ) -> std::result::Result<
        crate::audio2face::DeviceBlendshapeSolveExecutor,
        crate::audio2x::TransferError<Self>,
    > {
        match crate::audio2face::DeviceBlendshapeSolveExecutor::from_source(
            crate::audio2face::GeometrySource::Diffusion(self),
            parameters,
        ) {
            Ok(executor) => Ok(executor),
            Err((error, crate::audio2face::GeometrySource::Diffusion(original))) => {
                Err(crate::audio2x::TransferError { error, original })
            }
            Err(_) => unreachable!("geometry source variant is preserved"),
        }
    }

    /// Loads an owning Diffusion executor and all model-side state.
    pub fn load(parameters: DiffusionGeometryExecutorCreationParameters) -> crate::Result<Self> {
        let model = Model::load(&parameters.model_path)?;
        if model.kind() != ModelKind::Diffusion {
            return Err(crate::Error::InvalidSchema(
                "diffusion executor requires a diffusion model".into(),
            ));
        }
        let NetworkDocument::Geometry(network) = model.network() else {
            return Err(crate::Error::InvalidSchema(
                "diffusion network is missing".into(),
            ));
        };
        let (model_parameters, audio) = match (&network.params, &network.audio_params) {
            (GeometryParameters::Diffusion(value), GeometryAudioParameters::Diffusion(audio)) => {
                (value, audio)
            }
            _ => {
                return Err(crate::Error::InvalidSchema(
                    "diffusion network schema mismatch".into(),
                ));
            }
        };
        if parameters.common.tracks.is_empty() {
            return Err(crate::Error::InvalidArgument {
                field: "tracks",
                reason: "at least one track is required".into(),
            });
        }
        if parameters.identity_index >= model_parameters.identities.len() {
            return Err(crate::Error::OutOfBounds {
                field: "identity_index",
                index: parameters.identity_index,
                len: model_parameters.identities.len(),
            });
        }
        let contract = DiffusionContract::new(model_parameters, audio)?;
        let sample_rate = contract.sample_rate;
        let owned_contract = contract.clone();
        let device = GpuDevice::new(parameters.common.device_ordinal)?;
        let backend = TensorRtDiffusionBackend::load(
            Arc::clone(&device),
            model.engine_path(),
            contract.clone(),
        )?;
        let config = match model.parameters(0)? {
            ModelParameters::Geometry(value) => value,
            _ => {
                return Err(crate::Error::InvalidSchema(
                    "diffusion geometry config is missing".into(),
                ));
            }
        };
        let model_data = GeometryModelData::load_diffusion(model.model_data_path(0)?)?;
        let processor = model_data.diffusion_postprocessor(config, contract.result_layout)?;
        let postprocessors = vec![processor; parameters.common.tracks.len()];
        let track_count = parameters.common.tracks.len();
        let dt =
            parameters.frame_rate.denominator() as f32 / parameters.frame_rate.numerator() as f32;
        let gpu_postprocessor =
            model_data.gpu_postprocessor(&device, backend.stream(), config, track_count, dt)?;
        let execution = DiffusionExecutor::new(contract, track_count, 0)?;
        let skin_output = device.allocate(
            owned_contract
                .result_layout
                .skin
                .saturating_mul(track_count),
        )?;
        let tongue_output = device.allocate(
            owned_contract
                .result_layout
                .tongue
                .saturating_mul(track_count),
        )?;
        let jaw_output = device.allocate(16_usize.saturating_mul(track_count))?;
        let eyes_output = device.allocate(6_usize.saturating_mul(track_count))?;
        let emotion_output = device.allocate(owned_contract.emotion_size)?;
        let result_stream = device.create_stream()?;
        Ok(Self {
            execution,
            backend,
            postprocessors,
            gpu_postprocessor,
            contract: owned_contract,
            tracks: parameters.common.tracks,
            input_strength: parameters.input_strength,
            identity_index: parameters.identity_index,
            constant_noise: parameters.constant_noise,
            frame_rate: parameters.frame_rate,
            sample_rate,
            execution_option: parameters.common.execution_option,
            skin_output,
            tongue_output,
            jaw_output,
            eyes_output,
            emotion_output,
            result_stream,
            device,
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
    pub fn execution_option(&self) -> crate::audio2face::GeometryExecutionOption {
        self.execution_option
    }
    pub fn set_execution_option(&mut self, value: crate::audio2face::GeometryExecutionOption) {
        self.execution_option = value;
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
    pub fn emotions(
        &self,
        track: usize,
    ) -> crate::Result<&Arc<crate::audio2x::EmotionAccumulator>> {
        self.tracks
            .get(track)
            .map(|value| &value.emotions)
            .ok_or(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            })
    }

    /// Executes one available internal inference and applies the owned
    /// post-processor before invoking the host adapter callback.
    pub fn execute_host<C>(
        &mut self,
        mut callback: C,
    ) -> crate::Result<crate::animation::DiffusionExecutionStatus>
    where
        C: FnMut(DiffusionCallbackMetadata, &RegressionGeometry) -> ControlFlow<()>,
    {
        let tracks = self
            .tracks
            .iter()
            .map(|track| DiffusionTrack {
                audio: &track.audio,
                emotions: &track.emotions,
                identity_index: self.identity_index,
                input_strength: self.input_strength,
            })
            .collect::<Vec<_>>();
        let mut postprocessors = std::mem::take(&mut self.postprocessors);
        let dt = self.frame_rate.denominator() as f32 / self.frame_rate.numerator() as f32;
        let result = self
            .execution
            .execute(&tracks, &mut self.backend, |metadata, output| {
                let Some(processor) = postprocessors.get_mut(metadata.track) else {
                    return false;
                };
                let geometry = match processor.process(output, dt) {
                    Ok(geometry) => geometry,
                    Err(_) => return false,
                };
                matches!(callback(metadata, &geometry), ControlFlow::Continue(()))
            });
        self.postprocessors = postprocessors;
        result
    }

    pub fn reset_track(&mut self, track: usize) -> crate::Result<()> {
        <Self as crate::audio2x::Executor>::reset_track(self, track)
    }
    pub const fn constant_noise(&self) -> bool {
        self.constant_noise
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2x::Executor for DiffusionGeometryExecutor {
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
            || self.tracks[track].emotions.state().dropped_emotions != 0
        {
            return Err(crate::Error::InputHistoryUnavailable { track });
        }
        self.execution.reset(track)?;
        self.postprocessors
            .get_mut(track)
            .ok_or(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            })?
            .reset()?;
        self.gpu_postprocessor
            .reset_track(track, self.backend.stream())
    }

    fn has_execution_started(&self, track: usize) -> crate::Result<bool> {
        if track >= self.tracks.len() {
            return Err(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            });
        }
        Ok(self.execution.next_inference_index(track)? != 0)
    }

    fn available_execution_count(&self, track: usize) -> crate::Result<usize> {
        let audio = self.audio(track)?;
        Ok(self
            .contract
            .progress
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
        Ok(self.contract.frame_progress.window(frame)?.target)
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2face::FaceExecutor for DiffusionGeometryExecutor {
    fn next_audio_sample_to_read(&self, track: usize) -> crate::Result<usize> {
        self.audio(track)?;
        let inference = self.execution.next_inference_index(track)?;
        Ok(self.contract.progress.window(inference)?.start.max(0) as usize)
    }

    fn next_emotion_timestamp_to_read(&self, track: usize) -> crate::Result<i64> {
        self.audio(track)?;
        let inference = self.execution.next_inference_index(track)?;
        Ok(self.contract.progress.window(inference)?.target)
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2face::GeometryExecutor for DiffusionGeometryExecutor {
    fn execution_option(&self) -> crate::audio2face::GeometryExecutionOption {
        self.execution_option
    }

    fn set_execution_option(
        &mut self,
        value: crate::audio2face::GeometryExecutionOption,
    ) -> crate::Result<()> {
        self.execution_option = value;
        Ok(())
    }

    fn skin_geometry_size(&self) -> usize {
        self.contract.result_layout.skin
    }
    fn tongue_geometry_size(&self) -> usize {
        self.contract.result_layout.tongue
    }
    fn jaw_transform_size(&self) -> usize {
        16
    }
    fn eyes_rotation_size(&self) -> usize {
        6
    }

    fn execute(
        &mut self,
        callbacks: crate::audio2face::GeometryCallbacks<'_>,
    ) -> crate::Result<crate::audio2x::Execution> {
        let crate::audio2face::GeometryCallbacks {
            results,
            mut emotions,
        } = callbacks;
        let tracks = self
            .tracks
            .iter()
            .map(|track| DiffusionTrack {
                audio: &track.audio,
                emotions: &track.emotions,
                identity_index: self.identity_index,
                input_strength: self.input_strength,
            })
            .collect::<Vec<_>>();
        let option = self.execution_option;
        let skin_output = &mut self.skin_output;
        let tongue_output = &mut self.tongue_output;
        let jaw_output = &mut self.jaw_output;
        let eyes_output = &mut self.eyes_output;
        let emotion_output = &mut self.emotion_output;
        let emotion_stream = &self.result_stream;
        let source_tracks = &self.tracks;
        let mut emitted_frames = 0;
        let mut callback_error = None;
        let status = self.execution.execute_device(
            &tracks,
            &mut self.backend,
            &mut self.gpu_postprocessor,
            skin_output,
            tongue_output,
            jaw_output,
            eyes_output,
            |metadata, skin, tongue, jaw, eyes, stream| {
                if callback_error.is_some() {
                    return false;
                }
                let public_metadata = crate::audio2x::CallbackMetadata {
                    track_index: metadata.track,
                    frame_index: metadata.frame,
                    timestamp: metadata.timestamp,
                    next_timestamp: metadata.next_timestamp,
                };
                if let Some(emotion_callback) = emotions.as_deref_mut() {
                    let values = match source_tracks[metadata.track]
                        .emotions
                        .read(metadata.timestamp)
                    {
                        Ok(values) => values,
                        Err(error) => {
                            callback_error = Some(crate::Error::InvalidSchema(format!(
                                "diffusion emotion read failed: {error}"
                            )));
                            return false;
                        }
                    };
                    if let Err(error) = emotion_output.copy_from(&values, emotion_stream) {
                        callback_error = Some(error);
                        return false;
                    }
                    emotion_callback(crate::audio2face::FaceEmotions {
                        metadata: public_metadata,
                        values: crate::audio2x::DeviceComponentResults {
                            values: emotion_output.view(),
                            stream: emotion_stream.as_ref(),
                        },
                    });
                }
                emitted_frames += 1;
                matches!(
                    results(crate::audio2face::GeometryResults {
                        metadata: public_metadata,
                        skin: option
                            .contains(crate::audio2face::GeometryExecutionOption::SKIN)
                            .then_some(crate::audio2x::DeviceComponentResults {
                                values: skin,
                                stream,
                            }),
                        tongue: option
                            .contains(crate::audio2face::GeometryExecutionOption::TONGUE)
                            .then_some(crate::audio2x::DeviceComponentResults {
                                values: tongue,
                                stream,
                            }),
                        jaw: option
                            .contains(crate::audio2face::GeometryExecutionOption::JAW)
                            .then_some(crate::audio2x::DeviceComponentResults {
                                values: jaw,
                                stream,
                            }),
                        eyes: option
                            .contains(crate::audio2face::GeometryExecutionOption::EYES)
                            .then_some(crate::audio2x::DeviceComponentResults {
                                values: eyes,
                                stream,
                            }),
                    }),
                    ControlFlow::Continue(())
                )
            },
        );
        let status = status?;
        if let Some(error) = callback_error {
            return Err(error);
        }
        let (state, executed_tracks) = match status {
            crate::animation::DiffusionExecutionStatus::AwaitingInput => {
                (crate::audio2x::ExecutionState::AwaitingInput, 0)
            }
            crate::animation::DiffusionExecutionStatus::Complete => {
                (crate::audio2x::ExecutionState::Complete, 0)
            }
            crate::animation::DiffusionExecutionStatus::Executed { tracks } => {
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
