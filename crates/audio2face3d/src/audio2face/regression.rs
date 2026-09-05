//! Regression-model Audio2Face facade declarations.

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
    GeometryModelData, RegressionCallbackMetadata, RegressionContract, RegressionExecutor,
    RegressionGeometry, RegressionPostprocessor, RegressionTrack, TensorRtRegressionBackend,
};
#[cfg(feature = "tensorrt")]
use crate::common::{GeometryAudioParameters, GeometryParameters, NetworkDocument};
#[cfg(feature = "tensorrt")]
use crate::cuda::GpuDevice;
#[cfg(feature = "tensorrt")]
use crate::{Model, ModelKind, ModelParameters};

/// Network dimensions read from an Audio2Face Regression model.
///
/// Corresponds to `nva2f::IRegressionModel::NetworkInfo` in
/// `audio2face-sdk/include/audio2face/model_regression.h`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NetworkInfo {
    pub implicit_emotion_length: usize,
    pub explicit_emotion_length: usize,
    pub skin_shape_count: usize,
    pub tongue_shape_count: usize,
    pub skin_result_size: usize,
    pub tongue_result_size: usize,
    pub jaw_result_size: usize,
    pub eyes_result_size: usize,
    pub audio_buffer_length: usize,
    pub audio_buffer_offset: usize,
    pub sample_rate: usize,
}

/// Parameters for loading an owning Regression geometry executor.
///
/// Corresponds to `nva2f::IRegressionModel::GeometryExecutorCreationParameters`
/// in `audio2face-sdk/include/audio2face/executor_regression.h`, combined with
/// the owned common resources from `nva2f::GeometryExecutorCreationParameters`.
pub struct RegressionGeometryExecutorCreationParameters {
    pub model_path: PathBuf,
    pub common: GeometryExecutorCreationParameters,
    pub input_strength: f32,
    pub frame_rate: FrameRate,
    pub source_emotion_shot: Option<String>,
    pub source_emotion_frame: usize,
}

/// Parameters for loading an owning interactive Regression geometry executor.
///
/// `batch_size` is the executor's internal inference/cache batch size, not a
/// track count. The common interactive inputs must be closed and retain all
/// history required for random access.
///
/// Corresponds to the arguments of
/// `nva2f::CreateRegressionGeometryInteractiveExecutor` in
/// `audio2face-sdk/include/audio2face/audio2face.h`.
pub struct RegressionGeometryInteractiveExecutorCreationParameters {
    pub model_path: PathBuf,
    pub common: GeometryInteractiveExecutorCreationParameters,
    pub input_strength: f32,
    pub frame_rate: FrameRate,
    pub source_emotion_shot: Option<String>,
    pub source_emotion_frame: usize,
    pub batch_size: usize,
}

/// Non-generic owning Regression geometry executor facade.
///
/// Corresponds to `nva2f::IRegressionModel::GeometryExecutor` in the model
/// namespace and implements the `nva2f::IGeometryExecutor` contract from
/// `audio2face-sdk/include/audio2face/executor.h`. This is the completed-facade
/// replacement for the current `crate::animation::RegressionExecutor`; its
/// backend and post-processor type parameters remain private implementation
/// details.
#[cfg(feature = "tensorrt")]
pub struct RegressionGeometryExecutor {
    /// The execution object owns the TensorRT backend and post-process state.
    /// It is deliberately concrete here; no backend type appears in the public
    /// signature or constructor.
    execution: RegressionExecutor,
    backend: TensorRtRegressionBackend,
    postprocessors: Vec<RegressionPostprocessor>,
    contract: RegressionContract,
    tracks: Vec<crate::audio2face::GeometryTrackResources>,
    implicit_emotions: Vec<Vec<f32>>,
    input_strength: f32,
    frame_rate: FrameRate,
    sample_rate: usize,
    execution_option: crate::audio2face::GeometryExecutionOption,
    started: Vec<bool>,
    skin_output: crate::cuda::DeviceBuffer<f32>,
    tongue_output: crate::cuda::DeviceBuffer<f32>,
    jaw_output: crate::cuda::DeviceBuffer<f32>,
    eyes_output: crate::cuda::DeviceBuffer<f32>,
    emotion_output: crate::cuda::DeviceBuffer<f32>,
    result_stream: crate::cuda::CudaStream,
    device: Arc<GpuDevice>,
}

/// Non-generic owning interactive Regression geometry executor facade.
///
/// Corresponds to `nva2f::IRegressionModel::GeometryInteractiveExecutor` and
/// the `nva2f::IGeometryInteractiveExecutor` contract in
/// `audio2face-sdk/include/audio2face/interactive_executor.h`. It replaces the
/// current generic `crate::animation::InteractiveRegressionExecutor<B, P>`.
#[cfg(feature = "tensorrt")]
pub struct RegressionGeometryInteractiveExecutor {
    _opaque: Infallible,
    _not_sync: Cell<()>,
}

#[cfg(feature = "tensorrt")]
impl RegressionGeometryExecutor {
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
            crate::audio2face::GeometrySource::Regression(self),
            parameters,
        ) {
            Ok(executor) => Ok(executor),
            Err((error, crate::audio2face::GeometrySource::Regression(original))) => {
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
            crate::audio2face::GeometrySource::Regression(self),
            parameters,
        ) {
            Ok(executor) => Ok(executor),
            Err((error, crate::audio2face::GeometrySource::Regression(original))) => {
                Err(crate::audio2x::TransferError { error, original })
            }
            Err(_) => unreachable!("geometry source variant is preserved"),
        }
    }

    /// Loads a model and takes ownership of the resources used by each track.
    ///
    /// This is the synchronous construction hook used by the Step 3 native
    /// path. Runtime-independent asynchronous factories are layered on top in
    /// the factory step; the executor itself never borrows `Model` or caller
    /// supplied model data.
    pub fn load(parameters: RegressionGeometryExecutorCreationParameters) -> crate::Result<Self> {
        let model = Model::load(&parameters.model_path)?;
        if model.kind() != ModelKind::Regression {
            return Err(crate::Error::InvalidSchema(
                "regression executor requires a regression model".into(),
            ));
        }
        let NetworkDocument::Geometry(network) = model.network() else {
            return Err(crate::Error::InvalidSchema(
                "regression network is missing".into(),
            ));
        };
        let (model_parameters, audio) = match (&network.params, &network.audio_params) {
            (GeometryParameters::Regression(value), GeometryAudioParameters::Regression(audio)) => {
                (value, audio)
            }
            _ => {
                return Err(crate::Error::InvalidSchema(
                    "regression network schema mismatch".into(),
                ));
            }
        };
        if parameters.common.tracks.is_empty() {
            return Err(crate::Error::InvalidArgument {
                field: "tracks",
                reason: "at least one track is required".into(),
            });
        }
        let contract = RegressionContract::new(
            model_parameters,
            audio,
            parameters.frame_rate.numerator(),
            parameters.frame_rate.denominator(),
        )?;
        let device = GpuDevice::new(parameters.common.device_ordinal)?;
        let backend = TensorRtRegressionBackend::load(
            Arc::clone(&device),
            model.engine_path(),
            contract.clone(),
        )?;
        let config = match model.parameters(0)? {
            ModelParameters::Geometry(value) => value,
            _ => {
                return Err(crate::Error::InvalidSchema(
                    "regression geometry config is missing".into(),
                ));
            }
        };
        let model_data = GeometryModelData::load_regression(model.model_data_path(0)?)?;
        let processor = model_data.regression_postprocessor(
            config,
            model_parameters.num_shapes_skin,
            model_parameters.num_shapes_tongue,
        )?;
        let processors = vec![processor; parameters.common.tracks.len()];
        let sample_rate = contract.sample_rate;
        let owned_contract = contract.clone();
        let track_count = parameters.common.tracks.len();
        let execution = RegressionExecutor::new(contract, track_count)?;
        let skin_output = device.allocate(owned_contract.result_skin_size)?;
        let tongue_output = device.allocate(owned_contract.result_tongue_size)?;
        let jaw_output = device.allocate(16)?;
        let eyes_output = device.allocate(6)?;
        let emotion_output = device.allocate(owned_contract.emotion_size)?;
        let result_stream = device.create_stream()?;
        Ok(Self {
            execution,
            backend,
            postprocessors: processors,
            contract: owned_contract,
            implicit_emotions: vec![vec![0.0; model_parameters.implicit_emotion_len]; track_count],
            input_strength: parameters.input_strength,
            frame_rate: parameters.frame_rate,
            sample_rate,
            execution_option: parameters.common.execution_option,
            started: vec![false; track_count],
            tracks: parameters.common.tracks,
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

    /// Executes the owned internal scheduler and exposes its host geometry
    /// output to the native adapter. The device-view callback is attached by
    /// the CUDA facade once the result buffer/fence is available.
    pub fn execute_host<C>(
        &mut self,
        mut callback: C,
    ) -> crate::Result<crate::animation::PumpStatus>
    where
        C: FnMut(RegressionCallbackMetadata, &RegressionGeometry) -> ControlFlow<()>,
    {
        let tracks = self
            .tracks
            .iter()
            .enumerate()
            .map(|(index, track)| RegressionTrack {
                audio: &track.audio,
                emotions: &track.emotions,
                implicit_emotion: &self.implicit_emotions[index],
                input_strength: self.input_strength,
            })
            .collect::<Vec<_>>();
        let mut postprocessors = std::mem::take(&mut self.postprocessors);
        let result = self
            .execution
            .pump(&tracks, &mut self.backend, |metadata, output| {
                let Some(processor) = postprocessors.get_mut(metadata.track) else {
                    return false;
                };
                let geometry = match processor.process(
                    output,
                    self.frame_rate.denominator() as f32 / self.frame_rate.numerator() as f32,
                ) {
                    Ok(geometry) => geometry,
                    Err(_) => return false,
                };
                matches!(callback(metadata, &geometry), ControlFlow::Continue(()))
            });
        self.postprocessors = postprocessors;
        if result.is_ok() {
            self.started.fill(true);
        }
        result
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2x::Executor for RegressionGeometryExecutor {
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
        self.postprocessors[track].reset()?;
        self.started[track] = false;
        Ok(())
    }

    fn has_execution_started(&self, track: usize) -> crate::Result<bool> {
        self.started
            .get(track)
            .copied()
            .ok_or(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            })
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
            .saturating_sub(self.execution.next_frame_index(track)?))
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
        Ok(Some(self.contract.progress.available_windows(
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
        Ok(self.contract.progress.window(frame)?.target)
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2face::FaceExecutor for RegressionGeometryExecutor {
    fn next_audio_sample_to_read(&self, track: usize) -> crate::Result<usize> {
        self.audio(track)?;
        let frame = self.execution.next_frame_index(track)?;
        Ok(self.contract.progress.window(frame)?.start.max(0) as usize)
    }

    fn next_emotion_timestamp_to_read(&self, track: usize) -> crate::Result<i64> {
        self.audio(track)?;
        let frame = self.execution.next_frame_index(track)?;
        Ok(self.contract.progress.window(frame)?.target)
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2face::GeometryExecutor for RegressionGeometryExecutor {
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
        self.contract.result_skin_size
    }
    fn tongue_geometry_size(&self) -> usize {
        self.contract.result_tongue_size
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
            .enumerate()
            .map(|(index, track)| RegressionTrack {
                audio: &track.audio,
                emotions: &track.emotions,
                implicit_emotion: &self.implicit_emotions[index],
                input_strength: self.input_strength,
            })
            .collect::<Vec<_>>();
        let mut postprocessors = std::mem::take(&mut self.postprocessors);
        let option = self.execution_option;
        let stream = &self.result_stream;
        let skin_output = &mut self.skin_output;
        let tongue_output = &mut self.tongue_output;
        let jaw_output = &mut self.jaw_output;
        let eyes_output = &mut self.eyes_output;
        let emotion_output = &mut self.emotion_output;
        let source_tracks = &self.tracks;
        let implicit_emotions = &self.implicit_emotions;
        let dt = self.frame_rate.denominator() as f32 / self.frame_rate.numerator() as f32;
        let mut emitted_frames = 0;
        let mut active_tracks = vec![false; self.tracks.len()];
        let mut callback_error = None;
        let status = self
            .execution
            .pump(&tracks, &mut self.backend, |metadata, inference| {
                active_tracks[metadata.track] = true;
                if callback_error.is_some() {
                    return false;
                }
                let geometry = match postprocessors[metadata.track].process(inference, dt) {
                    Ok(geometry) => geometry,
                    Err(error) => {
                        callback_error = Some(error);
                        return false;
                    }
                };
                let copied = (|| -> crate::Result<()> {
                    if option.contains(crate::audio2face::GeometryExecutionOption::SKIN) {
                        skin_output.copy_from(&geometry.skin, stream)?;
                    }
                    if option.contains(crate::audio2face::GeometryExecutionOption::TONGUE) {
                        tongue_output.copy_from(&geometry.tongue, stream)?;
                    }
                    if option.contains(crate::audio2face::GeometryExecutionOption::JAW) {
                        jaw_output.copy_from(&geometry.jaw_transform, stream)?;
                    }
                    if option.contains(crate::audio2face::GeometryExecutionOption::EYES) {
                        let eyes = [
                            geometry.eyes_rotation.right[0],
                            geometry.eyes_rotation.right[1],
                            geometry.eyes_rotation.right[2],
                            geometry.eyes_rotation.left[0],
                            geometry.eyes_rotation.left[1],
                            geometry.eyes_rotation.left[2],
                        ];
                        eyes_output.copy_from(&eyes, stream)?;
                    }
                    Ok(())
                })();
                if let Err(error) = copied {
                    callback_error = Some(error);
                    return false;
                }
                let public_metadata = crate::audio2x::CallbackMetadata {
                    track_index: metadata.track,
                    frame_index: metadata.frame,
                    timestamp: metadata.timestamp,
                    next_timestamp: metadata.next_timestamp,
                };
                if let Some(emotion_callback) = emotions.as_deref_mut() {
                    let mut values = match source_tracks[metadata.track]
                        .emotions
                        .read(metadata.timestamp)
                    {
                        Ok(values) => values,
                        Err(error) => {
                            callback_error = Some(crate::Error::InvalidSchema(format!(
                                "regression emotion read failed: {error}"
                            )));
                            return false;
                        }
                    };
                    values.extend_from_slice(&implicit_emotions[metadata.track]);
                    if let Err(error) = emotion_output.copy_from(&values, stream) {
                        callback_error = Some(error);
                        return false;
                    }
                    emotion_callback(crate::audio2face::FaceEmotions {
                        metadata: public_metadata,
                        values: crate::audio2x::DeviceComponentResults {
                            values: emotion_output.view(),
                            stream: stream.as_ref(),
                        },
                    });
                }
                emitted_frames += 1;
                matches!(
                    results(crate::audio2face::GeometryResults {
                        metadata: public_metadata,
                        skin: option
                            .contains(crate::audio2face::GeometryExecutionOption::SKIN)
                            .then(|| crate::audio2x::DeviceComponentResults {
                                values: skin_output.view(),
                                stream: stream.as_ref(),
                            }),
                        tongue: option
                            .contains(crate::audio2face::GeometryExecutionOption::TONGUE)
                            .then(|| crate::audio2x::DeviceComponentResults {
                                values: tongue_output.view(),
                                stream: stream.as_ref(),
                            }),
                        jaw: option
                            .contains(crate::audio2face::GeometryExecutionOption::JAW)
                            .then(|| crate::audio2x::DeviceComponentResults {
                                values: jaw_output.view(),
                                stream: stream.as_ref(),
                            }),
                        eyes: option
                            .contains(crate::audio2face::GeometryExecutionOption::EYES)
                            .then(|| crate::audio2x::DeviceComponentResults {
                                values: eyes_output.view(),
                                stream: stream.as_ref(),
                            }),
                    }),
                    ControlFlow::Continue(())
                )
            });
        self.postprocessors = postprocessors;
        let status = status?;
        if let Some(error) = callback_error {
            return Err(error);
        }
        for (started, active) in self.started.iter_mut().zip(&active_tracks) {
            *started |= *active;
        }
        let state = match status {
            crate::animation::PumpStatus::AwaitingInput => {
                crate::audio2x::ExecutionState::AwaitingInput
            }
            crate::animation::PumpStatus::Complete => crate::audio2x::ExecutionState::Complete,
            crate::animation::PumpStatus::Interrupted => crate::audio2x::ExecutionState::Progress,
        };
        Ok(crate::audio2x::Execution::ready(
            crate::audio2x::ExecutionReport {
                state,
                executed_tracks: active_tracks.iter().filter(|active| **active).count(),
                emitted_frames,
            },
        ))
    }
}
