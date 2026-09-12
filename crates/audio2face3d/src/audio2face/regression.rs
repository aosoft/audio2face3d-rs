//! Regression-model Audio2Face facade declarations.

#[cfg(feature = "tensorrt")]
fn load_implicit_emotion(
    model: &crate::Model,
    shot: Option<&str>,
    frame: usize,
    dimension: usize,
) -> crate::Result<Vec<f32>> {
    use crate::common::{Error, ModelDocument, NpzArchive, load_model};
    let invalid = |message: &str| Error::InvalidSchema(message.into());
    if dimension == 0 {
        return Ok(Vec::new());
    }
    let crate::ModelParameters::Geometry(config) = model.parameters(0)? else {
        return Err(invalid("regression geometry config is missing"));
    };
    let (shot, frame) = match shot {
        Some(shot) => (shot, frame),
        None => (
            config
                .source_shot
                .as_deref()
                .ok_or_else(|| invalid("regression source shot is missing"))?,
            usize::try_from(
                config
                    .source_frame
                    .ok_or_else(|| invalid("regression source frame is missing"))?,
            )
            .map_err(|_| invalid("regression source frame is negative"))?,
        ),
    };
    let ModelDocument::Single(descriptor) = load_model(model.descriptor_path())? else {
        return Err(invalid(
            "regression emotion database requires a single model",
        ));
    };
    let path = descriptor
        .emotion_database_path
        .ok_or_else(|| invalid("regression emotion database is missing"))?;
    let mut archive = NpzArchive::open(&path)?;
    let names = archive.strings("emo_spec_names")?;
    let starts = archive.i32("emo_spec_start")?;
    let sizes = archive.i32("emo_spec_size")?;
    let shape = archive.shape("emo_db")?;
    let values = archive.f32("emo_db")?;
    select_implicit_emotion(
        &names, &starts, &sizes, &shape, &values, shot, frame, dimension,
    )
}

#[cfg(feature = "tensorrt")]
#[allow(clippy::too_many_arguments)]
fn select_implicit_emotion(
    names: &[String],
    starts: &[i32],
    sizes: &[i32],
    shape: &[usize],
    values: &[f32],
    shot: &str,
    frame: usize,
    dimension: usize,
) -> crate::Result<Vec<f32>> {
    let invalid = |message: &str| crate::Error::InvalidSchema(message.into());
    if names.len() != starts.len()
        || names.len() != sizes.len()
        || shape.len() != 2
        || shape[1] != dimension
        || dimension == 0
        || shape[0].checked_mul(dimension) != Some(values.len())
    {
        return Err(invalid("invalid implicit emotion database dimensions"));
    }
    let matches: Vec<_> = names
        .iter()
        .enumerate()
        .filter(|(_, name)| name.as_str() == shot)
        .map(|(i, _)| i)
        .collect();
    if matches.len() != 1 {
        return Err(invalid("implicit emotion shot is missing or duplicated"));
    }
    let index = matches[0];
    let start =
        usize::try_from(starts[index]).map_err(|_| invalid("negative emotion shot start"))?;
    let size = usize::try_from(sizes[index]).map_err(|_| invalid("negative emotion shot size"))?;
    if frame >= size || start.checked_add(size).is_none_or(|end| end > shape[0]) {
        return Err(invalid("implicit emotion frame or shot is out of bounds"));
    }
    let offset = (start + frame) * dimension;
    let result = values[offset..offset + dimension].to_vec();
    if !result.iter().all(|value| value.is_finite()) {
        return Err(invalid("implicit emotion contains nonfinite values"));
    }
    Ok(result)
}

#[cfg(all(test, feature = "tensorrt"))]
mod implicit_emotion_tests {
    use super::select_implicit_emotion;
    #[test]
    fn selects_shot_relative_frame_and_rejects_invalid_database() {
        let names = vec!["neutral".to_owned()];
        let values = [9.0, 8.0, 1.0, 2.0, 3.0, 4.0];
        let select = |starts: &[i32], sizes: &[i32], shape: &[usize], shot, frame| {
            select_implicit_emotion(&names, starts, sizes, shape, &values, shot, frame, 2)
        };
        assert_eq!(
            select(&[1], &[2], &[3, 2], "neutral", 1).unwrap(),
            [3.0, 4.0]
        );
        assert!(select(&[1], &[2], &[3, 2], "missing", 0).is_err());
        assert!(select(&[1], &[2], &[3, 2], "neutral", 2).is_err());
        assert!(select(&[-1], &[2], &[3, 2], "neutral", 0).is_err());
        assert!(select(&[2], &[2], &[3, 2], "neutral", 0).is_err());
        assert!(select(&[1], &[2], &[2, 3], "neutral", 0).is_err());
        assert!(select(&[], &[2], &[3, 2], "neutral", 0).is_err());
        assert!(
            select_implicit_emotion(
                &["neutral".into(), "neutral".into()],
                &[0, 1],
                &[1, 1],
                &[3, 2],
                &values,
                "neutral",
                0,
                2,
            )
            .is_err()
        );
        assert!(
            select_implicit_emotion(
                &names,
                &[0],
                &[1],
                &[1, 2],
                &[f32::NAN, 0.0],
                "neutral",
                0,
                2,
            )
            .is_err()
        );
    }
}

use std::path::PathBuf;

#[cfg(feature = "tensorrt")]
use std::cell::Cell;
#[cfg(feature = "tensorrt")]
use std::future::Future;
#[cfg(feature = "tensorrt")]
use std::ops::ControlFlow;
#[cfg(feature = "tensorrt")]
use std::pin::Pin;
#[cfg(feature = "tensorrt")]
use std::sync::Arc;

use crate::audio2face::{
    GeometryExecutorCreationParameters, GeometryInteractiveExecutorCreationParameters,
};
use crate::audio2x::FrameRate;

#[cfg(feature = "tensorrt")]
use crate::animation::{
    GeometryModelData, RegressionContract, RegressionGeometry,
    RegressionGeometryInteractiveExecution, RegressionPostprocessor, RegressionScheduler,
    RegressionTrack, TensorRtRegressionBackend,
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

/// Canonical asynchronous factory for standard Regression geometry.
///
/// Corresponds to `nva2f::CreateRegressionGeometryExecutor` in
/// `audio2face-sdk/include/audio2face/audio2face.h`.
///
/// ```no_run
/// # async fn example(parameters: audio2face3d::audio2face::regression::RegressionGeometryExecutorCreationParameters) -> audio2face3d::Result<()> {
/// use audio2face3d::audio2face::{GeometryCallbacks, GeometryExecutor, GeometryResults};
/// use audio2face3d::audio2face::regression::{RegressionGeometryExecutor, RegressionGeometryExecutorFactory};
/// use std::ops::ControlFlow;
///
/// let mut executor: RegressionGeometryExecutor =
///     RegressionGeometryExecutorFactory::load(parameters).await?;
/// let mut frames = 0;
/// let mut callback = |_: GeometryResults<'_>| {
///     frames += 1; // A synchronous callback may borrow stack state.
///     ControlFlow::Continue(())
/// };
/// executor.execute(GeometryCallbacks { results: &mut callback, emotions: None })?.await?;
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "tensorrt")]
pub struct RegressionGeometryExecutorFactory;

#[cfg(feature = "tensorrt")]
impl RegressionGeometryExecutorFactory {
    /// Loads model files and initializes CUDA/TensorRT on a worker thread.
    pub fn load(
        parameters: RegressionGeometryExecutorCreationParameters,
    ) -> crate::audio2x::ExecutorFuture<'static, RegressionGeometryExecutor> {
        crate::audio2x::spawn_blocking_factory(move || {
            RegressionGeometryExecutor::load_sync(parameters)
        })
    }
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

/// Canonical asynchronous factory for interactive Regression geometry.
#[cfg(feature = "tensorrt")]
pub struct RegressionGeometryInteractiveExecutorFactory;

#[cfg(feature = "tensorrt")]
impl RegressionGeometryInteractiveExecutorFactory {
    /// Loads model files and initializes CUDA/TensorRT on a worker thread.
    pub fn load(
        parameters: RegressionGeometryInteractiveExecutorCreationParameters,
    ) -> crate::audio2x::ExecutorFuture<'static, RegressionGeometryInteractiveExecutor> {
        crate::audio2x::spawn_blocking_factory(move || {
            RegressionGeometryInteractiveExecutor::load_sync(parameters)
        })
    }
}

/// Non-generic owning Regression geometry executor facade.
///
/// Corresponds to `nva2f::IRegressionModel::GeometryExecutor` in the model
/// namespace and implements the `nva2f::IGeometryExecutor` contract from
/// `audio2face-sdk/include/audio2face/executor.h`. Backend and post-processor
/// types remain private implementation details.
#[cfg(feature = "tensorrt")]
pub struct RegressionGeometryExecutor {
    /// The execution object owns the TensorRT backend and post-process state.
    /// It is deliberately concrete here; no backend type appears in the public
    /// signature or constructor.
    execution: RegressionScheduler,
    backend: TensorRtRegressionBackend,
    postprocessors: Vec<RegressionPostprocessor>,
    gpu_postprocessor: crate::animation::GpuRegressionPcaPostprocessor,
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
/// `audio2face-sdk/include/audio2face/interactive_executor.h`.
#[cfg(feature = "tensorrt")]
pub struct RegressionGeometryInteractiveExecutor {
    execution:
        RegressionGeometryInteractiveExecution<TensorRtRegressionBackend, RegressionPostprocessor>,
    contract: RegressionContract,
    audio: Arc<crate::audio2x::AudioAccumulator>,
    emotions: Arc<crate::audio2x::EmotionAccumulator>,
    stream: crate::cuda::CudaStream,
    skin: crate::cuda::DeviceBuffer<f32>,
    tongue: crate::cuda::DeviceBuffer<f32>,
    jaw: crate::cuda::DeviceBuffer<f32>,
    eyes: crate::cuda::DeviceBuffer<f32>,
    batch_size: usize,
    core_interrupt: crate::animation::InteractiveGeometryInterrupt,
    interrupt_handle: crate::audio2x::InteractiveInterruptHandle,
    _not_sync: Cell<()>,
}

#[cfg(feature = "tensorrt")]
struct YieldOnce(bool);

#[cfg(feature = "tensorrt")]
impl Future for YieldOnce {
    type Output = ();

    fn poll(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.0 {
            std::task::Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }
}

#[cfg(feature = "tensorrt")]
fn validate_regression_inputs(
    audio: &crate::audio2x::AudioAccumulator,
    emotions: &crate::audio2x::EmotionAccumulator,
    contract: &RegressionContract,
) -> crate::Result<()> {
    let emotion_state = emotions.state();
    if !audio.is_closed()
        || audio.nb_dropped_samples() != 0
        || !emotion_state.closed
        || emotion_state.dropped_emotions != 0
    {
        return Err(crate::Error::InputHistoryUnavailable { track: 0 });
    }
    if emotions.state().emotion_size != contract.explicit_emotion_size {
        return Err(crate::Error::InvalidSchema(
            "interactive regression emotion dimensions differ".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "tensorrt")]
impl RegressionGeometryInteractiveExecutor {
    pub(crate) fn load_sync(
        parameters: RegressionGeometryInteractiveExecutorCreationParameters,
    ) -> crate::Result<Self> {
        if parameters.batch_size == 0 {
            return Err(crate::Error::InvalidArgument {
                field: "batch_size",
                reason: "batch size must be non-zero".into(),
            });
        }
        if !parameters.input_strength.is_finite() {
            return Err(crate::Error::InvalidArgument {
                field: "input_strength",
                reason: "input strength must be finite".into(),
            });
        }
        let model = Model::load(&parameters.model_path)?;
        if model.kind() != ModelKind::Regression {
            return Err(crate::Error::InvalidSchema(
                "regression interactive executor requires a regression model".into(),
            ));
        }
        let NetworkDocument::Geometry(network) = model.network() else {
            return Err(crate::Error::InvalidSchema(
                "regression network is missing".into(),
            ));
        };
        let (model_parameters, audio_parameters) = match (&network.params, &network.audio_params) {
            (GeometryParameters::Regression(value), GeometryAudioParameters::Regression(audio)) => {
                (value, audio)
            }
            _ => {
                return Err(crate::Error::InvalidSchema(
                    "regression network schema mismatch".into(),
                ));
            }
        };
        let contract = RegressionContract::new(
            model_parameters,
            audio_parameters,
            parameters.frame_rate.numerator(),
            parameters.frame_rate.denominator(),
        )?;
        validate_regression_inputs(
            &parameters.common.audio,
            &parameters.common.emotions,
            &contract,
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
        let audio = Arc::clone(&parameters.common.audio);
        let emotions = Arc::clone(&parameters.common.emotions);
        let execution = RegressionGeometryInteractiveExecution::new(
            backend,
            contract.clone(),
            processor,
            Arc::clone(&audio),
            Arc::clone(&emotions),
            load_implicit_emotion(
                &model,
                parameters.source_emotion_shot.as_deref(),
                parameters.source_emotion_frame,
                contract.implicit_emotion_size,
            )?,
            parameters.input_strength,
        )?;
        let stream = device.create_stream()?;
        let skin = device.allocate(contract.result_skin_size)?;
        let tongue = device.allocate(contract.result_tongue_size)?;
        let jaw = device.allocate(16)?;
        let eyes = device.allocate(6)?;
        let core_interrupt = execution.interrupt_handle();
        Ok(Self {
            execution,
            contract,
            audio,
            emotions,
            stream,
            skin,
            tongue,
            jaw,
            eyes,
            batch_size: parameters.batch_size,
            core_interrupt,
            interrupt_handle: crate::audio2x::InteractiveInterruptHandle::new(),
            _not_sync: Cell::new(()),
        })
    }

    pub fn total_frames(&self) -> crate::Result<usize> {
        self.execution.total_frames()
    }
    pub fn sample_rate(&self) -> usize {
        self.execution.sampling_rate()
    }
    pub fn frame_rate(&self) -> FrameRate {
        let (numerator, denominator) = self.execution.frame_rate();
        FrameRate::new(numerator, denominator).expect("validated model frame rate")
    }
    pub fn frame_timestamp(&self, frame: usize) -> crate::Result<i64> {
        self.execution.frame_timestamp(frame)
    }
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
    pub fn interrupt_handle(&self) -> crate::audio2x::InteractiveInterruptHandle {
        self.interrupt_handle.clone()
    }
    pub fn cuda_stream(&self) -> &crate::cuda::CudaStream {
        &self.stream
    }
    pub fn audio_accumulator(&self) -> &Arc<crate::audio2x::AudioAccumulator> {
        &self.audio
    }
    pub fn emotion_accumulator(&self) -> &Arc<crate::audio2x::EmotionAccumulator> {
        &self.emotions
    }
    pub fn invalidate(
        &mut self,
        layer: crate::audio2face::GeometryInvalidationLayer,
    ) -> crate::Result<()> {
        self.execution.invalidate(map_invalidation_layer(layer));
        Ok(())
    }
    pub fn is_valid(&self, layer: crate::audio2face::GeometryInvalidationLayer) -> bool {
        self.execution.is_valid(map_invalidation_layer(layer))
    }
    pub fn set_input_strength(&mut self, value: f32) -> crate::Result<()> {
        self.execution.set_input_strength(value)
    }
    pub fn skin_geometry_size(&self) -> usize {
        self.contract.result_skin_size
    }
    pub fn tongue_geometry_size(&self) -> usize {
        self.contract.result_tongue_size
    }
    pub fn jaw_transform_size(&self) -> usize {
        16
    }
    pub fn eyes_rotation_size(&self) -> usize {
        6
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_geometry(
        skin: &mut crate::cuda::DeviceBuffer<f32>,
        tongue: &mut crate::cuda::DeviceBuffer<f32>,
        jaw: &mut crate::cuda::DeviceBuffer<f32>,
        eyes: &mut crate::cuda::DeviceBuffer<f32>,
        stream: &crate::cuda::CudaStream,
        metadata: crate::animation::InteractiveGeometryMetadata,
        geometry: &RegressionGeometry,
        callback: &mut (
                 dyn for<'r> FnMut(crate::audio2face::GeometryResults<'r>) -> ControlFlow<()> + Send
             ),
    ) -> crate::Result<bool> {
        skin.copy_from(&geometry.skin, stream)?;
        tongue.copy_from(&geometry.tongue, stream)?;
        jaw.copy_from(&geometry.jaw_transform, stream)?;
        let eyes_host = [
            geometry.eyes_rotation.right[0],
            geometry.eyes_rotation.right[1],
            geometry.eyes_rotation.right[2],
            geometry.eyes_rotation.left[0],
            geometry.eyes_rotation.left[1],
            geometry.eyes_rotation.left[2],
        ];
        eyes.copy_from(&eyes_host, stream)?;
        let metadata = crate::audio2x::CallbackMetadata {
            track_index: 0,
            frame_index: metadata.frame,
            timestamp: metadata.timestamp,
            next_timestamp: metadata.next_timestamp,
        };
        Ok(matches!(
            callback(crate::audio2face::GeometryResults {
                metadata,
                skin: Some(crate::audio2x::DeviceComponentResults {
                    values: skin.view(),
                    stream: stream.as_ref(),
                }),
                tongue: Some(crate::audio2x::DeviceComponentResults {
                    values: tongue.view(),
                    stream: stream.as_ref(),
                }),
                jaw: Some(crate::audio2x::DeviceComponentResults {
                    values: jaw.view(),
                    stream: stream.as_ref(),
                }),
                eyes: Some(crate::audio2x::DeviceComponentResults {
                    values: eyes.view(),
                    stream: stream.as_ref(),
                }),
            }),
            ControlFlow::Continue(())
        ))
    }

    fn compute_frame_sync_with_generation(
        &mut self,
        frame: usize,
        callback: &mut (
                 dyn for<'r> FnMut(crate::audio2face::GeometryResults<'r>) -> ControlFlow<()> + Send
             ),
        generation: u64,
        stateful: bool,
    ) -> crate::Result<crate::audio2x::InteractiveExecutionReport> {
        let mut callback_error = None;
        let mut emitted_frames = 0;
        let stream = &self.stream;
        let skin = &mut self.skin;
        let tongue = &mut self.tongue;
        let jaw = &mut self.jaw;
        let eyes = &mut self.eyes;
        let core_interrupt = self.core_interrupt.clone();
        let interrupt_handle = self.interrupt_handle.clone();
        let mut emit = |metadata, geometry: &RegressionGeometry| {
            if interrupt_handle.is_interrupted_since(generation) {
                core_interrupt.interrupt();
                return false;
            }
            if callback_error.is_some() {
                return false;
            }
            let result = Self::emit_geometry(
                skin, tongue, jaw, eyes, stream, metadata, geometry, callback,
            );
            match result {
                Ok(continue_compute) => {
                    emitted_frames += 1;
                    if interrupt_handle.is_interrupted_since(generation) {
                        core_interrupt.interrupt();
                        false
                    } else {
                        continue_compute
                    }
                }
                Err(error) => {
                    callback_error = Some(error);
                    false
                }
            }
        };
        let status = if stateful {
            self.execution.compute_frame_stateful(frame, &mut emit)?
        } else {
            self.execution.compute_frame(frame, &mut emit)?
        };
        if let Some(error) = callback_error {
            return Err(error);
        }
        let status = match status {
            crate::animation::InteractiveGeometryStatus::Complete { .. } => {
                crate::audio2x::InteractiveExecutionStatus::Complete
            }
            crate::animation::InteractiveGeometryStatus::Interrupted { .. } => {
                crate::audio2x::InteractiveExecutionStatus::Interrupted
            }
        };
        Ok(crate::audio2x::InteractiveExecutionReport {
            status,
            emitted_frames,
        })
    }

    pub fn compute_frame<'a>(
        &'a mut self,
        frame: usize,
        callback: &'a mut (
                    dyn for<'r> FnMut(crate::audio2face::GeometryResults<'r>) -> ControlFlow<()>
                        + Send
                ),
    ) -> crate::audio2x::ExecutorFuture<'a, crate::audio2x::InteractiveExecutionReport> {
        Box::pin(async move {
            let generation = self.interrupt_handle.generation();
            YieldOnce(false).await;
            if self.interrupt_handle.is_interrupted_since(generation) {
                return Ok(crate::audio2x::InteractiveExecutionReport {
                    status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                    emitted_frames: 0,
                });
            }
            self.compute_frame_sync_with_generation(frame, callback, generation, false)
        })
    }

    pub fn compute_all_frames<'a>(
        &'a mut self,
        callback: &'a mut (
                    dyn for<'r> FnMut(crate::audio2face::GeometryResults<'r>) -> ControlFlow<()>
                        + Send
                ),
    ) -> crate::audio2x::ExecutorFuture<'a, crate::audio2x::InteractiveExecutionReport> {
        Box::pin(async move {
            let generation = self.interrupt_handle.generation();
            YieldOnce(false).await;
            let total = self.total_frames()?;
            if self.interrupt_handle.is_interrupted_since(generation) {
                return Ok(crate::audio2x::InteractiveExecutionReport {
                    status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                    emitted_frames: 0,
                });
            }
            self.execution.prepare_all()?;
            let mut emitted_frames = 0;
            for frame in 0..total {
                if self.interrupt_handle.is_interrupted_since(generation) {
                    return Ok(crate::audio2x::InteractiveExecutionReport {
                        status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                        emitted_frames,
                    });
                }
                let report =
                    self.compute_frame_sync_with_generation(frame, callback, generation, true)?;
                emitted_frames += report.emitted_frames;
                if report.status == crate::audio2x::InteractiveExecutionStatus::Interrupted {
                    return Ok(crate::audio2x::InteractiveExecutionReport {
                        status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                        emitted_frames,
                    });
                }
                YieldOnce(false).await;
            }
            Ok(crate::audio2x::InteractiveExecutionReport {
                status: crate::audio2x::InteractiveExecutionStatus::Complete,
                emitted_frames,
            })
        })
    }
}

/// Creates an owning interactive Regression geometry executor without
/// blocking the thread that polls the returned future.
#[cfg(feature = "tensorrt")]
pub fn create_regression_geometry_interactive_executor(
    parameters: RegressionGeometryInteractiveExecutorCreationParameters,
) -> crate::audio2x::ExecutorFuture<'static, RegressionGeometryInteractiveExecutor> {
    RegressionGeometryInteractiveExecutorFactory::load(parameters)
}

#[cfg(feature = "tensorrt")]
impl crate::audio2x::InteractiveExecutor for RegressionGeometryInteractiveExecutor {
    fn invalidate_all(&mut self) -> crate::Result<()> {
        self.invalidate(crate::audio2face::GeometryInvalidationLayer::All)
    }

    fn is_fully_valid(&self) -> bool {
        self.is_valid(crate::audio2face::GeometryInvalidationLayer::All)
    }

    fn total_frame_count(&self) -> crate::Result<usize> {
        self.total_frames()
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate()
    }

    fn frame_rate(&self) -> FrameRate {
        self.frame_rate()
    }

    fn frame_timestamp(&self, frame: usize) -> crate::Result<i64> {
        self.frame_timestamp(frame)
    }

    fn interrupt_handle(&self) -> crate::audio2x::InteractiveInterruptHandle {
        self.interrupt_handle()
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2face::GeometryInteractiveExecutor for RegressionGeometryInteractiveExecutor {
    fn invalidate_geometry(
        &mut self,
        layer: crate::audio2face::GeometryInvalidationLayer,
    ) -> crate::Result<()> {
        self.invalidate(layer)
    }

    fn is_geometry_valid(&self, layer: crate::audio2face::GeometryInvalidationLayer) -> bool {
        self.is_valid(layer)
    }

    fn skin_geometry_size(&self) -> usize {
        self.skin_geometry_size()
    }

    fn tongue_geometry_size(&self) -> usize {
        self.tongue_geometry_size()
    }

    fn jaw_transform_size(&self) -> usize {
        self.jaw_transform_size()
    }

    fn eyes_rotation_size(&self) -> usize {
        self.eyes_rotation_size()
    }

    fn compute_frame<'a>(
        &'a mut self,
        frame: usize,
        callback: &'a mut (
                    dyn for<'r> FnMut(crate::audio2face::GeometryResults<'r>) -> ControlFlow<()>
                        + Send
                ),
    ) -> crate::audio2x::ExecutorFuture<'a, crate::audio2x::InteractiveExecutionReport> {
        RegressionGeometryInteractiveExecutor::compute_frame(self, frame, callback)
    }

    fn compute_all_frames<'a>(
        &'a mut self,
        callback: &'a mut (
                    dyn for<'r> FnMut(crate::audio2face::GeometryResults<'r>) -> ControlFlow<()>
                        + Send
                ),
    ) -> crate::audio2x::ExecutorFuture<'a, crate::audio2x::InteractiveExecutionReport> {
        RegressionGeometryInteractiveExecutor::compute_all_frames(self, callback)
    }
}

#[cfg(feature = "tensorrt")]
fn map_invalidation_layer(
    layer: crate::audio2face::GeometryInvalidationLayer,
) -> crate::animation::GeometryInvalidationLayer {
    match layer {
        crate::audio2face::GeometryInvalidationLayer::None => {
            crate::animation::GeometryInvalidationLayer::None
        }
        crate::audio2face::GeometryInvalidationLayer::All => {
            crate::animation::GeometryInvalidationLayer::All
        }
        crate::audio2face::GeometryInvalidationLayer::Inference => {
            crate::animation::GeometryInvalidationLayer::Inference
        }
        crate::audio2face::GeometryInvalidationLayer::Skin => {
            crate::animation::GeometryInvalidationLayer::Skin
        }
        crate::audio2face::GeometryInvalidationLayer::Tongue => {
            crate::animation::GeometryInvalidationLayer::Tongue
        }
        crate::audio2face::GeometryInvalidationLayer::Teeth => {
            crate::animation::GeometryInvalidationLayer::Teeth
        }
        crate::audio2face::GeometryInvalidationLayer::Eyes => {
            crate::animation::GeometryInvalidationLayer::Eyes
        }
    }
}

#[cfg(feature = "tensorrt")]
impl RegressionGeometryExecutor {
    pub(crate) fn device_arc(&self) -> Arc<GpuDevice> {
        Arc::clone(&self.device)
    }

    pub fn cuda_stream(&self) -> &crate::cuda::CudaStream {
        &self.result_stream
    }

    pub fn audio_accumulator(
        &self,
        track: usize,
    ) -> crate::Result<&Arc<crate::audio2x::AudioAccumulator>> {
        self.tracks
            .get(track)
            .map(|resources| &resources.audio)
            .ok_or(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            })
    }

    pub fn emotion_accumulator(
        &self,
        track: usize,
    ) -> crate::Result<&Arc<crate::audio2x::EmotionAccumulator>> {
        self.tracks
            .get(track)
            .map(|resources| &resources.emotions)
            .ok_or(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: self.tracks.len(),
            })
    }

    pub fn set_input_strength(&mut self, value: f32) -> crate::Result<()> {
        if !value.is_finite() {
            return Err(crate::Error::InvalidArgument {
                field: "input_strength",
                reason: "input strength must be finite".into(),
            });
        }
        self.input_strength = value;
        Ok(())
    }

    pub fn set_implicit_emotion(&mut self, track: usize, values: &[f32]) -> crate::Result<()> {
        let track_count = self.implicit_emotions.len();
        let target = self
            .implicit_emotions
            .get_mut(track)
            .ok_or(crate::Error::OutOfBounds {
                field: "track",
                index: track,
                len: track_count,
            })?;
        if target.len() != values.len() || values.iter().any(|value| !value.is_finite()) {
            return Err(crate::Error::InvalidArgument {
                field: "implicit_emotion",
                reason: "implicit emotion dimensions or values are invalid".into(),
            });
        }
        target.copy_from_slice(values);
        Ok(())
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
    pub(crate) fn load_sync(
        parameters: RegressionGeometryExecutorCreationParameters,
    ) -> crate::Result<Self> {
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
        let dt =
            parameters.frame_rate.denominator() as f32 / parameters.frame_rate.numerator() as f32;
        let gpu_postprocessor = model_data.gpu_regression_postprocessor(
            &device,
            backend.stream(),
            config,
            track_count,
            dt,
            owned_contract.result_layout,
        )?;
        let execution = RegressionScheduler::new(contract, track_count)?;
        let skin_output =
            device.allocate(owned_contract.result_skin_size.saturating_mul(track_count))?;
        let tongue_output = device.allocate(
            owned_contract
                .result_tongue_size
                .saturating_mul(track_count),
        )?;
        let jaw_output = device.allocate(16_usize.saturating_mul(track_count))?;
        let eyes_output = device.allocate(6_usize.saturating_mul(track_count))?;
        let emotion_output = device.allocate(owned_contract.emotion_size)?;
        let result_stream = device.create_stream()?;
        Ok(Self {
            execution,
            backend,
            postprocessors: processors,
            gpu_postprocessor,
            contract: owned_contract,
            implicit_emotions: vec![
                load_implicit_emotion(
                    &model,
                    parameters.source_emotion_shot.as_deref(),
                    parameters.source_emotion_frame,
                    model_parameters.implicit_emotion_len
                )?;
                track_count
            ],
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
}

/// Creates an owning Regression geometry executor without blocking the thread
/// that polls the returned future.
#[cfg(feature = "tensorrt")]
pub fn create_regression_geometry_executor(
    parameters: RegressionGeometryExecutorCreationParameters,
) -> crate::audio2x::ExecutorFuture<'static, RegressionGeometryExecutor> {
    RegressionGeometryExecutorFactory::load(parameters)
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
        self.gpu_postprocessor
            .reset_track(track, self.backend.stream())?;
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
        let option = self.execution_option;
        let skin_output = &mut self.skin_output;
        let tongue_output = &mut self.tongue_output;
        let jaw_output = &mut self.jaw_output;
        let eyes_output = &mut self.eyes_output;
        let emotion_output = &mut self.emotion_output;
        let emotion_stream = &self.result_stream;
        let source_tracks = &self.tracks;
        let implicit_emotions = &self.implicit_emotions;
        let mut emitted_frames = 0;
        let mut active_tracks = vec![false; self.tracks.len()];
        let mut callback_error = None;
        let status = self.execution.pump_device(
            &tracks,
            &mut self.backend,
            &mut self.gpu_postprocessor,
            skin_output,
            tongue_output,
            jaw_output,
            eyes_output,
            |metadata, skin, tongue, jaw, eyes, stream| {
                active_tracks[metadata.track] = true;
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
