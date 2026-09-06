//! Classifier-model Audio2Emotion facade declarations.

use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "tensorrt")]
use std::cell::Cell;
#[cfg(feature = "tensorrt")]
use std::ops::ControlFlow;
#[cfg(feature = "tensorrt")]
use std::task::Poll;

use crate::audio2emotion::{
    EmotionExecutorCreationParameters, EmotionInteractiveExecutorCreationParameters,
    PostProcessData, PostProcessParams,
};
#[cfg(feature = "tensorrt")]
use crate::audio2x::InteractiveExecutor;
use crate::audio2x::{EmotionAccumulator, FrameRate};

#[cfg(feature = "tensorrt")]
use crate::common::NetworkDocument;
#[cfg(feature = "tensorrt")]
use crate::cuda::GpuDevice;
#[cfg(feature = "tensorrt")]
use crate::emotion::{
    ClassifierContract, ClassifierScheduler, EmotionTrack, TensorRtClassifierBackend,
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

/// Canonical asynchronous factory for [`ClassifierEmotionExecutor`].
#[cfg(feature = "tensorrt")]
pub struct ClassifierEmotionExecutorFactory;

/// Canonical asynchronous factory for
/// [`ClassifierEmotionInteractiveExecutor`].
#[cfg(feature = "tensorrt")]
pub struct ClassifierEmotionInteractiveExecutorFactory;

#[cfg(feature = "tensorrt")]
impl ClassifierEmotionExecutorFactory {
    pub fn load(
        parameters: ClassifierEmotionExecutorCreationParameters,
    ) -> crate::audio2x::ExecutorFuture<'static, ClassifierEmotionExecutor> {
        crate::audio2x::spawn_blocking_factory(move || {
            ClassifierEmotionExecutor::load_sync(parameters)
        })
    }
}

#[cfg(feature = "tensorrt")]
impl ClassifierEmotionInteractiveExecutorFactory {
    pub fn load(
        parameters: ClassifierEmotionInteractiveExecutorCreationParameters,
    ) -> crate::audio2x::ExecutorFuture<'static, ClassifierEmotionInteractiveExecutor> {
        crate::audio2x::spawn_blocking_factory(move || {
            ClassifierEmotionInteractiveExecutor::load_sync(parameters)
        })
    }
}

/// Creates the owning standard classifier executor asynchronously.
#[cfg(feature = "tensorrt")]
pub fn create_classifier_emotion_executor(
    parameters: ClassifierEmotionExecutorCreationParameters,
) -> crate::audio2x::ExecutorFuture<'static, ClassifierEmotionExecutor> {
    ClassifierEmotionExecutorFactory::load(parameters)
}

/// Creates the owning interactive classifier executor asynchronously.
#[cfg(feature = "tensorrt")]
pub fn create_classifier_emotion_interactive_executor(
    parameters: ClassifierEmotionInteractiveExecutorCreationParameters,
) -> crate::audio2x::ExecutorFuture<'static, ClassifierEmotionInteractiveExecutor> {
    ClassifierEmotionInteractiveExecutorFactory::load(parameters)
}

/// Non-generic owning classifier emotion executor facade.
///
/// Corresponds to `nva2e::IClassifierModel::EmotionExecutor` and implements
/// `nva2e::IEmotionExecutor` from
/// `audio2emotion-sdk/include/audio2emotion/executor.h`. The classifier backend
/// is a private implementation detail.
#[cfg(feature = "tensorrt")]
pub struct ClassifierEmotionExecutor {
    execution: ClassifierScheduler,
    backend: TensorRtClassifierBackend,
    contract: ClassifierContract,
    tracks: Vec<crate::audio2emotion::EmotionTrackResources>,
    preferred_emotions: Vec<Arc<EmotionAccumulator>>,
    input_strength: f32,
    frame_rate: FrameRate,
    sample_rate: usize,
    gpu_processor: crate::emotion::GpuEmotionPostProcessor,
    device_output: crate::cuda::DeviceBuffer<f32>,
}

/// Non-generic owning interactive classifier emotion executor facade.
///
/// Corresponds to `nva2e::IClassifierModel::EmotionInteractiveExecutor` and
/// `nva2e::IEmotionInteractiveExecutor` from
/// `audio2emotion-sdk/include/audio2emotion/interactive_executor.h`.
#[cfg(feature = "tensorrt")]
pub struct ClassifierEmotionInteractiveExecutor {
    inner: crate::emotion::ClassifierInteractiveExecution<TensorRtClassifierBackend>,
    contract: ClassifierContract,
    audio: Arc<crate::audio2x::AudioAccumulator>,
    preferred_emotions: Option<Arc<EmotionAccumulator>>,
    frame_rate: FrameRate,
    sample_rate: usize,
    output: crate::cuda::DeviceBuffer<f32>,
    stream: crate::cuda::CudaStream,
    interrupt: crate::audio2x::InteractiveInterruptHandle,
    post_processing_valid: bool,
    _not_sync: Cell<()>,
}

#[cfg(feature = "tensorrt")]
impl ClassifierEmotionExecutor {
    /// Loads an owning classifier executor. The model, TensorRT backend, and
    /// post-processing state are all moved into this value; no `Model` borrow
    /// is retained after construction.
    pub(crate) fn load_sync(
        parameters: ClassifierEmotionExecutorCreationParameters,
    ) -> crate::Result<Self> {
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
        let gpu_parameters = vec![model_parameters.clone(); parameters.common.tracks.len()];
        let gpu_processor = crate::emotion::GpuEmotionPostProcessor::new(
            &device,
            backend.stream(),
            data.clone(),
            &gpu_parameters,
        )?;
        let execution = ClassifierScheduler::new(
            contract.clone(),
            data,
            model_parameters,
            parameters.common.tracks.len(),
        )?;
        let device_output = device.allocate(
            execution
                .output_emotion_length()
                .checked_mul(parameters.common.tracks.len())
                .ok_or(crate::Error::IntegerOverflow {
                    field: "emotion_device_output",
                    value: parameters.common.tracks.len(),
                    target: "usize",
                })?,
        )?;
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
            gpu_processor,
            device_output,
        })
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
    /// Returns the stream owned by the classifier backend.
    pub fn cuda_stream(&self) -> &crate::cuda::CudaStream {
        self.backend.stream()
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

    /// Borrows the caller-owned audio accumulator retained by this executor.
    pub fn audio_accumulator(
        &self,
        track: usize,
    ) -> crate::Result<&Arc<crate::audio2x::AudioAccumulator>> {
        self.audio(track)
    }

    /// Borrows the preferred-emotion accumulator configured for a track.
    pub fn emotion_accumulator(
        &self,
        track: usize,
    ) -> crate::Result<&Arc<crate::audio2x::EmotionAccumulator>> {
        self.preferred_emotions
            .get(track)
            .ok_or(crate::Error::OutOfBounds {
                field: "emotion_accumulator",
                index: track,
                len: self.preferred_emotions.len(),
            })
    }
    pub fn emotion_count(&self) -> usize {
        self.execution.output_emotion_length()
    }

    pub fn reset_track(&mut self, track: usize) -> crate::Result<()> {
        <Self as crate::audio2x::Executor>::reset_track(self, track)
    }
}

#[cfg(feature = "tensorrt")]
impl ClassifierEmotionInteractiveExecutor {
    /// Loads the single-track interactive classifier facade.
    pub(crate) fn load_sync(
        parameters: ClassifierEmotionInteractiveExecutorCreationParameters,
    ) -> crate::Result<Self> {
        if parameters.batch_size == 0 {
            return Err(crate::Error::InvalidArgument {
                field: "batch_size",
                reason: "must be non-zero".into(),
            });
        }
        let model = Model::load(&parameters.model_path)?;
        if model.kind() != ModelKind::Emotion {
            return Err(crate::Error::InvalidSchema(
                "classifier interactive executor requires an emotion model".into(),
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
        validate_interactive_inputs(
            &parameters.common.audio,
            parameters.common.preferred_emotions.as_deref(),
            data.output_emotion_length,
        )?;
        if data.output_emotion_length != parameters.post_process_data.output_emotion_length
            && parameters.post_process_data.output_emotion_length != 0
        {
            return Err(crate::Error::SizeMismatch {
                field: "post_process_data.output_emotion_length",
                expected: data.output_emotion_length,
                actual: parameters.post_process_data.output_emotion_length,
            });
        }
        let device = GpuDevice::new(parameters.common.device_ordinal)?;
        let (backend, contract) = TensorRtClassifierBackend::load_interactive(
            Arc::clone(&device),
            model.engine_path(),
            network.audio_params.samplerate,
            network.emotions.len(),
            parameters.frame_rate.numerator(),
            parameters.frame_rate.denominator(),
            parameters.inferences_to_skip,
        )?;
        if parameters.batch_size > backend.max_batch_size() {
            return Err(crate::Error::InvalidArgument {
                field: "batch_size",
                reason: format!(
                    "must not exceed classifier engine maximum {}",
                    backend.max_batch_size()
                ),
            });
        }
        let mut inner = crate::emotion::ClassifierInteractiveExecution::new(
            backend,
            contract.clone(),
            data.clone(),
            model_parameters,
        )?;
        inner.set_input_strength(parameters.input_strength)?;
        inner.set_batch_size(parameters.batch_size)?;
        let output = device.allocate(data.output_emotion_length)?;
        let stream = device.create_stream()?;
        Ok(Self {
            inner,
            contract,
            audio: Arc::clone(&parameters.common.audio),
            preferred_emotions: parameters.common.preferred_emotions,
            frame_rate: parameters.frame_rate,
            sample_rate: network.audio_params.samplerate,
            output,
            stream,
            interrupt: crate::audio2x::InteractiveInterruptHandle::new(),
            post_processing_valid: false,
            _not_sync: Cell::new(()),
        })
    }

    /// Returns the stream used for device result copies.
    pub fn cuda_stream(&self) -> &crate::cuda::CudaStream {
        &self.stream
    }

    /// Borrows the closed audio timeline retained by this executor.
    pub fn audio_accumulator(&self) -> &Arc<crate::audio2x::AudioAccumulator> {
        &self.audio
    }

    /// Borrows the optional preferred-emotion timeline, when configured.
    pub fn emotion_accumulator(&self) -> crate::Result<&Arc<crate::audio2x::EmotionAccumulator>> {
        self.preferred_emotions
            .as_ref()
            .ok_or(crate::Error::InvalidState {
                operation: "access emotion accumulator",
                state: "preferred emotions are not configured",
            })
    }

    fn compute_one(
        &mut self,
        frame: usize,
        generation: u64,
        callback: &mut (
                 dyn for<'r> FnMut(crate::audio2emotion::EmotionResults<'r>) -> ControlFlow<()>
                     + Send
             ),
    ) -> crate::Result<crate::emotion::InteractiveEmotionStatus> {
        let interrupt = self.interrupt.clone();
        let output = &mut self.output;
        let stream = &self.stream;
        let preferred = self.preferred_emotions.as_deref();
        let mut copy_error = None;
        let status =
            self.inner
                .compute_frame(frame, &self.audio, preferred, |metadata, values| {
                    if interrupt.is_interrupted_since(generation) {
                        return false;
                    }
                    if let Err(error) = output.copy_from(values, stream) {
                        copy_error = Some(error);
                        return false;
                    }
                    let keep_going = matches!(
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
                    );
                    keep_going && !interrupt.is_interrupted_since(generation)
                })?;
        if let Some(error) = copy_error {
            return Err(error);
        }
        Ok(status)
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2x::InteractiveExecutor for ClassifierEmotionInteractiveExecutor {
    fn invalidate_all(&mut self) -> crate::Result<()> {
        self.inner.invalidate_audio();
        self.post_processing_valid = false;
        Ok(())
    }

    fn is_fully_valid(&self) -> bool {
        self.inner.inference_cache_is_valid() && self.post_processing_valid
    }

    fn total_frame_count(&self) -> crate::Result<usize> {
        validate_interactive_inputs(
            &self.audio,
            self.preferred_emotions.as_deref(),
            self.inner.emotion_count(),
        )?;
        self.inner.frame_count(&self.audio)
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

    fn interrupt_handle(&self) -> crate::audio2x::InteractiveInterruptHandle {
        self.interrupt.clone()
    }
}

#[cfg(feature = "tensorrt")]
impl crate::audio2emotion::EmotionInteractiveExecutor for ClassifierEmotionInteractiveExecutor {
    fn invalidate_emotion(
        &mut self,
        layer: crate::audio2emotion::EmotionInvalidationLayer,
    ) -> crate::Result<()> {
        match layer {
            crate::audio2emotion::EmotionInvalidationLayer::None => {}
            crate::audio2emotion::EmotionInvalidationLayer::Inference
            | crate::audio2emotion::EmotionInvalidationLayer::All => {
                self.inner.invalidate_audio();
                self.post_processing_valid = false;
            }
            crate::audio2emotion::EmotionInvalidationLayer::PostProcessing => {
                self.post_processing_valid = false;
            }
        }
        Ok(())
    }

    fn is_emotion_valid(&self, layer: crate::audio2emotion::EmotionInvalidationLayer) -> bool {
        match layer {
            crate::audio2emotion::EmotionInvalidationLayer::None => true,
            crate::audio2emotion::EmotionInvalidationLayer::Inference => {
                self.inner.inference_cache_is_valid()
            }
            crate::audio2emotion::EmotionInvalidationLayer::PostProcessing => {
                self.post_processing_valid
            }
            crate::audio2emotion::EmotionInvalidationLayer::All => {
                <Self as crate::audio2x::InteractiveExecutor>::is_fully_valid(self)
            }
        }
    }

    fn emotion_count(&self) -> usize {
        self.inner.emotion_count()
    }

    fn compute_frame<'a>(
        &'a mut self,
        frame: usize,
        callback: &'a mut (
                    dyn for<'r> FnMut(crate::audio2emotion::EmotionResults<'r>) -> ControlFlow<()>
                        + Send
                ),
    ) -> crate::audio2x::ExecutorFuture<'a, crate::audio2x::InteractiveExecutionReport> {
        Box::pin(async move {
            self.post_processing_valid = false;
            let total = self.total_frame_count()?;
            if frame >= total {
                return Err(crate::Error::OutOfBounds {
                    field: "frame",
                    index: frame,
                    len: total,
                });
            }
            let generation = self.interrupt.generation();
            match self.compute_one(frame, generation, callback)? {
                crate::emotion::InteractiveEmotionStatus::Complete { frames } => {
                    self.post_processing_valid = false;
                    Ok(crate::audio2x::InteractiveExecutionReport {
                        status: crate::audio2x::InteractiveExecutionStatus::Complete,
                        emitted_frames: frames,
                    })
                }
                crate::emotion::InteractiveEmotionStatus::Interrupted { frames } => {
                    self.post_processing_valid = false;
                    Ok(crate::audio2x::InteractiveExecutionReport {
                        status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                        emitted_frames: frames,
                    })
                }
            }
        })
    }

    fn compute_all_frames<'a>(
        &'a mut self,
        callback: &'a mut (
                    dyn for<'r> FnMut(crate::audio2emotion::EmotionResults<'r>) -> ControlFlow<()>
                        + Send
                ),
    ) -> crate::audio2x::ExecutorFuture<'a, crate::audio2x::InteractiveExecutionReport> {
        Box::pin(async move {
            self.post_processing_valid = false;
            let total = self.total_frame_count()?;
            let generation = self.interrupt.generation();
            let mut emitted = 0;
            for frame in 0..total {
                match self.compute_one(frame, generation, callback)? {
                    crate::emotion::InteractiveEmotionStatus::Complete { frames } => {
                        emitted += frames;
                    }
                    crate::emotion::InteractiveEmotionStatus::Interrupted { frames } => {
                        emitted += frames;
                        self.post_processing_valid = false;
                        return Ok(crate::audio2x::InteractiveExecutionReport {
                            status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                            emitted_frames: emitted,
                        });
                    }
                }
                if frame + 1 < total {
                    yield_once().await;
                }
            }
            self.post_processing_valid = true;
            Ok(crate::audio2x::InteractiveExecutionReport {
                status: crate::audio2x::InteractiveExecutionStatus::Complete,
                emitted_frames: emitted,
            })
        })
    }
}

#[cfg(feature = "tensorrt")]
fn validate_interactive_inputs(
    audio: &crate::audio2x::AudioAccumulator,
    preferred: Option<&EmotionAccumulator>,
    output_length: usize,
) -> crate::Result<()> {
    if !audio.is_closed() || audio.nb_dropped_samples() != 0 {
        return Err(crate::Error::InputHistoryUnavailable { track: 0 });
    }
    if let Some(preferred) = preferred {
        let state = preferred.state();
        if !state.closed
            || state.dropped_emotions != 0
            || state.last_dropped_timestamp != i64::MIN
            || state.emotion_size != output_length
        {
            return Err(crate::Error::InputHistoryUnavailable { track: 0 });
        }
    }
    Ok(())
}

#[cfg(feature = "tensorrt")]
async fn yield_once() {
    let mut yielded = false;
    std::future::poll_fn(move |cx| {
        if yielded {
            Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    })
    .await;
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
        self.execution.reset(track)?;
        self.gpu_processor.reset_track(track, self.backend.stream())
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
        let processor = &mut self.gpu_processor;
        let mut emitted_frames = 0;
        let status = self.execution.execute_device(
            &tracks,
            &mut self.backend,
            processor,
            output,
            |metadata, values, stream| {
                emitted_frames += 1;
                matches!(
                    callback(crate::audio2emotion::EmotionResults {
                        metadata: crate::audio2x::CallbackMetadata {
                            track_index: metadata.track,
                            frame_index: metadata.frame,
                            timestamp: metadata.timestamp,
                            next_timestamp: metadata.next_timestamp,
                        },
                        emotions: crate::audio2x::DeviceComponentResults { values, stream },
                    }),
                    ControlFlow::Continue(())
                )
            },
        )?;
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
