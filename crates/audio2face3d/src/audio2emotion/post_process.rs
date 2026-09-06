//! Post-process-only Audio2Emotion facade declarations.

use std::sync::Arc;

#[cfg(feature = "cuda")]
use std::ops::ControlFlow;
#[cfg(feature = "cuda")]
use std::task::Poll;

use crate::audio2emotion::{
    EmotionExecutorCreationParameters, EmotionInteractiveExecutorCreationParameters,
};
#[cfg(feature = "cuda")]
use crate::audio2x::InteractiveExecutor;
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

/// Canonical asynchronous factory for [`PostProcessEmotionExecutor`].
#[cfg(feature = "cuda")]
pub struct PostProcessEmotionExecutorFactory;

/// Canonical asynchronous factory for
/// [`PostProcessEmotionInteractiveExecutor`].
#[cfg(feature = "cuda")]
pub struct PostProcessEmotionInteractiveExecutorFactory;

#[cfg(feature = "cuda")]
impl PostProcessEmotionExecutorFactory {
    pub fn load(
        parameters: PostProcessEmotionExecutorCreationParameters,
    ) -> crate::audio2x::ExecutorFuture<'static, PostProcessEmotionExecutor> {
        crate::audio2x::spawn_blocking_factory(move || {
            PostProcessEmotionExecutor::load_sync(parameters)
        })
    }
}

#[cfg(feature = "cuda")]
impl PostProcessEmotionInteractiveExecutorFactory {
    pub fn load(
        parameters: PostProcessEmotionInteractiveExecutorCreationParameters,
    ) -> crate::audio2x::ExecutorFuture<'static, PostProcessEmotionInteractiveExecutor> {
        crate::audio2x::spawn_blocking_factory(move || {
            PostProcessEmotionInteractiveExecutor::load_sync(parameters)
        })
    }
}

/// Creates the owning standard post-process-only executor asynchronously.
#[cfg(feature = "cuda")]
pub fn create_post_process_emotion_executor(
    parameters: PostProcessEmotionExecutorCreationParameters,
) -> crate::audio2x::ExecutorFuture<'static, PostProcessEmotionExecutor> {
    PostProcessEmotionExecutorFactory::load(parameters)
}

/// Creates the owning interactive post-process-only executor asynchronously.
#[cfg(feature = "cuda")]
pub fn create_post_process_emotion_interactive_executor(
    parameters: PostProcessEmotionInteractiveExecutorCreationParameters,
) -> crate::audio2x::ExecutorFuture<'static, PostProcessEmotionInteractiveExecutor> {
    PostProcessEmotionInteractiveExecutorFactory::load(parameters)
}

/// Owning host emotion post-processor facade.
///
/// Corresponds to `nva2e::IPostProcessor` in
/// `audio2emotion-sdk/include/audio2emotion/postprocess.h` and replaces
/// `crate::emotion::EmotionPostProcessor` at the SDK-facing module boundary.
pub struct PostProcessor {
    inner: crate::emotion::EmotionPostProcessor,
    _not_sync: std::cell::Cell<()>,
}

/// Creates the host post-processor from facade values.
pub fn create_post_processor(
    data: PostProcessData,
    params: PostProcessParams,
) -> crate::Result<PostProcessor> {
    PostProcessor::new(data, params)
}

impl PostProcessor {
    /// Creates the host post-processor from the model-derived facade values.
    pub fn new(data: PostProcessData, params: PostProcessParams) -> crate::Result<Self> {
        Ok(Self {
            inner: crate::emotion::EmotionPostProcessor::new(
                into_internal_data(data),
                into_internal_params(params),
            )?,
            _not_sync: std::cell::Cell::new(()),
        })
    }

    /// Processes one inference emotion vector and returns host-owned output.
    pub fn process(&mut self, input: &[f32]) -> crate::Result<Vec<f32>> {
        self.inner.process(input)
    }

    /// Processes one inference vector using an optional preferred-emotion
    /// vector supplied by the caller.
    pub fn process_with_preferred(
        &mut self,
        input: &[f32],
        preferred_emotion: Option<&[f32]>,
    ) -> crate::Result<Vec<f32>> {
        self.inner.process_with_preferred(input, preferred_emotion)
    }

    /// Replaces runtime parameters after validating their dimensions.
    pub fn set_parameters(&mut self, params: PostProcessParams) -> crate::Result<()> {
        self.inner.set_parameters(into_internal_params(params))
    }

    /// Resets temporal smoothing state while retaining model parameters.
    pub fn reset(&mut self) {
        self.inner.reset();
    }
}

/// Non-generic owning post-process-only emotion executor facade.
///
/// Corresponds to `nva2e::IPostProcessModel::EmotionExecutor` and implements
/// `nva2e::IEmotionExecutor` from
/// `audio2emotion-sdk/include/audio2emotion/executor.h`.
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
/// `audio2emotion-sdk/include/audio2emotion/interactive_executor.h`. The
/// implementation details remain private at the facade boundary.
#[cfg(feature = "cuda")]
pub struct PostProcessEmotionInteractiveExecutor {
    inner: crate::emotion::InteractivePostProcessEmotionExecutor,
    contract: crate::emotion::PostProcessEmotionContract,
    audio: Arc<crate::audio2x::AudioAccumulator>,
    preferred_emotions: Option<Arc<EmotionAccumulator>>,
    frame_rate: FrameRate,
    output: crate::cuda::DeviceBuffer<f32>,
    stream: crate::cuda::CudaStream,
    interrupt: crate::audio2x::InteractiveInterruptHandle,
    post_processing_valid: bool,
    _not_sync: std::cell::Cell<()>,
}

#[cfg(feature = "cuda")]
impl PostProcessEmotionExecutor {
    /// Creates an owning post-process-only executor and its device result stream.
    pub(crate) fn load_sync(
        parameters: PostProcessEmotionExecutorCreationParameters,
    ) -> crate::Result<Self> {
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

    /// Returns the stream used for device result copies.
    pub fn cuda_stream(&self) -> &crate::cuda::CudaStream {
        &self.stream
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
impl PostProcessEmotionInteractiveExecutor {
    pub(crate) fn load_sync(
        parameters: PostProcessEmotionInteractiveExecutorCreationParameters,
    ) -> crate::Result<Self> {
        if parameters.batch_size == 0 {
            return Err(crate::Error::InvalidArgument {
                field: "batch_size",
                reason: "must be non-zero".into(),
            });
        }
        let data = into_internal_data(parameters.post_process_data);
        validate_interactive_inputs(
            &parameters.common.audio,
            parameters.common.preferred_emotions.as_deref(),
            data.output_emotion_length,
        )?;
        let contract = crate::emotion::PostProcessEmotionContract::new(
            parameters.sample_rate,
            parameters.frame_rate.numerator(),
            parameters.frame_rate.denominator(),
        )?;
        let inner = crate::emotion::InteractivePostProcessEmotionExecutor::new(
            contract.clone(),
            data.clone(),
            into_internal_params(parameters.post_process_params),
        )?;
        let device = crate::cuda::GpuDevice::new(parameters.common.device_ordinal)?;
        let stream = device.create_stream()?;
        let output = device.allocate(data.output_emotion_length)?;
        Ok(Self {
            inner,
            contract,
            audio: parameters.common.audio,
            preferred_emotions: parameters.common.preferred_emotions,
            frame_rate: parameters.frame_rate,
            output,
            stream,
            interrupt: crate::audio2x::InteractiveInterruptHandle::new(),
            post_processing_valid: false,
            _not_sync: std::cell::Cell::new(()),
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
        let core_interrupt = self.inner.interrupt_handle();
        let output = &mut self.output;
        let stream = &self.stream;
        let preferred = self.preferred_emotions.as_deref();
        let mut copy_error = None;
        let status =
            self.inner
                .compute_frame(frame, &self.audio, preferred, |metadata, values| {
                    if interrupt.is_interrupted_since(generation) {
                        core_interrupt.interrupt();
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

#[cfg(feature = "cuda")]
impl crate::audio2x::InteractiveExecutor for PostProcessEmotionInteractiveExecutor {
    fn invalidate_all(&mut self) -> crate::Result<()> {
        self.inner
            .invalidate(crate::emotion::PostProcessEmotionLayer::All);
        self.post_processing_valid = false;
        Ok(())
    }

    fn is_fully_valid(&self) -> bool {
        self.inner
            .is_valid(crate::emotion::PostProcessEmotionLayer::All)
            && self.post_processing_valid
    }

    fn total_frame_count(&self) -> crate::Result<usize> {
        validate_interactive_inputs(
            &self.audio,
            self.preferred_emotions.as_deref(),
            self.inner.output_emotion_length(),
        )?;
        self.inner.frame_count(&self.audio)
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
    fn interrupt_handle(&self) -> crate::audio2x::InteractiveInterruptHandle {
        self.interrupt.clone()
    }
}

#[cfg(feature = "cuda")]
impl crate::audio2emotion::EmotionInteractiveExecutor for PostProcessEmotionInteractiveExecutor {
    fn invalidate_emotion(
        &mut self,
        layer: crate::audio2emotion::EmotionInvalidationLayer,
    ) -> crate::Result<()> {
        let layer = match layer {
            crate::audio2emotion::EmotionInvalidationLayer::None => {
                crate::emotion::PostProcessEmotionLayer::None
            }
            crate::audio2emotion::EmotionInvalidationLayer::Inference => {
                crate::emotion::PostProcessEmotionLayer::Inference
            }
            crate::audio2emotion::EmotionInvalidationLayer::PostProcessing => {
                crate::emotion::PostProcessEmotionLayer::PostProcessing
            }
            crate::audio2emotion::EmotionInvalidationLayer::All => {
                crate::emotion::PostProcessEmotionLayer::All
            }
        };
        self.inner.invalidate(layer);
        if !matches!(layer, crate::emotion::PostProcessEmotionLayer::None) {
            self.post_processing_valid = false;
        }
        Ok(())
    }

    fn is_emotion_valid(&self, layer: crate::audio2emotion::EmotionInvalidationLayer) -> bool {
        let layer = match layer {
            crate::audio2emotion::EmotionInvalidationLayer::None => {
                crate::emotion::PostProcessEmotionLayer::None
            }
            crate::audio2emotion::EmotionInvalidationLayer::Inference => {
                crate::emotion::PostProcessEmotionLayer::Inference
            }
            crate::audio2emotion::EmotionInvalidationLayer::PostProcessing => {
                crate::emotion::PostProcessEmotionLayer::PostProcessing
            }
            crate::audio2emotion::EmotionInvalidationLayer::All => {
                crate::emotion::PostProcessEmotionLayer::All
            }
        };
        self.inner.is_valid(layer)
            && (layer != crate::emotion::PostProcessEmotionLayer::PostProcessing
                || self.post_processing_valid)
    }

    fn emotion_count(&self) -> usize {
        self.inner.output_emotion_length()
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

#[cfg(feature = "cuda")]
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

#[cfg(feature = "cuda")]
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
