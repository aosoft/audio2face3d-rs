//! Public Audio2Emotion executor contracts.
//!
//! This module corresponds to the public interfaces in
//! `audio2emotion-sdk/include/audio2emotion/executor.h` and
//! `audio2emotion-sdk/include/audio2emotion/interactive_executor.h`.
//!
//! # Migration map
//!
//! | Existing Rust API | SDK-facing declaration |
//! |---|---|
//! | `emotion::EmotionExecutor<B>` | `classifier::ClassifierEmotionExecutor` |
//! | `emotion::InteractiveEmotionExecutor<B>` | `classifier::ClassifierEmotionInteractiveExecutor` |
//! | `emotion::PostProcessEmotionExecutor` | `post_process::PostProcessEmotionExecutor` |
//! | `emotion::InteractivePostProcessEmotionExecutor` | `post_process::PostProcessEmotionInteractiveExecutor` |
//! | `emotion::EmotionPostProcessor` | [`post_process::PostProcessor`] |
//!
//! Completed executor declarations are feature-gated and remain in their
//! classifier or post-process child module. Backend types are not part of this
//! facade.

use std::ops::ControlFlow;
use std::sync::Arc;

use crate::audio2x::{
    AudioAccumulator, CallbackMetadata, DeviceComponentResults, EmotionAccumulator, Execution,
    Executor, ExecutorFuture, InteractiveExecutionReport, InteractiveExecutor, Result,
};

pub mod classifier;
pub mod post_process;

pub use post_process::{PostProcessData, PostProcessParams};

/// Shared input owned by one emotion track.
#[derive(Clone, Debug)]
pub struct EmotionTrackResources {
    pub audio: Arc<AudioAccumulator>,
}

/// Parameters common to all owning Audio2Emotion executors.
///
/// Corresponds to `nva2e::EmotionExecutorCreationParameters` in
/// `audio2emotion/executor.h`.
#[derive(Clone, Debug)]
pub struct EmotionExecutorCreationParameters {
    pub tracks: Vec<EmotionTrackResources>,
    pub device_ordinal: i32,
}

/// Closed, history-preserving input resources for one interactive emotion executor.
///
/// The completed factory requires `audio` to be closed with no dropped history.
/// `preferred_emotions`, when present, must have the same complete timeline
/// needed by post-processing.
#[derive(Clone, Debug)]
pub struct EmotionInteractiveExecutorCreationParameters {
    pub audio: Arc<AudioAccumulator>,
    pub preferred_emotions: Option<Arc<EmotionAccumulator>>,
    pub device_ordinal: i32,
}

/// Device-resident emotions produced for one frame.
///
/// Corresponds to `nva2e::IEmotionExecutor::Results` in
/// `audio2emotion-sdk/include/audio2emotion/executor.h`. The device view and
/// stream expire when the synchronous callback returns; continued use requires
/// an explicit caller-owned copy ordered on that stream.
pub struct EmotionResults<'a> {
    pub metadata: CallbackMetadata,
    pub emotions: DeviceComponentResults<'a>,
}

/// Common contract implemented by classifier and post-process-only executors.
///
/// Corresponds to `nva2e::IEmotionExecutor` in
/// `audio2emotion-sdk/include/audio2emotion/executor.h`. The callback runs on
/// the calling thread before `execute` returns. `ControlFlow::Break` suppresses
/// only subsequent frames for the same track in this call.
pub trait EmotionExecutor: Executor {
    fn emotion_count(&self) -> usize;
    fn next_audio_sample_to_read(&self, track: usize) -> Result<usize>;
    fn execute(
        &mut self,
        callback: &mut dyn for<'r> FnMut(EmotionResults<'r>) -> ControlFlow<()>,
    ) -> Result<Execution>;
}

/// Layer identifiers for interactive emotion invalidation.
///
/// Corresponds to the `kLayer*` constants of
/// `nva2e::IEmotionInteractiveExecutor` in
/// `audio2emotion-sdk/include/audio2emotion/interactive_executor.h`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(usize)]
pub enum EmotionInvalidationLayer {
    None = 0,
    All = 1,
    Inference = 2,
    PostProcessing = 3,
}

/// Interactive classifier/post-process-only emotion contract.
///
/// Corresponds to `nva2e::IEmotionInteractiveExecutor` in
/// `audio2emotion-sdk/include/audio2emotion/interactive_executor.h`. Each
/// compute method returns a runtime-independent `Send` future borrowing the
/// executor and callback. `Ready` means host computation and all callbacks for
/// the call have completed. Callback `Break` interrupts the entire current call
/// and produces `InteractiveExecutionStatus::Interrupted`.
pub trait EmotionInteractiveExecutor: InteractiveExecutor {
    fn invalidate_emotion(&mut self, layer: EmotionInvalidationLayer) -> Result<()>;
    fn is_emotion_valid(&self, layer: EmotionInvalidationLayer) -> bool;
    fn emotion_count(&self) -> usize;
    fn compute_frame<'a>(
        &'a mut self,
        frame: usize,
        callback: &'a mut (dyn for<'r> FnMut(EmotionResults<'r>) -> ControlFlow<()> + Send),
    ) -> ExecutorFuture<'a, InteractiveExecutionReport>;
    fn compute_all_frames<'a>(
        &'a mut self,
        callback: &'a mut (dyn for<'r> FnMut(EmotionResults<'r>) -> ControlFlow<()> + Send),
    ) -> ExecutorFuture<'a, InteractiveExecutionReport>;
}
