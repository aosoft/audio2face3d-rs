//! Audio2Emotion pipeline components.
//!
//! Classifier-backed execution is provided by [`EmotionExecutor`] and
//! [`InteractiveEmotionExecutor`]. For manual/preferred-emotion animation
//! without inference, use [`PostProcessEmotionExecutor`] or the owning
//! [`PostProcessEmotionExecutorBundle`]. Streaming executors return
//! [`EmotionExecutionStatus::AwaitingInput`] until the audio duration and any
//! enabled preferred-emotion data are available, then advance until
//! [`EmotionExecutionStatus::Complete`]. Interactive execution requires
//! closed, non-dropped accumulators and replays temporal post-process state
//! from frame zero.

mod binder;
mod executor;
#[cfg(feature = "cuda")]
mod gpu_postprocess;
mod interactive;
mod postprocess;
mod postprocess_executor;
mod postprocess_model;
#[cfg(feature = "tensorrt")]
mod tensorrt_backend;

pub use binder::EmotionBinder;
pub use executor::{
    ClassifierBackend, ClassifierContract, EmotionCallbackMetadata, EmotionExecutionStatus,
    EmotionExecutor, EmotionTrack,
};
#[cfg(feature = "cuda")]
pub use gpu_postprocess::{
    GpuEmotionKernelPath, GpuEmotionPostProcessFence, GpuEmotionPostProcessor,
};
pub use interactive::{InteractiveEmotionExecutor, InteractiveEmotionStatus};
pub use postprocess::{EmotionPostProcessData, EmotionPostProcessParameters, EmotionPostProcessor};
pub use postprocess_executor::{
    InteractivePostProcessEmotionExecutor, PostProcessEmotionContract, PostProcessEmotionExecutor,
    PostProcessEmotionInterrupt, PostProcessEmotionLayer, PostProcessEmotionTrack,
};
pub use postprocess_model::{
    PostProcessEmotionBundleOptions, PostProcessEmotionExecutorBundle, PostProcessEmotionModel,
};
#[cfg(feature = "tensorrt")]
pub use tensorrt_backend::TensorRtClassifierBackend;

pub const PIPELINE_NAME: &str = "audio2emotion";
