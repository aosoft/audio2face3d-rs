//! Audio2Emotion pipeline components.

mod binder;
mod executor;
#[cfg(feature = "cuda")]
mod gpu_postprocess;
mod interactive;
mod postprocess;
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
#[cfg(feature = "tensorrt")]
pub use tensorrt_backend::TensorRtClassifierBackend;

pub const PIPELINE_NAME: &str = "audio2emotion";
