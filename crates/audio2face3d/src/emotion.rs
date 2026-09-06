#![cfg_attr(not(feature = "tensorrt"), allow(unused_imports))]

//! Internal Audio2Emotion implementation components.
//!
//! The supported public API is exposed through [`crate::audio2emotion`].

mod binder;
mod executor;
#[cfg(feature = "cuda")]
mod gpu_postprocess;
mod interactive;
mod postprocess;
mod postprocess_executor;
#[cfg(feature = "tensorrt")]
mod tensorrt_backend;

pub use binder::EmotionBinder;
pub(crate) use executor::{ClassifierBackend, ClassifierScheduler, EmotionTrack};
pub use executor::{ClassifierContract, EmotionCallbackMetadata, EmotionExecutionStatus};
#[cfg(feature = "cuda")]
pub(crate) use gpu_postprocess::GpuEmotionPostProcessor;
pub(crate) use interactive::ClassifierInteractiveExecution;
pub use interactive::InteractiveEmotionStatus;
pub(crate) use postprocess::{
    EmotionPostProcessData, EmotionPostProcessParameters, EmotionPostProcessor,
};
pub(crate) use postprocess_executor::{
    InteractivePostProcessEmotionExecutor, PostProcessEmotionContract, PostProcessEmotionExecutor,
    PostProcessEmotionLayer, PostProcessEmotionTrack,
};
#[cfg(feature = "tensorrt")]
pub(crate) use tensorrt_backend::TensorRtClassifierBackend;
