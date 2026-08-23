//! Audio2Face pipeline components.

mod animator;
#[cfg(feature = "cuda")]
mod device_postprocess;
mod executor;
mod jaw;
mod pca;
mod postprocess;
mod regression;
#[cfg(feature = "tensorrt")]
mod tensorrt_backend;

pub use animator::{
    EyesAnimator, EyesAnimatorParams, EyesRotation, SkinAnimator, SkinAnimatorParams,
    TongueAnimator, TongueAnimatorParams,
};
#[cfg(feature = "cuda")]
pub use device_postprocess::{
    GpuRegressionModel, GpuRegressionOutputs, GpuRegressionPostprocessFence,
    GpuRegressionPostprocessor, GpuRegressionTrackParams,
};
pub use executor::{
    MAX_REGRESSION_TRACKS, PumpStatus, RegressionBackend, RegressionCallbackMetadata,
    RegressionExecutor, RegressionExecutorState, RegressionTrack,
};
pub use jaw::{JawParameters, JawTransform, rigid_transform};
pub use pca::PcaReconstruction;
pub use postprocess::{
    PostprocessedRegressionBackend, RegressionGeometry, RegressionPostprocessor,
};

pub use regression::{
    RegressionContract, RegressionFrameInput, RegressionResultLayout, RegressionResultSlices,
};
#[cfg(feature = "tensorrt")]
pub use tensorrt_backend::TensorRtRegressionBackend;

pub const PIPELINE_NAME: &str = "audio2face";
