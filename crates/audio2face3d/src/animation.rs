//! Audio2Face pipeline components.

mod animator;
mod blendshape;
#[cfg(feature = "cuda")]
mod device_noise;
#[cfg(feature = "cuda")]
mod device_postprocess;
mod diffusion;
mod diffusion_executor;
#[cfg(feature = "tensorrt")]
mod diffusion_tensorrt_backend;
mod executor;
#[cfg(feature = "cuda")]
mod gpu_blendshape;
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
pub use blendshape::{
    BlendshapeData, BlendshapeSolverKind, BlendshapeSolverParameters, CpuBlendshapeJobRunner,
    CpuBlendshapeSolver,
};
#[cfg(feature = "cuda")]
pub use device_noise::GpuPhiloxNoise;
#[cfg(feature = "cuda")]
pub use device_postprocess::{
    GpuRegressionModel, GpuRegressionOutputs, GpuRegressionPostprocessFence,
    GpuRegressionPostprocessor, GpuRegressionTrackParams,
};
pub use diffusion::{
    DiffusionContract, DiffusionFrameInput, DiffusionInferenceOutput, DiffusionPostprocessor,
    DiffusionResultLayout, DiffusionResultSlices, DiffusionState, PhiloxNoise,
};
pub use diffusion_executor::{
    DiffusionBackend, DiffusionCallbackMetadata, DiffusionExecutionStatus, DiffusionExecutor,
    DiffusionTrack, MAX_DIFFUSION_TRACKS,
};
#[cfg(feature = "tensorrt")]
pub use diffusion_tensorrt_backend::TensorRtDiffusionBackend;
pub use executor::{
    MAX_REGRESSION_TRACKS, PumpStatus, RegressionBackend, RegressionCallbackMetadata,
    RegressionExecutor, RegressionExecutorState, RegressionTrack,
};
#[cfg(feature = "cuda")]
pub use gpu_blendshape::{GpuBlendshapeSolveFence, GpuBlendshapeSolver};
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
