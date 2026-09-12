#![cfg_attr(not(feature = "tensorrt"), allow(unused_imports))]

//! Audio2Face pipeline components.

mod animator;
mod blendshape;
#[cfg(feature = "tensorrt")]
pub(crate) use blendshape::rhs::GpuRhs;
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
#[cfg(feature = "cuda")]
mod gpu_teeth;
mod interactive;
mod interactive_blendshape;
#[cfg(feature = "cuda")]
mod interactive_gpu_blendshape;
mod jaw;
mod model_buffers;
mod model_data;
mod pca;
mod postprocess;
mod regression;
#[cfg(feature = "tensorrt")]
mod tensorrt_backend;

pub use animator::EyesRotation;
pub(crate) use animator::{
    EyesAnimator, EyesAnimatorParams, SkinAnimator, SkinAnimatorParams, TongueAnimator,
    TongueAnimatorParams,
};
pub use blendshape::{
    BlendshapeData, BlendshapeSolverKind, BlendshapeSolverParameters, CpuBlendshapeSolver,
};
#[cfg(feature = "cuda")]
pub use device_noise::GpuPhiloxNoise;
#[cfg(feature = "cuda")]
pub(crate) use device_postprocess::GpuRegressionPcaPostprocessor;
#[cfg(feature = "cuda")]
pub(crate) use device_postprocess::{
    GpuRegressionModel, GpuRegressionOutputs, GpuRegressionPostprocessor, GpuRegressionTrackParams,
};
pub(crate) use diffusion::DiffusionPostprocessor;
pub use diffusion::{
    DiffusionContract, DiffusionFrameInput, DiffusionInferenceOutput, DiffusionResultLayout,
    DiffusionResultSlices, DiffusionState, PhiloxNoise,
};
pub(crate) use diffusion_executor::{DiffusionBackend, DiffusionScheduler, DiffusionTrack};
pub use diffusion_executor::{
    DiffusionCallbackMetadata, DiffusionExecutionStatus, MAX_DIFFUSION_TRACKS,
};
#[cfg(feature = "tensorrt")]
pub(crate) use diffusion_tensorrt_backend::TensorRtDiffusionBackend;
pub use executor::{MAX_REGRESSION_TRACKS, PumpStatus, RegressionCallbackMetadata};
pub(crate) use executor::{RegressionBackend, RegressionScheduler, RegressionTrack};
#[cfg(feature = "cuda")]
pub use gpu_blendshape::{GpuBlendshapeSolveFence, GpuBlendshapeSolver};
#[cfg(feature = "cuda")]
pub use gpu_teeth::{
    GpuMultiTrackTeethAnimator, GpuMultiTrackTeethFence, GpuTeethInputBatch, GpuTeethOutputBatch,
};
pub(crate) use interactive::{
    DiffusionGeometryInteractiveExecution, RegressionGeometryInteractiveExecution,
};
pub use interactive::{
    GeometryInvalidationLayer, InteractiveGeometryInterrupt, InteractiveGeometryMetadata,
    InteractiveGeometryStatus,
};
pub use interactive_blendshape::{
    BlendshapeInvalidationLayer, InteractiveBlendshapeLayer, InteractiveBlendshapeWeights,
};
#[cfg(feature = "cuda")]
pub use interactive_gpu_blendshape::{
    DEFAULT_INTERACTIVE_GPU_CACHE_FRAMES, InteractiveGpuBlendshapeLayer,
    InteractiveGpuBlendshapeOutput,
};
pub(crate) use jaw::JawTransform;
pub use jaw::{JawParameters, rigid_transform};
pub use model_buffers::{
    DiffusionBufferContract, GeometryResultLayout, ModelBindingContract, RegressionBufferContract,
    RuntimeBinding, TensorBatchInfo,
};
#[cfg(feature = "cuda")]
pub use model_buffers::{
    DiffusionInferenceInputBuffers, DiffusionInferenceOutputBuffers,
    DiffusionInferenceStateBuffers, DiffusionResultBuffers, GeometryResultBuffers,
    RegressionInferenceInputBuffers, RegressionInferenceOutputBuffers, RegressionResultBuffers,
};
pub use model_data::GeometryModelData;
pub(crate) use pca::PcaReconstruction;
pub use postprocess::RegressionGeometry;
pub(crate) use postprocess::{LayeredGeometryPostprocessor, RegressionPostprocessor};

pub use regression::{
    RegressionContract, RegressionFrameInput, RegressionResultLayout, RegressionResultSlices,
};
#[cfg(feature = "tensorrt")]
pub(crate) use tensorrt_backend::TensorRtRegressionBackend;

pub const PIPELINE_NAME: &str = "audio2face";
