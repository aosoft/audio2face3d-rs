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
#[cfg(feature = "cuda")]
pub use gpu_teeth::{
    GpuMultiTrackTeethAnimator, GpuMultiTrackTeethFence, GpuTeethInputBatch, GpuTeethOutputBatch,
};
pub use interactive::{
    GeometryInvalidationLayer, InteractiveDiffusionExecutor, InteractiveGeometryInterrupt,
    InteractiveGeometryMetadata, InteractiveGeometryStatus, InteractiveRegressionExecutor,
};
pub use interactive_blendshape::{
    BlendshapeInvalidationLayer, InteractiveBlendshapeLayer, InteractiveBlendshapeWeights,
};
#[cfg(feature = "cuda")]
pub use interactive_gpu_blendshape::{
    DEFAULT_INTERACTIVE_GPU_CACHE_FRAMES, InteractiveGpuBlendshapeLayer,
    InteractiveGpuBlendshapeOutput,
};
pub use jaw::{
    JawParameters, JawTransform, TeethAnimator, TeethAnimatorParameters, rigid_transform,
};
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
pub use pca::PcaReconstruction;
pub use postprocess::{
    LayeredGeometryPostprocessor, PostprocessedRegressionBackend, RegressionGeometry,
    RegressionPostprocessor,
};

pub use regression::{
    RegressionContract, RegressionFrameInput, RegressionResultLayout, RegressionResultSlices,
};
#[cfg(feature = "tensorrt")]
pub use tensorrt_backend::TensorRtRegressionBackend;

pub const PIPELINE_NAME: &str = "audio2face";
