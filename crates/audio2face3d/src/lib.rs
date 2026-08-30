//! Unofficial Rust implementation of the NVIDIA Audio2Face-3D SDK.

#[cfg(feature = "animation")]
pub mod animation;
mod benchmark;
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
mod blendshape_bundle;
pub mod common;
pub mod cuda;
#[cfg(feature = "emotion")]
pub mod emotion;
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
mod interactive_bundle;
mod model;
mod runtime;
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
mod session;
pub mod tensorrt;

pub use benchmark::{BenchmarkPhase, BenchmarkReport, BenchmarkRunner, Percentiles};
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
pub use blendshape_bundle::{
    BlendshapeExecutorBundle, BlendshapeOutput, BlendshapeSolverComponents,
    BlendshapeSolverComponentsMut,
};
pub use common::{Error, Result};
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
pub use interactive_bundle::{
    InteractiveBlendshapeExecutorBundle, InteractiveGeometryExecutorBundle,
    InteractivePipelineOptions,
};
pub use model::{Model, ModelKind, ModelParameters};
pub use runtime::RuntimeDiscovery;
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
pub use session::{
    CallbackMetadata, GeometryExecutorBundle, GeometryFrame, GeometryPipelineComponents,
    GeometryPipelineComponentsMut, PipelineOptions, PipelineOutput, PipelineStatus,
    PipelineTrackComponents, TensorRtPipeline, TrackParameters,
};
