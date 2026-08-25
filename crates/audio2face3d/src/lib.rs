//! Unofficial Rust implementation of the NVIDIA Audio2Face-3D SDK.

#[cfg(feature = "animation")]
pub mod animation;
mod benchmark;
pub mod common;
pub mod cuda;
#[cfg(feature = "emotion")]
pub mod emotion;
mod model;
mod runtime;
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
mod session;
pub mod tensorrt;

pub use benchmark::{BenchmarkPhase, BenchmarkReport, BenchmarkRunner, Percentiles};
pub use common::{Error, Result};
pub use model::{Model, ModelKind, ModelParameters};
pub use runtime::RuntimeDiscovery;
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
pub use session::{
    CallbackMetadata, PipelineOptions, PipelineOutput, PipelineStatus, TensorRtPipeline,
    TrackParameters,
};
