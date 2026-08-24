//! User-facing Audio2X facade.

mod benchmark;
mod model;
mod runtime;
#[cfg(all(feature = "face", feature = "emotion", feature = "tensorrt"))]
mod session;

pub use audio2x_core as core;
pub use audio2x_cuda as cuda;
pub use audio2x_inference as inference;
pub use benchmark::{BenchmarkPhase, BenchmarkReport, BenchmarkRunner, Percentiles};
pub use model::{Audio2xModel, ModelKind, ModelParameters};
pub use runtime::RuntimeDiscovery;
#[cfg(all(feature = "face", feature = "emotion", feature = "tensorrt"))]
pub use session::{
    CallbackMetadata, PipelineOptions, PipelineOutput, PipelineStatus, TensorRtPipeline,
    TrackParameters,
};

#[cfg(feature = "face")]
pub use audio2face as face;

#[cfg(feature = "emotion")]
pub use audio2emotion as emotion;
