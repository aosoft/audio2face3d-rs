//! Unofficial Rust implementation of the NVIDIA Audio2Face-3D SDK.

#[cfg(feature = "animation")]
pub mod animation;
#[cfg(feature = "emotion")]
pub mod audio2emotion;
#[cfg(feature = "animation")]
pub mod audio2face;
pub mod audio2x;
mod benchmark;
pub mod common;
pub mod cuda;
#[cfg(feature = "emotion")]
mod emotion;
mod model;
mod runtime;
pub mod tensorrt;

#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
pub use benchmark::RawNetworkBenchmark;
pub use benchmark::{BenchmarkPhase, BenchmarkReport, BenchmarkRunner, Percentiles};
pub use common::{Error, Result};
pub use model::{Model, ModelKind, ModelParameters};
pub use runtime::RuntimeDiscovery;
