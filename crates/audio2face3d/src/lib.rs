//! Unofficial Rust implementation of the NVIDIA Audio2Face-3D SDK.

#[cfg(feature = "animation")]
pub mod animation;
#[cfg(feature = "emotion")]
pub mod audio2emotion;
#[cfg(feature = "animation")]
pub mod audio2face;
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub mod audio2x;
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
mod benchmark;
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub mod common;
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub mod cuda;
#[cfg(feature = "emotion")]
mod emotion;
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
mod model;
pub mod runtime;
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub mod tensorrt;

#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub use benchmark::RawNetworkBenchmark;
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub use benchmark::{BenchmarkPhase, BenchmarkReport, BenchmarkRunner, Percentiles};
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub use common::{Error, Result};
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub use model::{Model, ModelKind, ModelParameters};
#[cfg(any(
    feature = "animation",
    feature = "emotion",
    feature = "cuda",
    feature = "tensorrt",
    feature = "cli"
))]
pub use runtime::RuntimeDiscovery;

pub mod client;
#[cfg(any(feature = "mock", feature = "native", feature = "grpc-server"))]
pub mod inference;
#[cfg(any(feature = "client-grpc", feature = "grpc-server"))]
pub mod protocol;
pub mod types;

mod context;
pub mod logging;
pub use context::{Audio2Face3DContext, Audio2Face3DContextBuilder};

#[cfg(feature = "runtime-cli")]
mod platform_config_file;

#[cfg(feature = "cli-logging")]
pub mod cli_logging;
