//! Inference engines and FIFO admission, independent of transport and async runtime.
//! Native operations run on dedicated control workers, separate from native job runners.
pub mod admission;
pub mod animation;
pub mod audio;
mod backend;
mod cancellation;
pub mod config;
#[cfg(feature = "native")]
mod emotion;
#[cfg(feature = "mock")]
mod mock;
#[cfg(feature = "native")]
mod parameters;
#[cfg(feature = "native")]
mod regression;
#[cfg(any(feature = "mock", feature = "native"))]
mod resample;
#[cfg(any(feature = "native", test))]
mod worker;

pub use crate::types;
pub use backend::{Backend, EngineFuture, Factory};
pub use cancellation::Cancellation;
pub use config::{BackendKind, Config, ConfigBuilder, MockPattern};

#[cfg(test)]
mod engine_tests;

#[cfg(test)]
mod buffer_profile;
