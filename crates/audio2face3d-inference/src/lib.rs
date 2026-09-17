//! Inference engines and FIFO admission, independent of transport and async runtime.
//! Native operations run on dedicated control workers, separate from native job runners.
pub mod admission;
pub mod animation;
pub mod audio;
mod backend;
mod cancellation;
pub mod config;
#[cfg(feature = "runtime")]
mod emotion;
mod mock;
#[cfg(feature = "runtime")]
mod parameters;
#[cfg(feature = "runtime")]
mod regression;
mod resample;
#[cfg(any(feature = "runtime", test))]
mod worker;

pub use audio2face3d_types as types;
pub use backend::{Backend, EngineFuture, Factory};
pub use cancellation::Cancellation;
pub use config::{BackendKind, Config, MockPattern};

#[cfg(test)]
mod engine_tests;
