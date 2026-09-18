//! Owned, bounded streaming sessions shared by direct and server inference.
//!
//! Select local inference with `mock` or `native`, and remote inference with
//! `client-grpc`. Native inference uses CUDA/TensorRT (CUDA/TensorRT, not an async runtime).
//! Drive input and output concurrently; successful input submission is local
//! acceptance, and only [types::OutputEvent::Completed] confirms a complete result.
//! Await [Control::closed] for backend cleanup and [Client::shutdown] before
//! disposing of a caller-owned server Tokio runtime.
//!
//! Receive Futures exclusively borrow their output until completed or dropped:
//! ```compile_fail
//! fn invalid(output: &mut audio2face3d::client::Output) {
//!     let first = output.recv();
//!     let second = output.recv();
//!     drop((first, second));
//! }
//! ```
//!
//! See the crate README and `examples/stream.rs` for concurrent streaming using
//! a standard-library executor. No async runtime is required by direct mode.
mod core;
#[cfg_attr(not(test), allow(dead_code))]
mod driver;
mod handle;
mod memory;
mod notify;
mod session;

pub use crate::types;
pub use handle::{Client, Limits, LimitsBuilder, Shutdown};
pub use session::{
    Closed, Control, Finish, Input, Output, Receive, SendChunk, Session, TrySendError,
};
#[cfg(test)]
mod tests;

#[cfg(any(feature = "mock", feature = "native"))]
mod direct;
#[cfg(any(feature = "mock", feature = "native"))]
mod executor;
#[cfg(any(feature = "mock", feature = "native"))]
pub use direct::{DirectConfig, DirectConfigBuilder};
#[cfg(feature = "client-grpc")]
mod server;
#[cfg(feature = "client-grpc")]
pub use server::{ServerConfig, ServerConfigBuilder};

#[cfg(any(feature = "mock", feature = "native"))]
pub use crate::inference::{BackendKind, Config as InferenceConfig, MockPattern};
