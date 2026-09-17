//! Owned, bounded streaming sessions shared by direct and server inference.
//!
//! Select a constructor through the `direct` / `server` features. Native direct
//! inference additionally requires `runtime` (CUDA/TensorRT, not an async runtime).
//! Drive input and output concurrently; successful input submission is local
//! acceptance, and only [types::OutputEvent::Completed] confirms a complete result.
//! Await [Control::closed] for backend cleanup and [Client::shutdown] before
//! disposing of a caller-owned server Tokio runtime.
//!
//! Receive Futures exclusively borrow their output until completed or dropped:
//! ```compile_fail
//! fn invalid(output: &mut audio2face3d_client::Output) {
//!     let first = output.recv();
//!     let second = output.recv();
//!     drop((first, second));
//! }
//! ```
//!
//! See the crate README and `examples/stream.rs` for concurrent streaming using
//! a standard-library executor. No async runtime is required by direct mode.
mod client;
mod core;
#[cfg_attr(not(test), allow(dead_code))]
mod driver;
mod memory;
mod notify;
mod session;

pub use audio2face3d_types as types;
pub use client::{Client, Limits, Shutdown};
pub use session::{
    Closed, Control, Finish, Input, Output, Receive, SendChunk, Session, TrySendError,
};
#[cfg(test)]
mod tests;

#[cfg(feature = "direct")]
mod direct;
#[cfg(feature = "direct")]
mod executor;
#[cfg(feature = "direct")]
pub use direct::DirectConfig;
#[cfg(feature = "server")]
mod server;
#[cfg(feature = "server")]
pub use server::ServerConfig;

#[cfg(feature = "direct")]
pub use audio2face3d_inference::{BackendKind, Config as InferenceConfig, MockPattern};
