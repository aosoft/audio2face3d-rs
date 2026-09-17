//! Common session control. Transport/direct constructors are added by their adapters.
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
