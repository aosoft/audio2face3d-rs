//! Portable mock server for the NVIDIA ACE controller protocol.
mod admission;
pub mod animation;
pub mod audio;
pub mod backend;
pub mod config;
pub mod proto;
pub mod server;
mod service;
mod session;

mod api;
pub use api::{ConfigError, Server, ServerBuilder};
pub use server::ServerError;

pub use config::Config as ServerConfig;
