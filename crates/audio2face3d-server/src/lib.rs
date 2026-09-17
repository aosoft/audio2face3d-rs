#![doc = include_str!("../README.md")]
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

pub mod auth;

mod request;

mod health;
mod lifecycle;
pub use health::HealthAuth;
pub use lifecycle::{CleanupCompletion, CleanupStage, ShutdownReport};
