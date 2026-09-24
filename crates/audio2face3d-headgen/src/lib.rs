//! Deterministic, CPU-only conversion of matching neutral and expression OBJ files.
//! Input files and generated assets remain under the caller's control.
pub const GENERATOR_VERSION: &str = env!("CARGO_PKG_VERSION");
pub mod config;
pub mod convert;
pub mod error;
mod normals;
pub mod obj;
pub mod report;
pub mod topology;
pub mod transform;
pub use config::Config;
pub use convert::{Conversion, convert, inspect};
