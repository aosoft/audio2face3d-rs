//! Reusable GUI functionality. Disable default features for model-only data and validation.

/// Current application GLB metadata schema.
pub const SCHEMA_VERSION: u32 = 1;

#[cfg(any(feature = "gltf-read", feature = "gltf-write"))]
pub mod gltf;
pub mod model;
pub mod rig;
pub mod validation;

pub use model::{HeadModel, Material, Mesh, Metadata, ModelError, MorphTarget};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
#[cfg(feature = "standalone-app")]
pub mod audio;
#[cfg(feature = "session")]
pub mod core;
#[cfg(feature = "standalone-app")]
pub mod desktop;
#[cfg(feature = "session")]
pub mod inference;
#[cfg(feature = "session")]
pub mod logging;
#[cfg(feature = "session")]
pub mod playback;
#[cfg(feature = "render-wgpu")]
pub mod render;
#[cfg(feature = "ui-egui")]
pub mod ui;
#[cfg(feature = "session")]
pub mod wav;

#[cfg(feature = "standalone-app")]
pub mod startup;
