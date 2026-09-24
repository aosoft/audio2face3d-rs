//! Portable display-model data. No inference, GPU, audio device or window ownership.

/// Current application GLB metadata schema.
pub const SCHEMA_VERSION: u32 = 1;

#[cfg(any(feature = "gltf-read", feature = "gltf-write"))]
pub mod gltf;
pub mod model;
pub mod rig;
pub mod validation;

pub use model::{HeadModel, Material, Mesh, Metadata, ModelError, MorphTarget};
