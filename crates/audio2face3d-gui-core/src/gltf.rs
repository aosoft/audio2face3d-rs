//! Restricted GLB profile. No filesystem access or external resource resolution.
#[cfg(feature = "gltf-read")]
mod read;
#[cfg(feature = "gltf-write")]
mod write;
#[cfg(feature = "gltf-read")]
pub use read::from_glb;
#[cfg(feature = "gltf-write")]
pub use write::{encoded_size, to_glb};
