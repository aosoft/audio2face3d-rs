//! User-facing Audio2X facade.

pub use audio2x_core as core;
pub use audio2x_cuda as cuda;
pub use audio2x_inference as inference;

#[cfg(feature = "face")]
pub use audio2face as face;

#[cfg(feature = "emotion")]
pub use audio2emotion as emotion;
