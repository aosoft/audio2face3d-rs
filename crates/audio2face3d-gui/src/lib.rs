//! Host-driven Audio2Face-3D inspection. Desktop integration is optional.

/// Application version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(feature = "desktop")]
pub mod audio;
pub mod core;
pub mod inference;
pub mod logging;
pub mod playback;
pub mod wav;

#[cfg(feature = "desktop")]
pub mod desktop;
#[cfg(feature = "desktop")]
pub mod startup;

#[cfg(feature = "render-wgpu")]
pub mod render;
#[cfg(feature = "ui-egui")]
pub mod ui;
