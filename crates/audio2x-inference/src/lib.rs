//! Safe Rust boundary for the TensorRT inference engine.

pub use audio2x_core::{Binding, BindingSchema};

mod error;
pub use error::InferenceError;

#[cfg(feature = "tensorrt")]
mod ffi;
#[cfg(feature = "tensorrt")]
mod session;

#[cfg(feature = "tensorrt")]
pub use session::{BindingBuffer, DeviceBindings, InferenceFence, TensorRtSession};

mod engine;

pub use engine::{EngineBuildRequest, EngineBuilder, EngineError, ShapeProfile};
