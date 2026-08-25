//! Safe Rust boundary for the TensorRT inference engine.

pub use crate::common::{Binding, BindingSchema};

mod error;
pub use error::InferenceError;

#[cfg(feature = "tensorrt")]
mod ffi;
#[cfg(feature = "tensorrt")]
mod session;

#[cfg(feature = "tensorrt")]
pub use session::{
    BindingBuffer, DeviceBindings, EngineEnvironment, InferenceFence, RuntimeTensorShape,
    TensorRtLogMessage, TensorRtSession,
};

mod engine;
mod metadata;
mod trt_info;

pub use engine::{EngineBuildRequest, EngineBuilder, EngineError, ShapeProfile};
pub use metadata::{
    Compatibility, CompatibilityIssue, EngineMetadata, IoTensorMetadata, ProfileMetadata,
};
pub use trt_info::{TrtBuildInfo, TrtBuildInfoError};
