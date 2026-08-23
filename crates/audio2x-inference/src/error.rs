use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InferenceError {
    #[error("TensorRT support is unavailable")]
    Unavailable,
    #[error("invalid inference binding: {0}")]
    InvalidBinding(String),
    #[error("duplicate inference binding: {0}")]
    DuplicateBinding(String),
    #[error("missing inference binding: {0}")]
    MissingBinding(String),
    #[error("missing runtime shape for dynamic input: {0}")]
    MissingShape(String),
    #[error("binding {name} belongs to device {actual}, expected {expected}")]
    DeviceMismatch {
        name: String,
        expected: i32,
        actual: i32,
    },
    #[error("binding {name} has {actual} bytes, expected {expected}")]
    SizeMismatch {
        name: String,
        actual: usize,
        expected: usize,
    },
    #[error("TensorRT operation {operation} failed: {message}")]
    Native {
        operation: &'static str,
        message: String,
    },
    #[error("CUDA error: {0}")]
    Cuda(#[from] audio2x_core::Audio2xError),
}
