use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
/// Shared failures for the SDK-facing facade and its internal adapters.
///
/// Corresponds to `nva2x` error categories in `audio2x-common/include/audio2x/error.h`.
/// Normal waiting, completion, and interruption belong to execution reports.
/// Legacy variants remain during migration; existing algorithms are not remapped here.
pub enum Error {
    #[error("{0}")]
    NativeRuntime(#[from] crate::runtime::NativeRuntimeError),
    #[error("invalid {field}: {reason}")]
    InvalidArgument { field: &'static str, reason: String },
    #[error("{field} index {index} is outside length {len}")]
    OutOfBounds {
        field: &'static str,
        index: usize,
        len: usize,
    },
    #[error("{field} has size {actual}, expected {expected}")]
    SizeMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("cannot {operation} while {state}")]
    InvalidState {
        operation: &'static str,
        state: &'static str,
    },
    #[error("execution has already started")]
    ExecutionAlreadyStarted,
    #[error("input history is unavailable for track {track}")]
    InputHistoryUnavailable { track: usize },
    #[error("unsupported operation: {operation}")]
    Unsupported { operation: &'static str },
    #[error("feature {feature} is unavailable")]
    FeatureUnavailable { feature: &'static str },
    #[error("model error: {message}")]
    Model { message: String },
    #[error("I/O {operation} failed: {message}")]
    Io {
        operation: &'static str,
        message: String,
    },
    #[error("TensorRT {operation} failed: {message}")]
    TensorRt {
        operation: &'static str,
        message: String,
    },
    #[error("worker failed for track {track}: {message}")]
    Worker { track: usize, message: String },
    #[error("{resource} lock is poisoned")]
    Poisoned { resource: &'static str },
    #[error("{field} value {value} does not fit in {target}")]
    IntegerOverflow {
        field: &'static str,
        value: usize,
        target: &'static str,
    },
    #[error("invalid tensor schema: {0}")]
    InvalidSchema(String),
    #[error("duplicate tensor binding: {0}")]
    DuplicateBinding(String),
    #[error("CUDA support is unavailable: {0}")]
    CudaUnavailable(String),
    #[error("CUDA {operation} failed with code {code}")]
    Cuda { operation: &'static str, code: u32 },
    #[error("resource belongs to device {actual}, expected device {expected}")]
    DeviceMismatch { expected: i32, actual: i32 },
}

pub fn checked_i32(value: usize, field: &'static str) -> Result<i32> {
    i32::try_from(value).map_err(|_| Error::IntegerOverflow {
        field,
        value,
        target: "i32",
    })
}

pub fn checked_u32(value: usize, field: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::IntegerOverflow {
        field,
        value,
        target: "u32",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_conversion_reports_field() {
        let error = checked_i32(usize::MAX, "element_count").unwrap_err();
        assert!(matches!(
            error,
            Error::IntegerOverflow {
                field: "element_count",
                ..
            }
        ));
    }
}
