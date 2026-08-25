use thiserror::Error;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
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
