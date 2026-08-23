//! CUDA ownership and asynchronous resource primitives.

pub mod build_config;

#[cfg(feature = "cuda")]
mod native;

#[cfg(feature = "cuda")]
pub use native::{
    CublasHandle, CudaEvent, CudaModule, CudaStream, CurandHandle, DeviceBuffer, DeviceView,
    GpuDevice,
};

use audio2x_core::{Audio2xError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(i32);

impl DeviceId {
    pub fn new(ordinal: i32) -> Result<Self> {
        if ordinal < 0 {
            return Err(Audio2xError::CudaUnavailable(format!(
                "negative device ordinal {ordinal}"
            )));
        }
        Ok(Self(ordinal))
    }

    pub const fn ordinal(self) -> i32 {
        self.0
    }
}

pub fn ensure_same_device(expected: DeviceId, actual: DeviceId) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(Audio2xError::DeviceMismatch {
            expected: expected.ordinal(),
            actual: actual.ordinal(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cross_device_resources() {
        let zero = DeviceId::new(0).unwrap();
        let one = DeviceId::new(1).unwrap();
        assert!(ensure_same_device(zero, one).is_err());
    }
}
