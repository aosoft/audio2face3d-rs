//! CUDA ownership and asynchronous resource primitives.

pub mod build_config;

#[cfg(feature = "cuda")]
mod accumulator;
#[cfg(feature = "cuda")]
mod native;

#[cfg(feature = "cuda")]
pub use accumulator::{DeviceAudioAccumulatorExt, DeviceFloatAccumulatorExt};

#[cfg(feature = "cuda")]
pub use native::{
    CublasFence, CublasHandle, CublasTranspose, CudaEvent, CudaFunction, CudaModule, CudaStream,
    CurandFence, CurandHandle, DeviceBuffer, DeviceView, GpuDevice, PcaDimensions,
};

use audio2x_core::{Audio2xError, Result};

#[cfg(feature = "cuda")]
fn audio2x_core_checked_i32(value: usize, field: &'static str) -> Result<i32> {
    audio2x_core::checked_i32(value, field)
}

#[cfg(feature = "cuda")]
pub fn regression_postprocess_ptx() -> &'static str {
    include_str!(env!("AUDIO2X_REGRESSION_POSTPROCESS_PTX"))
}

#[cfg(feature = "cuda")]
pub fn regression_jaw_ptx() -> &'static str {
    include_str!(env!("AUDIO2X_REGRESSION_JAW_PTX"))
}

#[cfg(feature = "cuda")]
pub fn blendshape_solver_ptx() -> &'static str {
    include_str!(env!("AUDIO2X_BLENDSHAPE_SOLVER_PTX"))
}

#[cfg(feature = "cuda")]
pub fn emotion_postprocess_ptx() -> &'static str {
    include_str!(env!("AUDIO2X_EMOTION_POSTPROCESS_PTX"))
}

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
