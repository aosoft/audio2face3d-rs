//! CUDA ownership and asynchronous resource primitives.
//!
//! [`DeviceView`] and [`CudaStreamRef`] are portable borrowed contracts: they
//! are available with or without the `cuda` feature, so device-resident result
//! signatures have one type identity in every feature configuration. Owning
//! streams, buffers, allocation, copies, and kernel operations require the
//! `cuda` feature.
//!
//! # `Send` and `Sync`
//!
//! Native CUDA ownership follows the audited D-04 matrix:
//!
//! - `GpuDevice`, `CudaStream`, `CudaEvent`, `CudaModule`, and `CudaFunction`:
//!   `Send + Sync`; each native entry point installs and restores its current
//!   context and destruction is ordered by ownership.
//! - `DeviceBuffer<T>` and [`DeviceView`]: `Send` when `T: Send`, and `Sync`
//!   when `T: Sync`.
//! - [`CudaStreamRef`]: `Send + Sync` while its callback-scoped lifetime is
//!   retained.
//! - `CublasHandle`: at least `Send`; `Sync` only with fixed configuration and
//!   serialized calls on one handle.
//! - `CurandHandle`: `Send + !Sync`.
//! - TensorRT sessions and completed executors: `Send + !Sync`.
//!
//! The native implementations provide the context restoration, concurrency
//! control, and drop-order guarantees behind these traits. `Send` executor futures remain compatible with
//! synchronous result callbacks because borrowed CUDA values are temporary on
//! the polling thread and are not retained across an `.await` boundary.

mod borrowed;

#[cfg(feature = "cuda")]
mod accumulator;
#[cfg(feature = "cuda")]
pub(crate) mod api;
#[cfg(feature = "cuda")]
mod native;

#[cfg(feature = "cuda")]
pub use accumulator::{DeviceAudioAccumulatorExt, DeviceFloatAccumulatorExt};

pub use borrowed::{CudaStreamRef, DeviceView};

#[cfg(feature = "cuda")]
pub(crate) use native::copy_device_view_to_host;

#[cfg(feature = "cuda")]
pub use native::{
    CublasFence, CublasHandle, CublasTranspose, CudaEvent, CudaFunction, CudaModule, CudaStream,
    CurandFence, CurandHandle, DeviceBuffer, GpuDevice, PcaDimensions,
};

use crate::common::{Error, Result};

#[cfg(feature = "cuda")]
fn checked_i32_for_cuda(value: usize, field: &'static str) -> Result<i32> {
    crate::common::checked_i32(value, field)
}

#[cfg(feature = "cuda")]
pub fn regression_postprocess_ptx() -> &'static str {
    include_str!(env!("AUDIO2FACE3D_REGRESSION_POSTPROCESS_PTX"))
}

#[cfg(feature = "cuda")]
pub fn regression_jaw_ptx() -> &'static str {
    include_str!(env!("AUDIO2FACE3D_REGRESSION_JAW_PTX"))
}

#[cfg(feature = "cuda")]
pub fn blendshape_solver_ptx() -> &'static str {
    include_str!(env!("AUDIO2FACE3D_BLENDSHAPE_SOLVER_PTX"))
}

#[cfg(feature = "cuda")]
pub fn emotion_postprocess_ptx() -> &'static str {
    include_str!(env!("AUDIO2FACE3D_EMOTION_POSTPROCESS_PTX"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(i32);

impl DeviceId {
    pub fn new(ordinal: i32) -> Result<Self> {
        if ordinal < 0 {
            return Err(Error::CudaUnavailable(format!(
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
        Err(Error::DeviceMismatch {
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
