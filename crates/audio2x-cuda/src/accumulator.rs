//! CUDA input adapters for the core accumulators.
//!
//! Transfers are ordered on the supplied stream and synchronized before the
//! method returns. The source allocation must therefore remain alive only for
//! the duration of the call.

use crate::{CudaStream, DeviceBuffer};
use audio2x_core::{AudioAccumulator, FloatAccumulator, Result};

pub trait DeviceFloatAccumulatorExt {
    fn accumulate_device(&self, source: &DeviceBuffer<f32>, stream: &CudaStream) -> Result<()>;
}

impl DeviceFloatAccumulatorExt for FloatAccumulator {
    fn accumulate_device(&self, source: &DeviceBuffer<f32>, stream: &CudaStream) -> Result<()> {
        let mut host = vec![0.0; source.len()];
        source.copy_to(&mut host, stream)?;
        self.accumulate(&host)
    }
}

pub trait DeviceAudioAccumulatorExt {
    fn accumulate_device(&self, source: &DeviceBuffer<f32>, stream: &CudaStream) -> Result<()>;
}

impl DeviceAudioAccumulatorExt for AudioAccumulator {
    fn accumulate_device(&self, source: &DeviceBuffer<f32>, stream: &CudaStream) -> Result<()> {
        let mut host = vec![0.0; source.len()];
        source.copy_to(&mut host, stream)?;
        self.accumulate(&host)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuDevice;

    #[test]
    fn accumulates_from_device_memory() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let mut source = device.allocate::<f32>(3).unwrap();
        source.copy_from(&[1.0, 2.0, 3.0], &stream).unwrap();
        let accumulator = FloatAccumulator::new(2, 0).unwrap();
        accumulator.accumulate_device(&source, &stream).unwrap();
        assert_eq!(accumulator.read(0, 3).unwrap(), [1.0, 2.0, 3.0]);
    }
}
