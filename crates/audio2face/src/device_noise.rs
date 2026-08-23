use audio2x_core::{Audio2xError, Result};
use audio2x_cuda::{CudaStream, CurandFence, CurandHandle, DeviceBuffer};

/// One device-side Philox generator per Diffusion track.
pub struct GpuPhiloxNoise {
    generators: Vec<CurandHandle>,
    size: usize,
}

impl GpuPhiloxNoise {
    pub fn new(stream: &CudaStream, tracks: usize, size: usize, seed: u64) -> Result<Self> {
        if tracks == 0 || size == 0 || size % 2 != 0 {
            return Err(invalid(
                "GPU Philox track count and even noise size must be non-zero",
            ));
        }
        let mut generators = Vec::with_capacity(tracks);
        for track in 0..tracks {
            let mut generator = CurandHandle::new(stream)?;
            generator.set_seed(seed.wrapping_add(track as u64))?;
            generators.push(generator);
        }
        Ok(Self { generators, size })
    }

    /// Enqueues one track's standard-normal noise generation.
    pub fn generate<'a>(
        &'a mut self,
        track: usize,
        output: &'a mut DeviceBuffer<f32>,
        stream: &'a CudaStream,
    ) -> Result<CurandFence<'a>> {
        if output.len() != self.size {
            return Err(invalid("GPU Philox output size mismatch"));
        }
        self.generators
            .get_mut(track)
            .ok_or_else(|| invalid("GPU Philox track is out of range"))?
            .generate_normal(output, stream)
    }

    pub fn reset(&mut self, track: usize, generate_index: usize) -> Result<()> {
        let offset = generate_index
            .checked_mul(self.size)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(|| invalid("GPU Philox reset offset overflow"))?;
        self.generators
            .get_mut(track)
            .ok_or_else(|| invalid("GPU Philox track is out of range"))?
            .set_offset(offset)
    }
}

fn invalid(message: impl Into<String>) -> Audio2xError {
    Audio2xError::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio2x_cuda::GpuDevice;

    #[test]
    fn gpu_tracks_are_isolated_and_reset_replays() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let mut noise = GpuPhiloxNoise::new(&stream, 2, 1024, 17).unwrap();
        let mut first_buffer = device.allocate(1024).unwrap();
        let mut second_buffer = device.allocate(1024).unwrap();
        noise
            .generate(0, &mut first_buffer, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        noise
            .generate(1, &mut second_buffer, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let mut first = vec![0.0; 1024];
        let mut second = vec![0.0; 1024];
        first_buffer.copy_to(&mut first, &stream).unwrap();
        second_buffer.copy_to(&mut second, &stream).unwrap();
        assert_ne!(first, second);
        noise.reset(0, 0).unwrap();
        noise
            .generate(0, &mut first_buffer, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let mut replay = vec![0.0; 1024];
        first_buffer.copy_to(&mut replay, &stream).unwrap();
        assert_eq!(replay, first);
    }
}
