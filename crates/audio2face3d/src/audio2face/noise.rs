//! SDK-facing Diffusion noise-generator factory.

use crate::animation::GpuPhiloxNoise;
use crate::audio2x::Result;
use crate::cuda::CudaStream;

/// Creates one device-side Philox generator per Diffusion track.
pub fn create_noise_generator(
    stream: &CudaStream,
    track_count: usize,
    noise_size: usize,
    seed: u64,
) -> Result<GpuPhiloxNoise> {
    GpuPhiloxNoise::new(stream, track_count, noise_size, seed)
}
