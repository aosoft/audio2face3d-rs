//! PTX target selection for the project's own CUDA kernels.

// All four kernels compile at compute_50, the lowest generic target available
// in the CUDA 12.9 toolchain used for verification. They use scalar FP32/FP64 math,
// integer operations and population count, with no newer architecture-specific
// features. Keep this baseline with the kernel code, not the build machine's GPU.
// A Toolkit that drops older targets raises the effective minimum automatically.
const KERNEL_PTX_BASELINE: u32 = 50;

pub fn minimum_target(supported: &str) -> Result<u32, String> {
    supported
        .lines()
        .filter_map(|line| line.trim().strip_prefix("compute_"))
        // Architecture/family-specific variants (for example 90a and 100f)
        // have narrower compatibility than generic PTX targets.
        .filter(|number| !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()))
        .filter_map(|number| number.parse::<u32>().ok())
        .filter(|number| *number >= KERNEL_PTX_BASELINE)
        .min()
        .ok_or_else(|| {
            format!("nvcc reported no generic PTX target at or above compute_{KERNEL_PTX_BASELINE}")
        })
}
