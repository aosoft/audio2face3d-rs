#[path = "src/build_config.rs"]
mod build_config;

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=AUDIO2X_CUDA_ARCHS");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cuda = env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .expect("CUDA_PATH is required when the cuda feature is enabled");
    let nvcc = cuda
        .join("bin")
        .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
    assert!(
        nvcc.is_file(),
        "CUDA compiler not found: {}",
        nvcc.display()
    );

    let architectures = env::var("AUDIO2X_CUDA_ARCHS").unwrap_or_else(|_| "86".into());
    let flags = build_config::gencode_flags(&architectures)
        .unwrap_or_else(|error| panic!("invalid AUDIO2X_CUDA_ARCHS: {error}"));
    for flag in flags {
        println!("cargo:warning=CUDA architecture flag: {flag}");
    }
}
