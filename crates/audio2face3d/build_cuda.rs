#[path = "build_cuda_arch.rs"]
mod build_cuda_arch;

use std::env;
use std::path::PathBuf;
use std::process::Command;

pub(crate) fn build(config: &crate::build_native::BuildConfig) {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=AUDIO2FACE3D_CUDA_HOST_COMPILER");
    println!("cargo:rerun-if-changed=cuda/regression_postprocess.cu");
    println!("cargo:rerun-if-changed=cuda/regression_jaw.cu");
    println!("cargo:rerun-if-changed=cuda/blendshape_solver.cu");
    println!("cargo:rerun-if-changed=cuda/emotion_postprocess.cu");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let cuda = &config.cuda_root;
    let nvcc = cuda
        .join("bin")
        .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
    assert!(
        nvcc.is_file(),
        "CUDA compiler not found: {}",
        nvcc.display()
    );

    println!("cargo:rerun-if-changed={}", nvcc.display());
    let supported = cuda_command(config, &nvcc)
        .arg("--list-gpu-arch")
        .output()
        .unwrap_or_else(|error| panic!("failed to query {}: {error}", nvcc.display()));
    assert!(
        supported.status.success(),
        "nvcc target query failed: {}",
        String::from_utf8_lossy(&supported.stderr)
    );
    let compute = build_cuda_arch::minimum_target(&String::from_utf8_lossy(&supported.stdout))
        .expect("select the minimum supported CUDA kernel PTX target");
    println!("cargo:rustc-env=AUDIO2FACE3D_KERNEL_COMPUTE_CAPABILITY={compute}");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join("regression_postprocess.ptx");
    let mut command = cuda_command(config, &nvcc);
    let status = command
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--allow-unsupported-compiler")
        .arg(format!("--gpu-architecture=compute_{compute}"))
        .arg("cuda/regression_postprocess.cu")
        .arg("--output-file")
        .arg(&output)
        .status()
        .unwrap_or_else(|error| panic!("failed to launch {}: {error}", nvcc.display()));
    assert!(status.success(), "CUDA postprocess PTX compilation failed");
    println!(
        "cargo:rustc-env=AUDIO2FACE3D_REGRESSION_POSTPROCESS_PTX={}",
        output.display()
    );
    let jaw_output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join("regression_jaw.ptx");
    let status = cuda_command(config, &nvcc)
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--allow-unsupported-compiler")
        .arg(format!("--gpu-architecture=compute_{compute}"))
        .arg("cuda/regression_jaw.cu")
        .arg("--output-file")
        .arg(&jaw_output)
        .status()
        .unwrap_or_else(|error| panic!("failed to launch {}: {error}", nvcc.display()));
    assert!(status.success(), "CUDA jaw PTX compilation failed");
    println!(
        "cargo:rustc-env=AUDIO2FACE3D_REGRESSION_JAW_PTX={}",
        jaw_output.display()
    );
    let blendshape_output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join("blendshape_solver.ptx");
    let status = cuda_command(config, &nvcc)
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--allow-unsupported-compiler")
        .arg(format!("--gpu-architecture=compute_{compute}"))
        .arg("cuda/blendshape_solver.cu")
        .arg("--output-file")
        .arg(&blendshape_output)
        .status()
        .unwrap_or_else(|error| panic!("failed to launch {}: {error}", nvcc.display()));
    assert!(status.success(), "CUDA blendshape PTX compilation failed");
    println!(
        "cargo:rustc-env=AUDIO2FACE3D_BLENDSHAPE_SOLVER_PTX={}",
        blendshape_output.display()
    );
    let emotion_output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join("emotion_postprocess.ptx");
    let status = cuda_command(config, &nvcc)
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--allow-unsupported-compiler")
        .arg(format!("--gpu-architecture=compute_{compute}"))
        .arg("cuda/emotion_postprocess.cu")
        .arg("--output-file")
        .arg(&emotion_output)
        .status()
        .unwrap_or_else(|error| panic!("failed to launch {}: {error}", nvcc.display()));
    assert!(status.success(), "CUDA emotion PTX compilation failed");
    println!(
        "cargo:rustc-env=AUDIO2FACE3D_EMOTION_POSTPROCESS_PTX={}",
        emotion_output.display()
    );
}

fn cuda_command(config: &crate::build_native::BuildConfig, nvcc: &std::path::Path) -> Command {
    let mut command = Command::new(nvcc);
    command.envs(config.compiler_env.iter().cloned());
    if let Some(compiler) = &config.cuda_host_compiler {
        command.arg("--compiler-bindir").arg(compiler);
    }
    command
}
