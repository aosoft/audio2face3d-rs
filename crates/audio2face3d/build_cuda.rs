#[path = "src/cuda/build_config.rs"]
mod build_config;

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

pub(crate) fn build() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=AUDIO2FACE3D_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=AUDIO2FACE3D_CUDA_HOST_COMPILER");
    println!("cargo:rerun-if-changed=cuda/regression_postprocess.cu");
    println!("cargo:rerun-if-changed=cuda/regression_jaw.cu");
    println!("cargo:rerun-if-changed=cuda/blendshape_solver.cu");
    println!("cargo:rerun-if-changed=cuda/emotion_postprocess.cu");
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

    let architectures = env::var("AUDIO2FACE3D_CUDA_ARCHS").unwrap_or_else(|_| "86".into());
    let flags = build_config::gencode_flags(&architectures)
        .unwrap_or_else(|error| panic!("invalid AUDIO2FACE3D_CUDA_ARCHS: {error}"));
    for flag in &flags {
        println!("cargo:warning=CUDA architecture flag: {flag}");
    }

    // PTX is virtual-architecture specific. Compile for the first requested
    // architecture so the driver can JIT it for that architecture and newer
    // devices. The remaining flags continue to describe the supported native
    // architecture matrix to Cargo's diagnostics.
    let first = flags
        .first()
        .expect("gencode_flags always returns at least one architecture");
    let compute = first
        .split("arch=compute_")
        .nth(1)
        .and_then(|value| value.split(',').next())
        .expect("validated gencode flag has a compute architecture");
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by Cargo"))
        .join("regression_postprocess.ptx");
    let host_compiler = cuda_host_compiler();
    let host_compiler_dir = host_compiler
        .parent()
        .expect("host compiler has a parent directory");
    let mut command = Command::new(&nvcc);
    let status = command
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--allow-unsupported-compiler")
        .arg(format!("--compiler-bindir={}", host_compiler_dir.display()))
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
    let status = Command::new(&nvcc)
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--allow-unsupported-compiler")
        .arg(format!("--compiler-bindir={}", host_compiler_dir.display()))
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
    let status = Command::new(&nvcc)
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--allow-unsupported-compiler")
        .arg(format!("--compiler-bindir={}", host_compiler_dir.display()))
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
    let status = Command::new(&nvcc)
        .arg("--ptx")
        .arg("--std=c++17")
        .arg("--allow-unsupported-compiler")
        .arg(format!("--compiler-bindir={}", host_compiler_dir.display()))
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

fn cuda_host_compiler() -> PathBuf {
    if let Some(path) = env::var_os("AUDIO2FACE3D_CUDA_HOST_COMPILER") {
        return PathBuf::from(path);
    }
    if cfg!(windows)
        && let Some(path) = visual_studio_2022_compiler()
    {
        return path;
    }
    cc::Build::new().cpp(true).get_compiler().path().to_owned()
}

fn visual_studio_2022_compiler() -> Option<PathBuf> {
    let program_files = env::var_os("ProgramFiles(x86)")?;
    let vswhere = PathBuf::from(program_files)
        .join("Microsoft Visual Studio")
        .join("Installer")
        .join("vswhere.exe");
    let output = Command::new(vswhere)
        .args([
            "-latest",
            "-products",
            "*",
            "-version",
            "[17.0,18.0)",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let installation = Path::new(std::str::from_utf8(&output.stdout).ok()?.trim());
    let tools = installation.join("VC").join("Tools").join("MSVC");
    let mut versions = fs::read_dir(tools)
        .ok()?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .collect::<Vec<_>>();
    versions.sort();
    versions
        .pop()
        .map(|version| {
            version
                .join("bin")
                .join("Hostx64")
                .join("x64")
                .join("cl.exe")
        })
        .filter(|compiler| compiler.is_file())
}
