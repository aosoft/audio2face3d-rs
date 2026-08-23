use std::env;
use std::path::{Path, PathBuf};

fn required_directory(name: &str) -> PathBuf {
    let path = env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{name} is required when the tensorrt feature is enabled"));
    assert!(
        path.is_dir(),
        "{name} is not a directory: {}",
        path.display()
    );
    path
}

fn first_directory(root: &Path, candidates: &[&str]) -> PathBuf {
    candidates
        .iter()
        .map(|candidate| root.join(candidate))
        .find(|candidate| candidate.is_dir())
        .unwrap_or_else(|| {
            panic!(
                "none of the library directories exist below {}: {}",
                root.display(),
                candidates.join(", ")
            )
        })
}

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT_DIR");
    if env::var_os("CARGO_FEATURE_TENSORRT").is_none() {
        return;
    }

    let cuda = required_directory("CUDA_PATH");
    let tensorrt = required_directory("TENSORRT_ROOT_DIR");
    assert!(
        cuda.join("include").is_dir(),
        "CUDA include directory is missing"
    );
    assert!(
        tensorrt.join("include/NvInfer.h").is_file(),
        "TensorRT NvInfer.h is missing"
    );
    let library = first_directory(&tensorrt, &["lib", "lib64", "lib/x64", "cuda/lib"]);
    let compiler = cc::Build::new().cpp(true).get_compiler();
    println!(
        "cargo:warning=TensorRT host compiler: {}",
        compiler.path().display()
    );
    println!("cargo:rustc-link-search=native={}", library.display());
    println!("cargo:rustc-link-lib=dylib=nvinfer_10");
}
