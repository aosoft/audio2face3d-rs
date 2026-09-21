use std::env;
pub(crate) fn build(config: &crate::build_native::BuildConfig) {
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=TENSORRT_ROOT_DIR");
    if env::var_os("CARGO_FEATURE_TENSORRT").is_none() {
        return;
    }

    let cuda = &config.cuda_root;
    let tensorrt = config
        .tensorrt_root
        .as_ref()
        .expect("TensorRT configuration");
    assert!(
        cuda.join("include").is_dir(),
        "CUDA include directory is missing"
    );
    assert!(
        tensorrt.join("include/NvInfer.h").is_file(),
        "TensorRT NvInfer.h is missing"
    );
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .file("cpp/tensorrt_shim.cpp")
        .include(cuda.join("include"))
        .include(tensorrt.join("include"))
        .warnings(true);
    if build.get_compiler().is_like_msvc() {
        build.flag("/std:c++17").flag("/EHsc");
    } else {
        build.flag("-std=c++17");
    }
    println!("cargo:rerun-if-changed=cpp/tensorrt_shim.h");
    println!("cargo:rerun-if-changed=cpp/tensorrt_shim.cpp");
    build.compile("audio2face3d_tensorrt_shim");
}
