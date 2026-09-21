#[allow(dead_code)]
#[path = "../build_native.rs"]
mod native_build;
use native_build::{config_candidates, parse_config};
use std::path::PathBuf;
fn file() -> PathBuf {
    std::env::temp_dir().join("a2f-build-test/platform.toml")
}
#[test]
fn file_configuration_is_complete_and_relative_to_itself() {
    let file = file();
    let config = parse_config(include_str!("fixtures/platform-config.toml"), &file, true).unwrap();
    assert_eq!(config.cuda_root, file.parent().unwrap().join("build/cuda"));
    assert_eq!(
        config.tensorrt_root,
        Some(file.parent().unwrap().join("shared/trt"))
    );
    assert_eq!(config.cuda_archs, "86,89");
    assert_eq!(
        config.cuda_host_compiler,
        Some(file.parent().unwrap().join("build/compiler"))
    );
}
#[test]
fn incomplete_or_unknown_configuration_does_not_fall_back() {
    for text in [
        "",
        "schema_version=2\ncuda_root='cuda'",
        "cuda-root='cuda'\nunknown=true",
        "cuda-root='cuda'\n[build]\ncuda-archs=[]",
    ] {
        assert!(parse_config(text, &file(), false).is_err(), "{text}");
    }
    assert!(parse_config("cuda-root='cuda'", &file(), true).is_err());
    assert!(parse_config("cuda-root='cuda'", &file(), false).is_ok());
}
#[test]
fn explicit_file_is_exclusive_and_registry_location_does_not_search_cwd() {
    let selected = file();
    assert_eq!(
        config_candidates(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            Some(file().with_file_name("user.toml")),
            Some(selected.clone())
        ),
        vec![selected]
    );
    let registry = std::env::temp_dir().join("cargo-registry/src/audio2face3d-0.1.0");
    assert_eq!(
        config_candidates(&registry, Some(file()), None),
        vec![file()]
    );
}

#[test]
fn version_headers_resolve_aliases_and_reject_cycles() {
    assert_eq!(
        native_build::parse_define("#define A B\n#define B (12)", "A").unwrap(),
        12
    );
    assert!(native_build::parse_define("#define A B\n#define B A", "A").is_err());
    assert!(native_build::parse_define("#define OTHER 1", "A").is_err());
}

#[test]
fn build_rejects_invalid_runtime_sections_even_when_not_used() {
    for tail in [
        "[runtime]\nsearch-policy='latest'",
        "[runtime]\ncuda-library-dirs=[]",
        "[build]\ncuda_archs=['86']",
        "[runtime]\ncuda-root='a'\ncuda-library-dirs=['b']",
    ] {
        assert!(parse_config(&format!("cuda-root='sdk'\n{tail}"), &file(), false).is_err());
    }
}

#[test]
fn absent_platform_file_uses_legacy_build_environment() {
    const CHILD_ROOT: &str = "AUDIO2FACE3D_TEST_BUILD_FALLBACK";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let root = PathBuf::from(root);
        let config = native_build::resolve(true).unwrap();
        assert_eq!(config.cuda_root, root.join("cuda"));
        assert_eq!(config.tensorrt_root, Some(root.join("trt")));
        assert_eq!(config.cuda_archs, "75,86");
        assert_eq!(config.cuda_host_compiler, Some(root.join("compiler")));
        return;
    }
    // Keep environment changes isolated from parallel tests, including Rust 2024 env safety.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/platform-config-work/build-fallback")
        .join(format!(
            "{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    for directory in [
        "cuda/include",
        "trt/include",
        "registry/src/package",
        "user",
    ] {
        std::fs::create_dir_all(root.join(directory)).unwrap();
    }
    for marker in ["cuda/include/cuda.h", "trt/include/NvInfer.h", "compiler"] {
        std::fs::write(root.join(marker), "").unwrap();
    }
    let root = root.canonicalize().unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "absent_platform_file_uses_legacy_build_environment",
            "--nocapture",
        ])
        .current_dir(&root)
        .env(CHILD_ROOT, &root)
        .env("CARGO_MANIFEST_DIR", root.join("registry/src/package"))
        .env_remove("AUDIO2FACE3D_PLATFORM_CONFIG")
        .env_remove("CUDARC_CUDA_VERSION")
        .env("LOCALAPPDATA", root.join("user"))
        .env("XDG_CONFIG_HOME", root.join("user"))
        .env("HOME", root.join("user"))
        .env("CUDA_PATH", root.join("cuda"))
        .env("TENSORRT_ROOT_DIR", root.join("trt"))
        .env("AUDIO2FACE3D_CUDA_ARCHS", "75,86")
        .env("AUDIO2FACE3D_CUDA_HOST_COMPILER", root.join("compiler"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn absent_build_configuration_accepts_multiple_installed_sdks_in_stable_order() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/platform-config-work/build-installed")
        .join(std::process::id().to_string());
    let first = root.join("v12.9");
    let second = root.join("v13.3");
    for sdk in [&first, &second] {
        std::fs::create_dir_all(sdk.join("include")).unwrap();
        std::fs::write(sdk.join("include/cuda.h"), "fixture").unwrap();
    }
    assert_eq!(
        native_build::first_installed_root(
            "cuda-root",
            vec![second, first.clone()],
            "include/cuda.h"
        )
        .unwrap(),
        first
    );
    assert!(
        native_build::first_installed_root(
            "cuda-root",
            vec![root.join("missing")],
            "include/cuda.h"
        )
        .is_err()
    );
}
