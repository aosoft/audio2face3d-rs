use audio2face3d::runtime::{NativeRuntimeConfig, NativeSearchPolicy, tools::NativeTool};
use std::{path::PathBuf, process::Command};

#[cfg(feature = "model-management")]
#[test]
fn explicit_engine_tools_use_the_owner_for_build_versions_and_existing_artifacts() {
    use audio2face3d::{
        Audio2Face3DContext,
        logging::integration::LogScope,
        model_management::{EnginePrecision, ModelEngineBuildRequest},
        tensorrt::{EngineBuildRequest, EngineBuilder},
    };
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .join("../../temp/native-runtime-work/fixtures")
        .join(format!(
            "explicit-build-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    for owner in ["owner", "foreign"] {
        for sdk in ["cuda", "trt"] {
            std::fs::create_dir_all(root.join(owner).join(sdk).join("bin")).unwrap();
        }
    }
    let tool = root.join("owner/trt/bin").join(if cfg!(windows) {
        "trtexec.exe"
    } else {
        "trtexec"
    });
    let compile = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2024", "--crate-name=explicit_build_tool"])
        .arg(manifest.join("tests/fixtures/native_loader/build_tool.rs"))
        .arg("-o")
        .arg(&tool)
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    std::fs::copy(
        &tool,
        root.join("owner/cuda/bin")
            .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" }),
    )
    .unwrap();
    let make_context = |name: &str| {
        Audio2Face3DContext::builder()
            .native_runtime(
                NativeRuntimeConfig::builder()
                    .cuda_root(root.join(name).join("cuda"))
                    .tensorrt_root(root.join(name).join("trt"))
                    .search_policy(NativeSearchPolicy::ExplicitOnly)
                    .build()
                    .unwrap(),
            )
            .build()
    };
    let context = make_context("owner");
    let other = make_context("foreign");
    let model = root.join("model");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("network.onnx"), b"onnx").unwrap();
    std::fs::write(
        model.join("model.json"),
        br#"{"networkPath":"network.trt"}"#,
    )
    .unwrap();
    std::fs::write(
        model.join("trt_info.json"),
        br#"{"trt_build_param":{},"defaults":{}}"#,
    )
    .unwrap();
    let request = ModelEngineBuildRequest {
        model_directory: model.clone(),
        precision: EnginePrecision::Fp32,
        device_id: 0,
        max_batch_size: None,
        replace: false,
        trtexec: tool.clone(),
    };
    let receipt = LogScope::new(other.clone())
        .in_scope(|| request.execute_with_context(&context))
        .unwrap();
    let expected = root
        .join("owner/cuda/bin")
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let unexpected = root
        .join("foreign")
        .canonicalize()
        .unwrap()
        .to_string_lossy()
        .to_string();
    let engine = std::fs::read_to_string(&receipt.engine).unwrap();
    assert!(engine.contains(&expected));
    assert!(!engine.contains(&unexpected));
    let provenance: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&receipt.provenance).unwrap()).unwrap();
    assert!(
        provenance["build"]["cuda_toolkit"]
            .as_str()
            .unwrap()
            .contains(&expected)
    );
    assert!(
        !provenance["build"]["cuda_toolkit"]
            .as_str()
            .unwrap()
            .contains(&unexpected)
    );
    // The compatibility entry captures its caller once and verifies the same identity.
    LogScope::new(context.clone())
        .in_scope(|| request.execute())
        .unwrap();
    let direct = EngineBuildRequest {
        onnx: model.join("network.onnx"),
        engine: root.join("direct.trt"),
        device_id: 0,
        profiles: vec![],
        extra_args: vec![],
    };
    LogScope::new(other.clone())
        .in_scope(|| {
            EngineBuilder {
                executable: tool,
                replace_existing: false,
            }
            .build_with_context(&direct, context.clone())
        })
        .unwrap();
    assert!(
        std::fs::read_to_string(&direct.engine)
            .unwrap()
            .contains(&expected)
    );
    assert!(context.native_runtime_info().is_none());
    assert!(other.native_runtime_info().is_none());
}
#[test]
fn tool_selection_changes_only_the_child_environment() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .join("../../temp/native-runtime-work/fixtures")
        .join(format!(
            "tools-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    let cuda = root.join("cuda/bin");
    let trt = root.join("trt/bin");
    std::fs::create_dir_all(&cuda).unwrap();
    std::fs::create_dir_all(&trt).unwrap();
    let binary = trt.join(if cfg!(windows) {
        "trtexec.exe"
    } else {
        "trtexec"
    });
    let output = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
        .args(["--edition=2024", "--crate-name=a2f_tool_fixture"])
        .arg(manifest.join("tests/fixtures/native_loader/tool.rs"))
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let variable = if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let before = std::env::var_os(variable);
    let config = NativeRuntimeConfig::builder()
        .cuda_root(root.join("cuda"))
        .tensorrt_root(root.join("trt"))
        .search_policy(NativeSearchPolicy::ExplicitOnly)
        .build()
        .unwrap();
    let mut command = config.tool_command(NativeTool::Trtexec, None).unwrap();
    assert!(std::path::Path::new(command.get_program()).is_absolute());
    let result = command.output().unwrap();
    assert!(result.status.success());
    let child = String::from_utf8(result.stdout).unwrap();
    assert!(child.contains(&cuda.canonicalize().unwrap().to_string_lossy().to_string()));
    assert!(child.contains(&trt.canonicalize().unwrap().to_string_lossy().to_string()));
    assert_eq!(std::env::var_os(variable), before);
    let missing = NativeRuntimeConfig::builder()
        .cuda_root(root.join("cuda"))
        .tensorrt_root(root.join("missing"))
        .search_policy(NativeSearchPolicy::ExplicitOnly)
        .build()
        .unwrap();
    assert!(missing.tool_command(NativeTool::Trtexec, None).is_err());
}

#[test]
fn unconfigured_tools_follow_path_order_with_multiple_sdk_candidates() {
    const CHILD_ROOT: &str = "AUDIO2FACE3D_TEST_TOOL_PATH";
    if let Some(root) = std::env::var_os(CHILD_ROOT) {
        let root = PathBuf::from(root);
        let default = NativeRuntimeConfig::default();
        let command = default.tool_command(NativeTool::Nvcc, None).unwrap();
        let tool = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
        assert_eq!(
            PathBuf::from(command.get_program()),
            root.join("z-first").join(tool).canonicalize().unwrap()
        );
        let explicit = NativeRuntimeConfig::builder()
            .cuda_library_dirs([root.join("z-first"), root.join("a-second")])
            .build()
            .unwrap();
        assert!(explicit.tool_command(NativeTool::Nvcc, None).is_err());
        return;
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/platform-config-work/tools-path")
        .join(std::process::id().to_string());
    let dirs = [root.join("z-first"), root.join("a-second")];
    for dir in &dirs {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(
            dir.join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" }),
            "fixture",
        )
        .unwrap();
        std::fs::write(
            dir.join(if cfg!(windows) {
                "cudart64_12.dll"
            } else {
                "libcudart.so.12"
            }),
            "fixture",
        )
        .unwrap();
    }
    let root = root.canonicalize().unwrap();
    let path = std::env::join_paths(dirs).unwrap();
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "unconfigured_tools_follow_path_order_with_multiple_sdk_candidates",
            "--nocapture",
        ])
        .env(CHILD_ROOT, &root)
        .env_remove("CUDA_PATH")
        .env_remove("TENSORRT_ROOT_DIR")
        .env("PATH", &path)
        .env("LD_LIBRARY_PATH", &path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
