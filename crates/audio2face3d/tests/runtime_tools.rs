use audio2face3d::runtime::{NativeRuntimeConfig, NativeSearchPolicy, tools::NativeTool};
use std::{path::PathBuf, process::Command};
#[test]
fn tool_selection_changes_only_the_child_environment() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .join("../../temp/native-runtime-work/fixtures")
        .join(format!("tools-{}", std::process::id()));
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
