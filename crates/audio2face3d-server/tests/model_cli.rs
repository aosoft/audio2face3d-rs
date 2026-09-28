#![cfg(feature = "cli")]
const BINARY: &str = env!("CARGO_BIN_EXE_audio2face3d-server");
use std::process::{Command, Output};
fn run(args: &[&str]) -> Output {
    let dir = std::env::temp_dir().join(format!(
        "a2f-model-cli-{}-{}",
        env!("CARGO_PKG_NAME"),
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    // Model commands must never read GUI settings or start a server backend.
    std::fs::write(dir.join("gui.toml"), "invalid = [").unwrap();
    Command::new(BINARY)
        .args(args)
        .current_dir(dir)
        .env_remove("AUDIO2FACE3D_PLATFORM_CONFIG")
        .env("AUDIO2FACE3D_API_KEY", "")
        .env_remove("A2F_TEST_MISSING_MODEL_TOKEN")
        .output()
        .unwrap()
}
#[test]
fn model_command_matches_feature() {
    let output = run(&["--help"]);
    assert!(output.status.success(), "{output:?}");
    let help = String::from_utf8_lossy(&output.stdout);
    assert_eq!(
        help.lines()
            .any(|line| line.trim_start().starts_with("model ")),
        cfg!(feature = "model-management"),
        "{help}"
    );
    let output = run(&["model", "list"]);
    assert_eq!(
        output.status.success(),
        cfg!(feature = "model-management"),
        "{output:?}"
    );
    #[cfg(feature = "model-management")]
    {
        let list = String::from_utf8_lossy(&output.stdout);
        for preset in ["diffusion", "claire", "james", "mark", "emotion"] {
            assert!(list.contains(preset), "{list}");
        }
    }
}
#[cfg(feature = "model-management")]
#[test]
fn shared_options_and_errors_work_without_starting_the_app() {
    for command in [
        "download",
        "download-revision",
        "engine",
        "engine-onnx",
        "prepare",
    ] {
        let output = run(&["model", command, "--help"]);
        assert!(output.status.success(), "{command}: {output:?}");
    }
    let output = run(&["model", "engine", "mark", "--max-batch=0"]);
    assert!(!output.status.success());
    let missing = std::env::temp_dir().join(format!(
        "a2f-no-token-{}-{}",
        env!("CARGO_PKG_NAME"),
        std::process::id()
    ));
    let output = run(&[
        "model",
        "download",
        "mark",
        missing.to_str().unwrap(),
        "A2F_TEST_MISSING_MODEL_TOKEN",
    ]);
    assert!(!output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("A2F_TEST_MISSING_MODEL_TOKEN"),
        "{output:?}"
    );
    assert!(!missing.exists());
    // Global platform arguments must work before and after the subcommand.
    for args in [
        vec!["--runtime-search", "explicit", "model", "list"],
        vec!["model", "list", "--runtime-search", "explicit"],
    ] {
        let output = run(&args);
        assert!(output.status.success(), "{args:?}: {output:?}");
    }
}

#[cfg(feature = "model-management")]
#[test]
fn existing_snapshot_is_verified_and_mismatch_is_preserved() {
    let preset = audio2face3d::model_management::model_preset("mark").unwrap();
    let root = std::env::temp_dir().join(format!(
        "a2f-verify-{}-{}",
        env!("CARGO_PKG_NAME"),
        std::process::id()
    ));
    let model = root.join(preset.output_directory);
    std::fs::create_dir_all(&model).unwrap();
    for file in ["model.json", "network_info.json", "trt_info.json"] {
        std::fs::write(model.join(file), "{}").unwrap();
    }
    std::fs::write(model.join("network.onnx"), "abc").unwrap();
    let provenance = serde_json::json!({
        "schema_version": 1, "repository": preset.repository, "revision": preset.revision,
        "network_onnx_sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    });
    std::fs::write(model.join(".audio2x-source.json"), provenance.to_string()).unwrap();
    let args = [
        "model",
        "download",
        "mark",
        root.to_str().unwrap(),
        "A2F_TEST_MISSING_MODEL_TOKEN",
    ];
    let output = run(&args);
    assert!(output.status.success(), "{output:?}");
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("verified; skipped"),
        "{output:?}"
    );
    std::fs::write(model.join("network.onnx"), "modified").unwrap();
    let output = run(&args);
    assert!(!output.status.success(), "{output:?}");
    assert_eq!(
        std::fs::read(model.join("network.onnx")).unwrap(),
        b"modified"
    );
    std::fs::remove_dir_all(root).unwrap();
}
