#![cfg(feature = "cli")]
use std::process::Command;
const SECRET: &str = "SentinelApiKeyNeverPrint";
fn run(args: &[&str], environment: Option<&str>) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_audio2face3d-server"));
    command
        .args(args)
        .env_remove("AUDIO2FACE3D_API_KEY")
        .env_remove("RUST_LOG");
    if let Some(value) = environment {
        command.env("AUDIO2FACE3D_API_KEY", value);
    }
    command.output().unwrap()
}
#[test]
fn parse_errors_help_and_startup_never_echo_secrets() {
    for args in [
        vec!["--api-key", SECRET, "--api-key", SECRET],
        vec!["--api-key", SECRET, "--listen", SECRET],
        vec!["--api-key", SECRET, "--unknown"],
        vec!["--api-key"],
        vec!["--api-key", ""],
    ] {
        let result = run(&args, None);
        assert!(!result.status.success());
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!text.contains(SECRET), "{text}");
    }
    for args in [
        vec!["--help"],
        vec!["--version"],
        vec!["--api-key", SECRET, "--help"],
    ] {
        let result = run(&args, Some(""));
        assert!(result.status.success());
        assert!(!String::from_utf8_lossy(&result.stdout).contains(SECRET));
        assert!(!String::from_utf8_lossy(&result.stderr).contains(SECRET));
    }
}
#[test]
fn descriptor_export_ignores_invalid_auth_environment() {
    let path = std::env::temp_dir().join(format!("a2f-descriptor-{}.bin", std::process::id()));
    let result = run(&["--export-descriptor", path.to_str().unwrap()], Some(""));
    assert!(result.status.success());
    assert_eq!(
        std::fs::read(&path).unwrap(),
        audio2face3d_server::proto::DESCRIPTOR
    );
    std::fs::remove_file(path).unwrap();
}
