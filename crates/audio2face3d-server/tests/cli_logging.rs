#![cfg(all(feature = "cli", feature = "mock"))]
use std::{
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
struct Process(Child);
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
#[test]
fn cli_logs_append_and_preserve_typed_json_fields() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../temp/logging-work/cli-tests");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!("{}.jsonl", std::process::id()));
    std::fs::write(&path, "{\"existing\":true}\n").unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_audio2face3d-server"));
    command
        .args([
            "--backend",
            "mock",
            "--listen",
            "127.0.0.1:0",
            "--log-format",
            "json",
            "--log-file",
        ])
        .arg(&path)
        .env("RUST_LOG", "off,audio2face3d_server=info")
        .env_remove("AUDIO2FACE3D_API_KEY")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = Process(command.spawn().unwrap());
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let text = std::fs::read_to_string(&path).unwrap();
        let values: Vec<serde_json::Value> = text
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        if let Some(serving) = values.iter().find(|v| v["message"] == "serving") {
            assert_eq!(values[0]["existing"], true);
            assert!(serving["timestamp_unix_ms"].is_u64());
            assert!(
                serving["fields"]["address"]
                    .as_str()
                    .unwrap()
                    .contains("127.0.0.1")
            );
            assert!(
                serving["fields"]["source"]
                    .as_str()
                    .unwrap()
                    .starts_with("audio2face3d_server")
            );
            break;
        }
        assert!(child.0.try_wait().unwrap().is_none(), "server exited");
        assert!(Instant::now() < deadline, "no startup record");
        std::thread::sleep(Duration::from_millis(10));
    }
}
#[test]
fn cli_reports_file_open_failure_and_help_lists_logging_options() {
    let binary = env!("CARGO_BIN_EXE_audio2face3d-server");
    let output = Command::new(binary)
        .args([
            "--backend",
            "mock",
            "--log-file",
            env!("CARGO_MANIFEST_DIR"),
        ])
        .env_remove("AUDIO2FACE3D_API_KEY")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let output = Command::new(binary).arg("--help").output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    for option in [
        "--log-format",
        "--log-file",
        "--log-overflow",
        "--log-queue-capacity",
    ] {
        assert!(text.contains(option));
    }
}
