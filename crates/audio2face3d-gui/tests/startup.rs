#![cfg(feature = "desktop")]
use audio2face3d::runtime::NativeSearchPolicy;
use audio2face3d_gui::startup::Args;
use clap::Parser;
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "a2f-gui-startup-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn write(&self, name: &str, text: &str) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, text).unwrap();
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn file_and_flags_reach_request_without_loading_native_libraries() {
    let fixture = Fixture::new();
    let file = fixture.write(
        "platform.toml",
        "cuda-root='cuda'\n[runtime]\ntensorrt-library-dirs=['trt/lib']\nsearch-policy='explicit'",
    );
    let options = Args::try_parse_from([
        "gui",
        "--platform-config",
        file.to_str().unwrap(),
        "--model",
        "model.json",
        "--wav",
        "voice.wav",
        "--head",
        "head.glb",
        "--infer",
        "--play-while-inferring",
        "--device",
        "2",
    ])
    .unwrap()
    .resolve()
    .unwrap();
    assert_eq!(options.head.as_deref(), Some(Path::new("head.glb")));
    assert_eq!(options.request.model, Path::new("model.json"));
    assert!(options.infer && options.request.pace_input);
    assert_eq!(options.request.device, 2);
    let native = options.request.native_runtime().unwrap();
    assert_eq!(native.cuda_root(), Some(fixture.0.join("cuda").as_path()));
    assert_eq!(native.tensorrt_library_dirs(), &[fixture.0.join("trt/lib")]);
    assert_eq!(native.search_policy(), NativeSearchPolicy::ExplicitOnly);
    let mut request = options.request;
    request.tensorrt_root = fixture.0.join("override");
    let native = request.native_runtime().unwrap();
    assert!(native.tensorrt_library_dirs().is_empty());
    assert_eq!(
        native.tensorrt_root(),
        Some(request.tensorrt_root.as_path())
    );
    assert_eq!(native.search_policy(), NativeSearchPolicy::ExplicitOnly);
}

#[test]
fn bad_explicit_config_and_unknown_arguments_are_errors() {
    let fixture = Fixture::new();
    for (name, text) in [
        ("unknown.toml", "unknown='x'"),
        (
            "conflict.toml",
            "[runtime]\ncuda-root='cuda'\ncuda-library-dirs=['lib']",
        ),
    ] {
        let file = fixture.write(name, text);
        assert!(
            Args::try_parse_from(["gui", "--platform-config", file.to_str().unwrap()])
                .unwrap()
                .resolve()
                .is_err()
        );
    }
    assert!(
        Args::try_parse_from([
            "gui",
            "--platform-config",
            fixture.0.join("missing.toml").to_str().unwrap()
        ])
        .unwrap()
        .resolve()
        .is_err()
    );
    assert!(Args::try_parse_from(["gui", "--unknown"]).is_err());
    assert!(Args::try_parse_from(["gui", "--infer"]).is_err());
    assert!(Args::try_parse_from(["gui", "--head", "a.glb", "b.glb"]).is_err());
}

#[test]
fn process_file_precedence_matches_the_existing_cli() {
    let fixture = Fixture::new();
    let user_root = fixture.0.join("user");
    std::fs::create_dir_all(user_root.join("audio2face3d")).unwrap();
    std::fs::write(
        user_root.join("audio2face3d/platform.toml"),
        "cuda-root='user-cuda'",
    )
    .unwrap();
    let cwd_file = fixture.write("platform.toml", "cuda-root='cwd-cuda'");
    let env_file = fixture.write("environment.toml", "cuda-root='env-cuda'");
    let explicit_file = fixture.write("explicit.toml", "cuda-root='explicit-cuda'");
    for (env_file, explicit, expected) in [
        (None, None, fixture.0.join("cwd-cuda")),
        (Some(env_file.as_path()), None, fixture.0.join("env-cuda")),
        (
            Some(env_file.as_path()),
            Some(explicit_file.as_path()),
            fixture.0.join("explicit-cuda"),
        ),
    ] {
        child(&fixture.0, &user_root, env_file, explicit, &expected);
    }
    std::fs::remove_file(cwd_file).unwrap();
    child(
        &fixture.0,
        &user_root,
        None,
        None,
        &user_root.join("audio2face3d/user-cuda"),
    );
}
fn child(
    cwd: &Path,
    user: &Path,
    env_file: Option<&Path>,
    explicit: Option<&Path>,
    expected: &Path,
) {
    let mut command = Command::new(std::env::current_exe().unwrap());
    command
        .args(["--exact", "resolve_in_child", "--nocapture"])
        .current_dir(cwd)
        .env("A2F_STARTUP_EXPECTED", expected)
        .env("LOCALAPPDATA", user)
        .env("XDG_CONFIG_HOME", user)
        .env_remove("AUDIO2FACE3D_PLATFORM_CONFIG")
        .env_remove("A2F_STARTUP_EXPLICIT");
    if let Some(path) = env_file {
        command.env("AUDIO2FACE3D_PLATFORM_CONFIG", path);
    }
    if let Some(path) = explicit {
        command.env("A2F_STARTUP_EXPLICIT", path);
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
#[test]
fn resolve_in_child() {
    let Some(expected) = std::env::var_os("A2F_STARTUP_EXPECTED") else {
        return;
    };
    let mut args = vec![std::ffi::OsString::from("gui")];
    if let Some(path) = std::env::var_os("A2F_STARTUP_EXPLICIT") {
        args.extend(["--platform-config".into(), path]);
    }
    let options = Args::try_parse_from(args).unwrap().resolve().unwrap();
    assert_eq!(
        options.request.native_runtime().unwrap().cuda_root(),
        Some(Path::new(&expected))
    );
}
