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
    assert!(
        Args::try_parse_from(["gui", "--infer"])
            .unwrap()
            .resolve()
            .is_err()
    );
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
    if let Some(path) = std::env::var_os("A2F_STARTUP_GUI_CONFIG") {
        args.extend(["--config".into(), path]);
    }
    let options = Args::try_parse_from(args).unwrap().resolve().unwrap();
    assert_eq!(
        options.request.native_runtime().unwrap().cuda_root(),
        Some(Path::new(&expected))
    );
}

#[test]
fn gui_config_paths_and_cli_overrides() {
    let fixture = Fixture::new();
    std::fs::create_dir(fixture.0.join("settings")).unwrap();
    let platform = fixture.write("platform.toml", "cuda-root='sdk/cuda'");
    let config = fixture.write(
        "settings/gui.toml",
        r#"
platform-config = "../platform.toml"
head = "head.glb"
[inference]
wav = "voice.wav"
auto-start = true
play-while-inferring = true
[local]
model = "model.json"
device = 3
[grpc]
endpoint = "http://example.test:52000"
api-key = "test-key"
"#,
    );
    let resolve = |extra: &[&str]| {
        let mut args = vec!["gui", "--config", config.to_str().unwrap()];
        args.extend_from_slice(extra);
        Args::try_parse_from(args).unwrap().resolve().unwrap()
    };
    let options = resolve(&[]);
    assert_eq!(options.head, Some(fixture.0.join("settings/head.glb")));
    assert_eq!(options.request.wav, fixture.0.join("settings/voice.wav"));
    assert_eq!(options.request.model, fixture.0.join("settings/model.json"));
    assert_eq!(options.request.device, 3);
    assert_eq!(options.request.api_key, "test-key");
    assert_eq!(options.request.endpoint, "http://example.test:52000");
    assert!(options.infer && options.request.pace_input);
    assert_eq!(
        options.request.runtime.cuda_root(),
        Some(fixture.0.join("settings/../sdk/cuda").as_path())
    );
    let options = resolve(&[
        "--head",
        "cli.glb",
        "--wav",
        "cli.wav",
        "--model",
        "cli.json",
        "--device",
        "0",
        "--endpoint",
        "http://localhost:1",
        "--api-key",
        "",
        "--infer=false",
        "--play-while-inferring=false",
        "--platform-config",
        platform.to_str().unwrap(),
        "--cuda-root",
        "cli-cuda",
    ]);
    assert_eq!(options.head.as_deref(), Some(Path::new("cli.glb")));
    assert_eq!(options.request.wav, Path::new("cli.wav"));
    assert_eq!(options.request.model, Path::new("cli.json"));
    assert_eq!(options.request.device, 0);
    assert_eq!(options.request.endpoint, "http://localhost:1");
    assert!(options.request.api_key.is_empty());
    assert!(!options.infer && !options.request.pace_input);
    assert_eq!(
        options.request.runtime.cuda_root(),
        Some(std::env::current_dir().unwrap().join("cli-cuda").as_path())
    );

    // An explicit CLI platform file replaces even an invalid GUI reference.
    std::fs::write(&config, "platform-config='missing.toml'").unwrap();
    resolve(&["--platform-config", platform.to_str().unwrap()]);
    assert!(
        Args::try_parse_from(["gui", "--config", config.to_str().unwrap()])
            .unwrap()
            .resolve()
            .is_err()
    );
}

#[test]
fn invalid_gui_settings_fail_without_fallback() {
    let fixture = Fixture::new();
    for text in [
        "unknown=1",
        "[inference]\nauto-start='yes'",
        "[local]\ndevice=-1",
        "[grpc]\nunknown=1",
        "head=''",
        "platform-config=''",
        "[inference]\nmode='typo'",
        "[inference]\nauto-start=true",
        "invalid toml",
    ] {
        let file = fixture.write("gui.toml", text);
        assert!(
            Args::try_parse_from(["gui", "--config", file.to_str().unwrap()])
                .unwrap()
                .resolve()
                .is_err(),
            "{text}"
        );
    }
    assert!(
        Args::try_parse_from([
            "gui",
            "--config",
            fixture.0.join("missing.toml").to_str().unwrap()
        ])
        .unwrap()
        .resolve()
        .is_err()
    );
}

#[test]
fn gui_example_is_valid() {
    let fixture = Fixture::new();
    let file = fixture.write("gui.toml", include_str!("../../../gui.example.toml"));
    // Avoid requiring a backend feature merely to validate the example's schema.
    let text = std::fs::read_to_string(&file)
        .unwrap()
        .replace("mode = \"grpc\"", "# mode = \"grpc\"");
    std::fs::write(&file, text).unwrap();
    Args::try_parse_from(["gui", "--config", file.to_str().unwrap()])
        .unwrap()
        .resolve()
        .unwrap();
}

#[test]
fn gui_platform_reference_precedes_environment_and_relative_config_uses_cwd() {
    let fixture = Fixture::new();
    fixture.write("gui.toml", "platform-config='shared.toml'");
    fixture.write("shared.toml", "cuda-root='gui-cuda'");
    fixture.write("environment.toml", "cuda-root='env-cuda'");
    let explicit = fixture.write("explicit.toml", "cuda-root='cli-cuda'");
    for cli in [false, true] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", "resolve_in_child", "--nocapture"])
            .current_dir(&fixture.0)
            .env("A2F_STARTUP_GUI_CONFIG", "gui.toml")
            .env("AUDIO2FACE3D_PLATFORM_CONFIG", "environment.toml")
            .env(
                "A2F_STARTUP_EXPECTED",
                fixture.0.join(if cli { "cli-cuda" } else { "gui-cuda" }),
            )
            .env_remove("A2F_STARTUP_EXPLICIT");
        if cli {
            command.env("A2F_STARTUP_EXPLICIT", &explicit);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn platform_example_is_valid() {
    let fixture = Fixture::new();
    let path = fixture.write(
        "platform.toml",
        include_str!("../../../platform.example.toml"),
    );
    let options = Args::try_parse_from(["gui", "--platform-config", path.to_str().unwrap()])
        .unwrap()
        .resolve()
        .unwrap();
    assert_eq!(
        options.request.runtime.cuda_root(),
        Some(fixture.0.join("SDK/CUDA/v12.9").as_path())
    );
}
