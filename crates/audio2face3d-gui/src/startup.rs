//! Standard-host arguments. Embedded hosts pass resolved options without reading process state.
use crate::{
    core::{Error, Result},
    inference::{Mode, Request},
};
use audio2face3d::runtime::cli::PlatformArgs;
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Local,
    Grpc,
    Mock,
}

/// Inspect Audio2Face-3D inference and blendshape animation.
#[derive(Debug, Parser)]
#[command(
    name = "audio2face3d-gui",
    version,
    args_conflicts_with_subcommands = true
)]
pub struct Args {
    #[cfg(feature = "obj2morph")]
    #[command(subcommand)]
    pub command: Option<Command>,
    #[command(flatten)]
    platform: PlatformArgs,
    /// GUI settings; searches the working directory, then the executable directory.
    #[arg(long, value_name = "TOML")]
    config: Option<PathBuf>,
    /// Optional legacy positional head GLB.
    #[arg(value_name = "HEAD_GLB", conflicts_with = "head")]
    positional_head: Option<PathBuf>,
    /// Override the standard head located next to the executable.
    #[arg(long, value_name = "GLB")]
    head: Option<PathBuf>,
    /// Inference backend (must be enabled in this build).
    #[arg(long, value_enum)]
    mode: Option<Backend>,
    /// Local model.json; required when starting local inference.
    #[arg(long, value_name = "JSON")]
    model: Option<PathBuf>,
    #[cfg(feature = "emotion")]
    /// Local Audio2Emotion model.json. Remote models are configured on the server.
    #[arg(long, value_name = "JSON")]
    emotion_model: Option<PathBuf>,
    /// Input WAV. Loading is deferred until inference starts.
    #[arg(long, value_name = "WAV")]
    wav: Option<PathBuf>,
    /// gRPC server address.
    #[arg(long)]
    endpoint: Option<String>,
    /// gRPC API key (prefer a private configuration file).
    #[arg(long)]
    api_key: Option<String>,
    /// Native GPU device index.
    #[arg(long)]
    device: Option<usize>,
    /// Start inference after the window opens; otherwise only populate the controls.
    #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    infer: Option<bool>,
    /// Pace input and play audio/curves while inference is running.
    #[arg(long, num_args = 0..=1, default_missing_value = "true", require_equals = true)]
    play_while_inferring: Option<bool>,
    /// Streaming startup/rebuffer target in milliseconds (10..=10000; default 100).
    #[arg(long)]
    stream_buffer_ms: Option<u32>,
}

#[cfg(feature = "obj2morph")]
#[derive(Debug, clap::Subcommand)]
pub enum Command {
    /// Convert neutral and expression OBJ files to a GUI head GLB (no window).
    Obj2morph(crate::obj2morph::cli::Arguments),
}

pub use crate::desktop::Options;

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct Config {
    platform_config: Option<PathBuf>,
    head: Option<PathBuf>,
    inference: InferenceConfig,
    local: LocalConfig,
    grpc: GrpcConfig,
    #[cfg(feature = "emotion")]
    emotion: crate::emotion::Settings,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct InferenceConfig {
    mode: Option<Backend>,
    wav: Option<PathBuf>,
    play_while_inferring: bool,
    stream_buffer_ms: Option<u32>,
    auto_start: bool,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LocalConfig {
    model: Option<PathBuf>,
    device: Option<usize>,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct GrpcConfig {
    endpoint: Option<String>,
    api_key: Option<String>,
}

impl Config {
    fn read(file: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(file)
            .map_err(|e| Error(format!("Cannot read GUI config {}: {e}", file.display())))?;
        let mut config: Self = toml::from_str(&text)
            .map_err(|e| Error(format!("Invalid GUI config {}: {e}", file.display())))?;
        for (key, path) in [
            ("platform-config", &mut config.platform_config),
            ("head", &mut config.head),
            ("inference.wav", &mut config.inference.wav),
            ("local.model", &mut config.local.model),
        ] {
            if let Some(path) = path {
                if path.as_os_str().is_empty() {
                    return Err(Error(format!(
                        "Invalid GUI config {}: {key} must not be empty",
                        file.display()
                    )));
                }
                if path.is_relative() {
                    *path = file.parent().unwrap().join(&*path);
                }
            }
        }
        #[cfg(feature = "emotion")]
        {
            config.emotion.validate()?;
            if let Some(path) = &mut config.emotion.model
                && path.is_relative()
            {
                *path = file.parent().unwrap().join(&*path);
            }
        }
        Ok(config)
    }
}

impl Args {
    /// CLI values override GUI settings; embedded hosts pass Options directly.
    pub fn resolve(self) -> Result<Options> {
        #[cfg(feature = "obj2morph")]
        if self.command.is_some() {
            return Err(Error(
                "Dispatch the conversion subcommand before resolving GUI settings".into(),
            ));
        }
        let exe = std::env::current_exe().map_err(|e| Error(e.to_string()))?;
        let cwd = std::env::current_dir().map_err(|e| Error(e.to_string()))?;
        self.resolve_at(&cwd, &exe)
    }

    fn resolve_at(self, cwd: &Path, executable: &Path) -> Result<Options> {
        let config = if let Some(path) = &self.config {
            Config::read(&cwd.join(path))?
        } else {
            let mut selected = Config::default();
            for path in [cwd.join("gui.toml"), executable.with_file_name("gui.toml")] {
                match std::fs::metadata(&path) {
                    Ok(_) => {
                        selected = Config::read(&path)?;
                        break;
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                    Err(e) => {
                        return Err(Error(format!(
                            "Cannot access GUI config {}: {e}",
                            path.display()
                        )));
                    }
                }
            }
            selected
        };
        let mut request = Request {
            runtime: self
                .platform
                .resolve_with_config(config.platform_config.as_deref())
                .map_err(|e| Error(e.to_string()))?,
            wav: self.wav.or(config.inference.wav).unwrap_or_default(),
            model: self.model.or(config.local.model).unwrap_or_default(),
            endpoint: self
                .endpoint
                .or(config.grpc.endpoint)
                .unwrap_or_else(|| "http://127.0.0.1:52000".into()),
            api_key: self.api_key.or(config.grpc.api_key).unwrap_or_default(),
            device: self.device.or(config.local.device).unwrap_or_default(),
            stream_buffer_ms: self
                .stream_buffer_ms
                .or(config.inference.stream_buffer_ms)
                .unwrap_or(100),
            pace_input: self
                .play_while_inferring
                .unwrap_or(config.inference.play_while_inferring),
            ..Default::default()
        };
        #[cfg(feature = "emotion")]
        {
            request.emotion = config.emotion;
            if let Some(path) = self.emotion_model {
                request.emotion.model = Some(path);
            }
            request.emotion.validate()?;
        }
        request.validate_stream_buffer()?;
        let infer = self.infer.unwrap_or(config.inference.auto_start);
        if let Some(mode) = self.mode.or(config.inference.mode) {
            request.mode = match mode {
                Backend::Local if cfg!(feature = "local") => Mode::Local,
                Backend::Grpc if cfg!(feature = "grpc") => Mode::Grpc,
                Backend::Mock if cfg!(feature = "mock") => Mode::Mock,
                _ if self.mode.is_none() && !infer => request.mode,
                _ => {
                    return Err(Error(format!(
                        "{mode:?} inference is not enabled in this build"
                    )));
                }
            };
        }
        if infer {
            request.validate()?;
        }
        Ok(Options {
            head: self.head.or(self.positional_head).or(config.head),
            request,
            infer,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stream_buffer_defaults_config_cli_precedence_and_validation() {
        let dir = tempfile::tempdir().unwrap();
        let exe = dir.path().join("gui.exe");
        let resolve = |args: &[&str]| {
            Args::try_parse_from(args)
                .unwrap()
                .resolve_at(dir.path(), &exe)
        };
        assert_eq!(resolve(&["gui"]).unwrap().request.stream_buffer_ms, 100);
        std::fs::write(
            dir.path().join("gui.toml"),
            "[inference]\nstream-buffer-ms=500",
        )
        .unwrap();
        assert_eq!(resolve(&["gui"]).unwrap().request.stream_buffer_ms, 500);
        assert_eq!(
            resolve(&["gui", "--stream-buffer-ms", "750"])
                .unwrap()
                .request
                .stream_buffer_ms,
            750
        );
        for invalid in ["0", "9", "10001"] {
            assert!(resolve(&["gui", "--stream-buffer-ms", invalid]).is_err());
        }
        std::fs::write(
            dir.path().join("gui.toml"),
            "[inference]\nstream-buffer-ms=0",
        )
        .unwrap();
        assert!(resolve(&["gui"]).is_err());
    }

    #[test]
    fn emotion_configuration_feature_gate() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("gui.toml");
        std::fs::write(&file, "[emotion]\nmodel='models/emotion/model.json'\nsend-to-server=true\n[emotion.beginning]\njoy=0.8").unwrap();
        #[cfg(feature = "emotion")]
        {
            let config = Config::read(&file).unwrap();
            assert_eq!(
                config.emotion.model,
                Some(dir.path().join("models/emotion/model.json"))
            );
            assert_eq!(config.emotion.beginning["joy"], 0.8);
            std::fs::write(&file, "[emotion]\nmodel=''").unwrap();
            assert!(Config::read(&file).is_err());
        }
        #[cfg(not(feature = "emotion"))]
        assert!(Config::read(&file).is_err());
    }

    #[test]
    #[cfg(not(any(feature = "grpc", feature = "local", feature = "mock")))]
    fn backend_free_build_opens_for_manual_inspection() {
        assert!(!crate::inference::inference_available());
        assert_eq!(Request::default().mode, Mode::Disabled);
        let dir = std::env::temp_dir().join(format!("a2f-manual-only-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("gui.toml");
        std::fs::write(&config, "[inference]\nmode='local'").unwrap();
        let exe = dir.join("gui.exe");
        let options = Args::try_parse_from(["gui"])
            .unwrap()
            .resolve_at(&dir, &exe)
            .unwrap();
        assert_eq!(options.request.mode, Mode::Disabled);
        assert!(!options.infer);
        assert!(options.request.validate().is_err());
        assert!(
            Args::try_parse_from(["gui", "--infer"])
                .unwrap()
                .resolve_at(&dir, &exe)
                .is_err()
        );
        std::fs::remove_file(config).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    #[cfg(all(feature = "grpc", not(feature = "local")))]
    fn unavailable_config_mode_falls_back_only_for_interactive_startup() {
        let dir =
            std::env::temp_dir().join(format!("a2f-gui-backend-fallback-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("gui.exe");
        let config = dir.join("gui.toml");
        std::fs::write(
            &config,
            "[inference]\nmode='local'\n[local]\nmodel='model.json'",
        )
        .unwrap();
        let resolve = |args: &[&str]| Args::try_parse_from(args).unwrap().resolve_at(&dir, &exe);
        let options = resolve(&["gui"]).unwrap();
        assert_eq!(options.request.mode, Mode::Grpc);
        assert_eq!(options.request.model, dir.join("model.json"));
        assert!(!options.infer);
        assert!(resolve(&["gui", "--mode", "local"]).is_err());
        assert!(resolve(&["gui", "--infer"]).is_err());
        assert_eq!(
            resolve(&["gui", "--mode", "grpc"]).unwrap().request.mode,
            Mode::Grpc
        );
        std::fs::write(&config, "[inference]\nmode='local'\nauto-start=true").unwrap();
        assert!(resolve(&["gui"]).is_err());
        assert_eq!(
            resolve(&["gui", "--infer=false"]).unwrap().request.mode,
            Mode::Grpc
        );
        std::fs::remove_file(config).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn config_search_prefers_explicit_then_working_directory_then_executable() {
        let dir =
            std::env::temp_dir().join(format!("a2f-gui-default-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("audio2face3d-gui.exe");
        let cwd = dir.join("working");
        std::fs::create_dir_all(&cwd).unwrap();
        std::fs::write(dir.join("gui.toml"), "head='default.glb'").unwrap();
        let options = Args::try_parse_from(["gui"])
            .unwrap()
            .resolve_at(&cwd, &exe)
            .unwrap();
        assert_eq!(options.head, Some(dir.join("default.glb")));
        std::fs::write(cwd.join("gui.toml"), "head='working.glb'").unwrap();
        let options = Args::try_parse_from(["gui"])
            .unwrap()
            .resolve_at(&cwd, &exe)
            .unwrap();
        assert_eq!(options.head, Some(cwd.join("working.glb")));
        let explicit = cwd.join("other.toml");
        std::fs::write(&explicit, "head='other.glb'").unwrap();
        let options = Args::try_parse_from(["gui", "--config", "other.toml"])
            .unwrap()
            .resolve_at(&cwd, &exe)
            .unwrap();
        assert_eq!(options.head, Some(cwd.join("other.glb")));
        assert!(
            Args::try_parse_from(["gui", "--config", "missing.toml"])
                .unwrap()
                .resolve_at(&cwd, &exe)
                .is_err()
        );
        std::fs::write(cwd.join("gui.toml"), "invalid").unwrap();
        assert!(
            Args::try_parse_from(["gui"])
                .unwrap()
                .resolve_at(&cwd, &exe)
                .is_err()
        );
        std::fs::remove_file(cwd.join("gui.toml")).unwrap();
        std::fs::write(dir.join("gui.toml"), "invalid").unwrap();
        assert!(
            Args::try_parse_from(["gui"])
                .unwrap()
                .resolve_at(&cwd, &exe)
                .is_err()
        );
        std::fs::remove_file(dir.join("gui.toml")).unwrap();
        assert!(
            Args::try_parse_from(["gui"])
                .unwrap()
                .resolve_at(&cwd, &exe)
                .unwrap()
                .head
                .is_none()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
