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
#[command(name = "audio2face3d-gui", version)]
pub struct Args {
    #[command(flatten)]
    platform: PlatformArgs,
    /// GUI settings; defaults to gui.toml next to the executable.
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
}

#[derive(Default)]
pub struct Options {
    pub head: Option<PathBuf>,
    pub request: Request,
    pub infer: bool,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct Config {
    platform_config: Option<PathBuf>,
    head: Option<PathBuf>,
    inference: InferenceConfig,
    local: LocalConfig,
    grpc: GrpcConfig,
}

#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct InferenceConfig {
    mode: Option<Backend>,
    wav: Option<PathBuf>,
    play_while_inferring: bool,
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
        Ok(config)
    }
}

impl Args {
    /// CLI values override GUI settings; embedded hosts pass Options directly.
    pub fn resolve(self) -> Result<Options> {
        let exe = std::env::current_exe().map_err(|e| Error(e.to_string()))?;
        self.resolve_next_to(&exe)
    }

    fn resolve_next_to(self, executable: &Path) -> Result<Options> {
        let config = if let Some(path) = &self.config {
            let path = std::path::absolute(path).map_err(|e| Error(e.to_string()))?;
            Config::read(&path)?
        } else {
            let path = executable.with_file_name("gui.toml");
            match std::fs::metadata(&path) {
                Ok(_) => Config::read(&path)?,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Config::default(),
                Err(e) => {
                    return Err(Error(format!(
                        "Cannot access GUI config {}: {e}",
                        path.display()
                    )));
                }
            }
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
            pace_input: self
                .play_while_inferring
                .unwrap_or(config.inference.play_while_inferring),
            ..Default::default()
        };
        if let Some(mode) = self.mode.or(config.inference.mode) {
            request.mode = match mode {
                Backend::Local if cfg!(feature = "local") => Mode::Local,
                Backend::Grpc if cfg!(feature = "grpc") => Mode::Grpc,
                Backend::Mock if cfg!(feature = "mock") => Mode::Mock,
                _ => {
                    return Err(Error(format!(
                        "{mode:?} inference is not enabled in this build"
                    )));
                }
            };
        }
        let infer = self.infer.unwrap_or(config.inference.auto_start);
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
    fn default_config_is_next_to_executable_and_explicit_config_replaces_it() {
        let dir =
            std::env::temp_dir().join(format!("a2f-gui-default-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let exe = dir.join("audio2face3d-gui.exe");
        std::fs::write(dir.join("gui.toml"), "head='default.glb'").unwrap();
        let options = Args::try_parse_from(["gui"])
            .unwrap()
            .resolve_next_to(&exe)
            .unwrap();
        assert_eq!(options.head, Some(dir.join("default.glb")));
        let explicit = dir.join("other.toml");
        std::fs::write(&explicit, "head='other.glb'").unwrap();
        let options = Args::try_parse_from(["gui", "--config", explicit.to_str().unwrap()])
            .unwrap()
            .resolve_next_to(&exe)
            .unwrap();
        assert_eq!(options.head, Some(dir.join("other.glb")));
        std::fs::write(dir.join("gui.toml"), "invalid").unwrap();
        assert!(
            Args::try_parse_from(["gui"])
                .unwrap()
                .resolve_next_to(&exe)
                .is_err()
        );
        std::fs::remove_file(dir.join("gui.toml")).unwrap();
        assert!(
            Args::try_parse_from(["gui"])
                .unwrap()
                .resolve_next_to(&exe)
                .unwrap()
                .head
                .is_none()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
