//! Standard-host arguments. Embedded hosts pass resolved options without reading process state.
use crate::{
    core::{Error, Result},
    inference::{Mode, Request},
};
use audio2face3d::runtime::cli::PlatformArgs;
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
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
    #[arg(long, default_value = "http://127.0.0.1:52000")]
    endpoint: String,
    /// Native GPU device index.
    #[arg(long, default_value_t = 0)]
    device: usize,
    /// Start inference after the window opens; otherwise only populate the controls.
    #[arg(long, requires = "wav")]
    infer: bool,
    /// Pace input and play audio/curves while inference is running.
    #[arg(long)]
    play_while_inferring: bool,
}

#[derive(Default)]
pub struct Options {
    pub head: Option<PathBuf>,
    pub request: Request,
    pub infer: bool,
}

impl Args {
    /// Uses the same file selection, validation and flag precedence as the existing CLIs.
    pub fn resolve(self) -> Result<Options> {
        let mut request = Request {
            runtime: self.platform.resolve().map_err(|e| Error(e.to_string()))?,
            wav: self.wav.unwrap_or_default(),
            model: self.model.unwrap_or_default(),
            endpoint: self.endpoint,
            device: self.device,
            pace_input: self.play_while_inferring,
            ..Default::default()
        };
        if let Some(mode) = self.mode {
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
        if self.infer {
            request.validate()?;
        }
        Ok(Options {
            head: self.head.or(self.positional_head),
            request,
            infer: self.infer,
        })
    }
}
