use audio2face3d_server::config::{BackendKind, Config, MockPattern};
use clap::Parser;
use std::{net::SocketAddr, path::PathBuf};
fn default_backend() -> &'static str {
    if cfg!(feature = "native") {
        "regression"
    } else {
        "mock"
    }
}
#[derive(Parser)]
#[command(version, about = "Audio2Face-3D controller gRPC server")]
pub struct Args {
    /// Single API key. Overrides AUDIO2FACE3D_API_KEY; neither means no authentication.
    #[arg(long, value_name = "KEY", allow_hyphen_values = false)]
    pub api_key: Option<std::ffi::OsString>,
    /// Health policy: public or same-as-inference.
    #[arg(long, default_value = "public", value_parser = parse_health_auth)]
    pub health_auth: audio2face3d_server::HealthAuth,
    #[arg(long, default_value = default_backend())]
    pub backend: BackendKind,
    #[arg(long, default_value = "127.0.0.1:52000")]
    pub listen: SocketAddr,
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Optional Audio2Emotion classifier descriptor (16000 Hz).
    #[arg(long)]
    pub emotion_model: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    pub device: usize,
    #[arg(long, default_value = "jaw-open-pulse")]
    pub mock_pattern: MockPattern,
    /// Select one of the 52 ACE curves instead of the preset pattern (mock only).
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(audio2face3d_server::animation::CURVE_NAMES))]
    pub mock_curve: Option<String>,
    /// Hold the selected curve at a constant weight; omit for a one-second pulse.
    #[arg(long, requires = "mock_curve", value_parser = parse_mock_value)]
    pub mock_value: Option<f32>,
    /// Add a fixed JawOpen baseline to another selected diagnostic curve.
    #[arg(long, requires = "mock_curve", value_parser = parse_mock_value)]
    pub mock_jaw_open: Option<f32>,
    #[arg(long, default_value_t = 1)]
    pub max_streams: usize,
    /// Maximum requests waiting for an execution slot, excluding active streams.
    #[arg(long, default_value_t = 64)]
    pub request_queue_capacity: usize,
    /// Maximum execution-slot wait in milliseconds; zero waits until cancellation.
    #[arg(long, default_value_t = 0)]
    pub request_queue_timeout_ms: u64,
    #[arg(long, default_value_t = 1_048_576)]
    pub max_message_bytes: usize,
    #[arg(long, default_value_t = 600)]
    pub max_audio_seconds: u32,
    #[arg(long, default_value_t = 16)]
    pub output_queue_capacity: usize,
    #[arg(long, default_value_t = 30_000)]
    pub input_idle_timeout_ms: u64,
    #[arg(long, default_value_t = 10_000)]
    pub output_timeout_ms: u64,
    #[arg(long, default_value_t = 5_000)]
    pub shutdown_timeout_ms: u64,
    /// Write the embedded descriptor set and exit (no listener).
    #[arg(long)]
    pub export_descriptor: Option<PathBuf>,
}

fn parse_mock_value(value: &str) -> Result<f32, String> {
    let value: f32 = value.parse().map_err(|_| "expected a number in [0, 1]")?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err("expected a finite number in [0, 1]".into());
    }
    Ok(value)
}

impl Args {
    pub fn config(&self) -> Config {
        Config {
            backend: self.backend,
            model: self.model.clone(),
            emotion_model: self.emotion_model.clone(),
            device: self.device,
            mock_pattern: self.mock_pattern,
            mock_curve: self.mock_curve.clone(),
            mock_value: self.mock_value,
            mock_jaw_open: self.mock_jaw_open,
            max_streams: self.max_streams,
            request_queue_capacity: self.request_queue_capacity,
            request_queue_timeout_ms: self.request_queue_timeout_ms,
            max_message_bytes: self.max_message_bytes,
            max_audio_seconds: self.max_audio_seconds,
            output_queue_capacity: self.output_queue_capacity,
            input_idle_timeout_ms: self.input_idle_timeout_ms,
            output_timeout_ms: self.output_timeout_ms,
            shutdown_timeout_ms: self.shutdown_timeout_ms,
        }
    }
}

fn parse_health_auth(value: &str) -> Result<audio2face3d_server::HealthAuth, &'static str> {
    match value {
        "public" => Ok(audio2face3d_server::HealthAuth::Public),
        "same-as-inference" => Ok(audio2face3d_server::HealthAuth::SameAsInference),
        _ => Err("expected public or same-as-inference"),
    }
}
impl std::fmt::Debug for Args {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Args")
            .field("backend", &self.backend)
            .field("listen", &self.listen)
            .field("api_key", &self.api_key.as_ref().map(|_| "[REDACTED]"))
            .finish_non_exhaustive()
    }
}
