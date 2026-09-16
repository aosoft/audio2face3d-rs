use clap::{Parser, ValueEnum};
use std::{net::SocketAddr, path::PathBuf, time::Duration};

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum BackendKind {
    Mock,
    Regression,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum MockPattern {
    JawOpenPulse,
    EyeBlinkLeft,
    EyeBlinkRight,
    MouthSmileLeft,
    MouthSmileRight,
}

#[derive(Clone, Debug, Parser)]
#[command(version, about = "Audio2Face-3D controller gRPC server")]
pub struct Config {
    #[arg(long, value_enum, default_value = "mock")]
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
    #[arg(long, value_enum, default_value = "jaw-open-pulse")]
    pub mock_pattern: MockPattern,
    /// Select one of the 52 ACE curves instead of the preset pattern (mock only).
    #[arg(long, value_parser = clap::builder::PossibleValuesParser::new(crate::animation::CURVE_NAMES))]
    pub mock_curve: Option<String>,
    /// Hold the selected curve at a constant weight; omit for a one-second pulse.
    #[arg(long, requires = "mock_curve", value_parser = parse_mock_value)]
    pub mock_value: Option<f32>,
    #[arg(long, default_value_t = 1)]
    pub max_streams: usize,
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

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.device > i32::MAX as usize {
            return Err("device ordinal exceeds i32 range".into());
        }
        if self.max_streams == 0
            || self.output_queue_capacity == 0
            || self.max_message_bytes < 4096
            || self.max_audio_seconds == 0
            || self.input_idle_timeout_ms == 0
            || self.output_timeout_ms == 0
            || self.shutdown_timeout_ms == 0
        {
            return Err(
                "limits and timeouts must be positive; max-message-bytes must be >= 4096".into(),
            );
        }
        if self.max_streams > tokio::sync::Semaphore::MAX_PERMITS
            || self.output_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
        {
            return Err("stream/queue limit exceeds Tokio capacity".into());
        }
        if self.backend != BackendKind::Mock
            && (self.mock_curve.is_some() || self.mock_value.is_some())
        {
            return Err("mock-curve/mock-value require the mock backend".into());
        }
        if self.backend == BackendKind::Regression {
            if !cfg!(feature = "runtime") {
                return Err("regression requires --features runtime".into());
            }
            if self.model.as_ref().is_none_or(|path| !path.is_file()) {
                return Err("--model must name an existing regression descriptor".into());
            }
        }
        if self
            .emotion_model
            .as_ref()
            .is_some_and(|path| !path.is_file())
        {
            return Err("--emotion-model must name an existing descriptor".into());
        }
        if self.backend == BackendKind::Mock
            && (self.model.is_some() || self.emotion_model.is_some() || self.device != 0)
        {
            return Err("model/device settings require the regression backend".into());
        }
        Ok(())
    }
    pub fn input_timeout(&self) -> Duration {
        Duration::from_millis(self.input_idle_timeout_ms)
    }
    pub fn output_timeout(&self) -> Duration {
        Duration::from_millis(self.output_timeout_ms)
    }
}

fn parse_mock_value(value: &str) -> Result<f32, String> {
    let value: f32 = value.parse().map_err(|_| "expected a number in [0, 1]")?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err("expected a finite number in [0, 1]".into());
    }
    Ok(value)
}
