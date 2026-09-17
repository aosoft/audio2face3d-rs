use std::{path::PathBuf, time::Duration};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Mock,
    Regression,
}

#[derive(Clone, Copy, Debug)]
pub enum MockPattern {
    JawOpenPulse,
    EyeBlinkLeft,
    EyeBlinkRight,
    MouthSmileLeft,
    MouthSmileRight,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub(crate) backend: BackendKind,
    pub(crate) model: Option<PathBuf>,
    /// Optional Audio2Emotion classifier descriptor (16000 Hz).
    pub(crate) emotion_model: Option<PathBuf>,
    pub(crate) device: usize,
    pub(crate) mock_pattern: MockPattern,
    /// Select one of the 52 ACE curves instead of the preset pattern (mock only).
    pub(crate) mock_curve: Option<String>,
    /// Hold the selected curve at a constant weight; omit for a one-second pulse.
    pub(crate) mock_value: Option<f32>,
    /// Add a fixed JawOpen baseline to another selected diagnostic curve.
    pub(crate) mock_jaw_open: Option<f32>,
    pub(crate) max_streams: usize,
    /// Maximum requests waiting for an execution slot, excluding active streams.
    pub(crate) request_queue_capacity: usize,
    /// Maximum execution-slot wait in milliseconds; zero waits until cancellation.
    pub(crate) request_queue_timeout_ms: u64,
    pub(crate) max_message_bytes: usize,
    pub(crate) max_audio_seconds: u32,
    pub(crate) output_queue_capacity: usize,
    pub(crate) input_idle_timeout_ms: u64,
    pub(crate) output_timeout_ms: u64,
    pub(crate) shutdown_timeout_ms: u64,
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self
            .mock_curve
            .as_ref()
            .is_some_and(|c| !crate::animation::CURVE_NAMES.contains(&c.as_str()))
        {
            return Err("unknown mock curve".into());
        }
        if (self.mock_value.is_some() || self.mock_jaw_open.is_some()) && self.mock_curve.is_none()
        {
            return Err("mock weights require mock-curve".into());
        }
        for v in [self.mock_value, self.mock_jaw_open].into_iter().flatten() {
            if !v.is_finite() || !(0.0..=1.0).contains(&v) {
                return Err("mock weights must be finite and in 0..1".into());
            }
        }
        if self.device > i32::MAX as usize {
            return Err("device ordinal exceeds i32 range".into());
        }
        if self.max_streams == 0
            || self.request_queue_capacity == 0
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
            || self.request_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
            || self.output_queue_capacity > tokio::sync::Semaphore::MAX_PERMITS
        {
            return Err("stream/queue limit exceeds Tokio capacity".into());
        }
        if self.backend != BackendKind::Mock
            && (self.mock_curve.is_some()
                || self.mock_value.is_some()
                || self.mock_jaw_open.is_some())
        {
            return Err("mock-curve/mock-value require the mock backend".into());
        }
        if self.mock_jaw_open.is_some() && self.mock_curve.as_deref() == Some("JawOpen") {
            return Err("mock-jaw-open requires a selected curve other than JawOpen".into());
        }
        if self.backend == BackendKind::Mock && !cfg!(feature = "mock") {
            return Err("mock backend is not compiled".into());
        }
        if self.backend == BackendKind::Regression {
            if !cfg!(feature = "native") {
                return Err("regression requires --features native".into());
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

impl From<MockPattern> for audio2face3d::inference::MockPattern {
    fn from(pattern: MockPattern) -> Self {
        match pattern {
            MockPattern::JawOpenPulse => Self::JawOpenPulse,
            MockPattern::EyeBlinkLeft => Self::EyeBlinkLeft,
            MockPattern::EyeBlinkRight => Self::EyeBlinkRight,
            MockPattern::MouthSmileLeft => Self::MouthSmileLeft,
            MockPattern::MouthSmileRight => Self::MouthSmileRight,
        }
    }
}

impl Config {
    pub(crate) fn default() -> Self {
        Self {
            backend: if cfg!(feature = "native") {
                BackendKind::Regression
            } else {
                BackendKind::Mock
            },
            model: None,
            emotion_model: None,
            device: 0,
            mock_pattern: MockPattern::JawOpenPulse,
            mock_curve: None,
            mock_value: None,
            mock_jaw_open: None,
            max_streams: 1,
            request_queue_capacity: 64,
            request_queue_timeout_ms: 0,
            max_message_bytes: 1_048_576,
            max_audio_seconds: 600,
            output_queue_capacity: 16,
            input_idle_timeout_ms: 30_000,
            output_timeout_ms: 10_000,
            shutdown_timeout_ms: 5_000,
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "mock" => Ok(Self::Mock),
            "regression" => Ok(Self::Regression),
            _ => Err("unknown BackendKind".into()),
        }
    }
}

impl std::str::FromStr for MockPattern {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "jaw-open-pulse" => Ok(Self::JawOpenPulse),
            "eye-blink-left" => Ok(Self::EyeBlinkLeft),
            "eye-blink-right" => Ok(Self::EyeBlinkRight),
            "mouth-smile-left" => Ok(Self::MouthSmileLeft),
            "mouth-smile-right" => Ok(Self::MouthSmileRight),
            _ => Err("unknown MockPattern".into()),
        }
    }
}

/// Consuming builder; validation runs in build before resources are started.
#[derive(Clone, Debug)]
#[must_use]
pub struct ConfigBuilder {
    config: Config,
}
impl Config {
    pub fn builder(backend: BackendKind) -> ConfigBuilder {
        ConfigBuilder {
            config: Config {
                backend,
                ..Config::default()
            },
        }
    }
}
impl ConfigBuilder {
    pub fn backend(mut self, value: BackendKind) -> Self {
        self.config.backend = value;
        self
    }
    pub fn model(mut self, value: impl Into<PathBuf>) -> Self {
        self.config.model = Some(value.into());
        self
    }
    pub fn optional_model(mut self, value: Option<PathBuf>) -> Self {
        self.config.model = value;
        self
    }
    pub fn emotion_model(mut self, value: impl Into<PathBuf>) -> Self {
        self.config.emotion_model = Some(value.into());
        self
    }
    pub fn optional_emotion_model(mut self, value: Option<PathBuf>) -> Self {
        self.config.emotion_model = value;
        self
    }
    pub fn device(mut self, value: usize) -> Self {
        self.config.device = value;
        self
    }
    pub fn mock_pattern(mut self, value: MockPattern) -> Self {
        self.config.mock_pattern = value;
        self
    }
    pub fn mock_curve(mut self, value: impl Into<String>) -> Self {
        self.config.mock_curve = Some(value.into());
        self
    }
    pub fn optional_mock_curve(mut self, value: Option<String>) -> Self {
        self.config.mock_curve = value;
        self
    }
    pub fn mock_value(mut self, value: f32) -> Self {
        self.config.mock_value = Some(value);
        self
    }
    pub fn optional_mock_value(mut self, value: Option<f32>) -> Self {
        self.config.mock_value = value;
        self
    }
    pub fn mock_jaw_open(mut self, value: f32) -> Self {
        self.config.mock_jaw_open = Some(value);
        self
    }
    pub fn optional_mock_jaw_open(mut self, value: Option<f32>) -> Self {
        self.config.mock_jaw_open = value;
        self
    }
    pub fn max_streams(mut self, value: usize) -> Self {
        self.config.max_streams = value;
        self
    }
    pub fn request_queue_capacity(mut self, value: usize) -> Self {
        self.config.request_queue_capacity = value;
        self
    }
    pub fn request_queue_timeout_ms(mut self, value: u64) -> Self {
        self.config.request_queue_timeout_ms = value;
        self
    }
    pub fn max_message_bytes(mut self, value: usize) -> Self {
        self.config.max_message_bytes = value;
        self
    }
    pub fn max_audio_seconds(mut self, value: u32) -> Self {
        self.config.max_audio_seconds = value;
        self
    }
    pub fn output_queue_capacity(mut self, value: usize) -> Self {
        self.config.output_queue_capacity = value;
        self
    }
    pub fn input_idle_timeout_ms(mut self, value: u64) -> Self {
        self.config.input_idle_timeout_ms = value;
        self
    }
    pub fn output_timeout_ms(mut self, value: u64) -> Self {
        self.config.output_timeout_ms = value;
        self
    }
    pub fn shutdown_timeout_ms(mut self, value: u64) -> Self {
        self.config.shutdown_timeout_ms = value;
        self
    }
    pub fn build(self) -> Result<Config, crate::ConfigError> {
        self.config.validate().map_err(crate::ConfigError)?;
        Ok(self.config)
    }
}

impl Config {
    pub fn into_builder(self) -> ConfigBuilder {
        ConfigBuilder { config: self }
    }
    pub fn backend(&self) -> BackendKind {
        self.backend
    }
    pub fn model(&self) -> &Option<PathBuf> {
        &self.model
    }
    pub fn emotion_model(&self) -> &Option<PathBuf> {
        &self.emotion_model
    }
    pub fn device(&self) -> usize {
        self.device
    }
    pub fn mock_pattern(&self) -> MockPattern {
        self.mock_pattern
    }
    pub fn mock_curve(&self) -> &Option<String> {
        &self.mock_curve
    }
    pub fn mock_value(&self) -> Option<f32> {
        self.mock_value
    }
    pub fn mock_jaw_open(&self) -> Option<f32> {
        self.mock_jaw_open
    }
    pub fn max_streams(&self) -> usize {
        self.max_streams
    }
    pub fn request_queue_capacity(&self) -> usize {
        self.request_queue_capacity
    }
    pub fn request_queue_timeout_ms(&self) -> u64 {
        self.request_queue_timeout_ms
    }
    pub fn max_message_bytes(&self) -> usize {
        self.max_message_bytes
    }
    pub fn max_audio_seconds(&self) -> u32 {
        self.max_audio_seconds
    }
    pub fn output_queue_capacity(&self) -> usize {
        self.output_queue_capacity
    }
    pub fn input_idle_timeout_ms(&self) -> u64 {
        self.input_idle_timeout_ms
    }
    pub fn output_timeout_ms(&self) -> u64 {
        self.output_timeout_ms
    }
    pub fn shutdown_timeout_ms(&self) -> u64 {
        self.shutdown_timeout_ms
    }
}
