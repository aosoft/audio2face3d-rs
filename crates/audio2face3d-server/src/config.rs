use audio2face3d::inference;
use std::{path::PathBuf, time::Duration};

// Keep distinct public types for downstream code implementing a trait for both
// the server and inference enums. Parsing is owned by inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Mock,
    Regression,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MockPattern {
    JawOpenPulse,
    EyeBlinkLeft,
    EyeBlinkRight,
    MouthSmileLeft,
    MouthSmileRight,
}

/// Shared inference settings plus server-specific transport limits.
#[derive(Clone, Debug)]
pub struct Config {
    engine: inference::Config,
    transport: TransportConfig,
}

#[derive(Clone, Debug)]
struct TransportConfig {
    max_streams: usize,
    request_queue_capacity: usize,
    request_queue_timeout_ms: u64,
    max_message_bytes: usize,
    output_queue_capacity: usize,
    input_idle_timeout_ms: u64,
    output_timeout_ms: u64,
    shutdown_timeout_ms: u64,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            max_streams: 1,
            request_queue_capacity: 64,
            request_queue_timeout_ms: 0,
            max_message_bytes: 1_048_576,
            output_queue_capacity: 16,
            input_idle_timeout_ms: 30_000,
            output_timeout_ms: 10_000,
            shutdown_timeout_ms: 5_000,
        }
    }
}

impl TransportConfig {
    fn validate(&self) -> Result<(), String> {
        if self.max_streams == 0
            || self.request_queue_capacity == 0
            || self.output_queue_capacity == 0
            || self.max_message_bytes < 4096
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
        Ok(())
    }
}

fn validate_server_startup(engine: &inference::Config) -> Result<(), String> {
    match engine.backend() {
        inference::BackendKind::Mock if !cfg!(feature = "mock") => {
            return Err("mock backend is not compiled".into());
        }
        inference::BackendKind::Regression => {
            if !cfg!(feature = "native") {
                return Err("regression requires --features native".into());
            }
            if engine.model().as_ref().is_none_or(|path| !path.is_file()) {
                return Err("--model must name an existing regression descriptor".into());
            }
        }
        _ => {}
    }
    if engine
        .emotion_model()
        .as_ref()
        .is_some_and(|path| !path.is_file())
    {
        return Err("--emotion-model must name an existing descriptor".into());
    }
    Ok(())
}

impl From<BackendKind> for inference::BackendKind {
    fn from(value: BackendKind) -> Self {
        match value {
            BackendKind::Mock => Self::Mock,
            BackendKind::Regression => Self::Regression,
        }
    }
}

impl From<inference::BackendKind> for BackendKind {
    fn from(value: inference::BackendKind) -> Self {
        match value {
            inference::BackendKind::Mock => Self::Mock,
            inference::BackendKind::Regression => Self::Regression,
        }
    }
}

impl std::str::FromStr for BackendKind {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        value.parse::<inference::BackendKind>().map(Into::into)
    }
}

impl From<MockPattern> for inference::MockPattern {
    fn from(value: MockPattern) -> Self {
        match value {
            MockPattern::JawOpenPulse => Self::JawOpenPulse,
            MockPattern::EyeBlinkLeft => Self::EyeBlinkLeft,
            MockPattern::EyeBlinkRight => Self::EyeBlinkRight,
            MockPattern::MouthSmileLeft => Self::MouthSmileLeft,
            MockPattern::MouthSmileRight => Self::MouthSmileRight,
        }
    }
}

impl From<inference::MockPattern> for MockPattern {
    fn from(value: inference::MockPattern) -> Self {
        match value {
            inference::MockPattern::JawOpenPulse => Self::JawOpenPulse,
            inference::MockPattern::EyeBlinkLeft => Self::EyeBlinkLeft,
            inference::MockPattern::EyeBlinkRight => Self::EyeBlinkRight,
            inference::MockPattern::MouthSmileLeft => Self::MouthSmileLeft,
            inference::MockPattern::MouthSmileRight => Self::MouthSmileRight,
        }
    }
}

impl std::str::FromStr for MockPattern {
    type Err = String;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        value.parse::<inference::MockPattern>().map(Into::into)
    }
}

/// Consuming builder; validation runs before server resources are started.
#[derive(Clone, Debug)]
#[must_use]
pub struct ConfigBuilder {
    engine: inference::ConfigBuilder,
    transport: TransportConfig,
}

impl ConfigBuilder {
    /// Replaces all inference settings while retaining the transport limits.
    /// Later individual setters override the corresponding engine values.
    pub fn engine(mut self, value: inference::Config) -> Self {
        self.engine = value.into_builder();
        self
    }

    pub fn backend(mut self, value: BackendKind) -> Self {
        self.engine = self.engine.backend(value.into());
        self
    }

    pub fn model(mut self, value: impl Into<PathBuf>) -> Self {
        self.engine = self.engine.model(value);
        self
    }

    pub fn optional_model(mut self, value: Option<PathBuf>) -> Self {
        self.engine = self.engine.optional_model(value);
        self
    }

    pub fn emotion_model(mut self, value: impl Into<PathBuf>) -> Self {
        self.engine = self.engine.emotion_model(value);
        self
    }

    pub fn optional_emotion_model(mut self, value: Option<PathBuf>) -> Self {
        self.engine = self.engine.optional_emotion_model(value);
        self
    }

    pub fn device(mut self, value: usize) -> Self {
        self.engine = self.engine.device(value);
        self
    }

    pub fn max_audio_seconds(mut self, value: u32) -> Self {
        self.engine = self.engine.max_audio_seconds(value);
        self
    }

    pub fn mock_pattern(mut self, value: MockPattern) -> Self {
        self.engine = self.engine.mock_pattern(value.into());
        self
    }

    pub fn mock_curve(mut self, value: impl Into<String>) -> Self {
        self.engine = self.engine.mock_curve(value);
        self
    }

    pub fn optional_mock_curve(mut self, value: Option<String>) -> Self {
        self.engine = self.engine.optional_mock_curve(value);
        self
    }

    pub fn mock_value(mut self, value: f32) -> Self {
        self.engine = self.engine.mock_value(value);
        self
    }

    pub fn optional_mock_value(mut self, value: Option<f32>) -> Self {
        self.engine = self.engine.optional_mock_value(value);
        self
    }

    pub fn mock_jaw_open(mut self, value: f32) -> Self {
        self.engine = self.engine.mock_jaw_open(value);
        self
    }

    pub fn optional_mock_jaw_open(mut self, value: Option<f32>) -> Self {
        self.engine = self.engine.optional_mock_jaw_open(value);
        self
    }

    pub fn max_streams(mut self, value: usize) -> Self {
        self.transport.max_streams = value;
        self
    }

    pub fn request_queue_capacity(mut self, value: usize) -> Self {
        self.transport.request_queue_capacity = value;
        self
    }

    pub fn request_queue_timeout_ms(mut self, value: u64) -> Self {
        self.transport.request_queue_timeout_ms = value;
        self
    }

    pub fn max_message_bytes(mut self, value: usize) -> Self {
        self.transport.max_message_bytes = value;
        self
    }

    pub fn output_queue_capacity(mut self, value: usize) -> Self {
        self.transport.output_queue_capacity = value;
        self
    }

    pub fn input_idle_timeout_ms(mut self, value: u64) -> Self {
        self.transport.input_idle_timeout_ms = value;
        self
    }

    pub fn output_timeout_ms(mut self, value: u64) -> Self {
        self.transport.output_timeout_ms = value;
        self
    }

    pub fn shutdown_timeout_ms(mut self, value: u64) -> Self {
        self.transport.shutdown_timeout_ms = value;
        self
    }

    pub fn build(self) -> Result<Config, crate::ConfigError> {
        let config = self.build_settings()?;
        validate_server_startup(&config.engine).map_err(crate::ConfigError)?;
        Ok(config)
    }

    // The transport unit tests also validate settings without requiring a
    // compiled backend. Public build and Factory::prepare check availability.
    pub(crate) fn build_settings(self) -> Result<Config, crate::ConfigError> {
        let engine = self
            .engine
            .build()
            .map_err(|error| crate::ConfigError(error.message().to_owned()))?;
        self.transport.validate().map_err(crate::ConfigError)?;
        Ok(Config {
            engine,
            transport: self.transport,
        })
    }
}

impl Config {
    pub fn builder(backend: BackendKind) -> ConfigBuilder {
        ConfigBuilder {
            engine: inference::Config::builder(backend.into()),
            transport: TransportConfig::default(),
        }
    }

    pub fn into_builder(self) -> ConfigBuilder {
        ConfigBuilder {
            engine: self.engine.into_builder(),
            transport: self.transport,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        self.engine
            .validate()
            .map_err(|error| error.message().to_owned())?;
        self.transport.validate()?;
        validate_server_startup(&self.engine)
    }

    pub fn engine(&self) -> &inference::Config {
        &self.engine
    }

    pub fn backend(&self) -> BackendKind {
        self.engine.backend().into()
    }

    pub fn model(&self) -> &Option<PathBuf> {
        self.engine.model()
    }

    pub fn emotion_model(&self) -> &Option<PathBuf> {
        self.engine.emotion_model()
    }

    pub fn device(&self) -> usize {
        self.engine.device()
    }

    pub fn max_audio_seconds(&self) -> u32 {
        self.engine.max_audio_seconds()
    }

    pub fn mock_pattern(&self) -> MockPattern {
        self.engine.mock_pattern().into()
    }

    pub fn mock_curve(&self) -> &Option<String> {
        self.engine.mock_curve()
    }

    pub fn mock_value(&self) -> Option<f32> {
        self.engine.mock_value()
    }

    pub fn mock_jaw_open(&self) -> Option<f32> {
        self.engine.mock_jaw_open()
    }

    pub fn max_streams(&self) -> usize {
        self.transport.max_streams
    }

    pub fn request_queue_capacity(&self) -> usize {
        self.transport.request_queue_capacity
    }

    pub fn request_queue_timeout_ms(&self) -> u64 {
        self.transport.request_queue_timeout_ms
    }

    pub fn max_message_bytes(&self) -> usize {
        self.transport.max_message_bytes
    }

    pub fn output_queue_capacity(&self) -> usize {
        self.transport.output_queue_capacity
    }

    pub fn input_idle_timeout_ms(&self) -> u64 {
        self.transport.input_idle_timeout_ms
    }

    pub fn output_timeout_ms(&self) -> u64 {
        self.transport.output_timeout_ms
    }

    pub fn shutdown_timeout_ms(&self) -> u64 {
        self.transport.shutdown_timeout_ms
    }

    pub fn input_timeout(&self) -> Duration {
        Duration::from_millis(self.input_idle_timeout_ms())
    }

    pub fn output_timeout(&self) -> Duration {
        Duration::from_millis(self.output_timeout_ms())
    }
}
