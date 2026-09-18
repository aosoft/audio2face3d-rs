use crate::types::{AudioFormat, Error, Result, SampleFormat};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendKind {
    Mock,
    Regression,
}
#[derive(Clone, Copy, Debug, Default)]
pub enum MockPattern {
    #[default]
    JawOpenPulse,
    EyeBlinkLeft,
    EyeBlinkRight,
    MouthSmileLeft,
    MouthSmileRight,
}

/// Engine settings only. Transport, request admission and application runtime are separate.
#[derive(Clone, Debug)]
pub struct Config {
    pub(crate) backend: BackendKind,
    pub(crate) model: Option<PathBuf>,
    pub(crate) emotion_model: Option<PathBuf>,
    pub(crate) device: usize,
    pub(crate) max_audio_seconds: u32,
    pub(crate) mock_pattern: MockPattern,
    pub(crate) mock_curve: Option<String>,
    pub(crate) mock_value: Option<f32>,
    pub(crate) mock_jaw_open: Option<f32>,
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
            max_audio_seconds: 600,
            mock_pattern: MockPattern::default(),
            mock_curve: None,
            mock_value: None,
            mock_jaw_open: None,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.device > i32::MAX as usize || self.max_audio_seconds == 0 {
            return Err(Error::invalid(
                "invalid device ordinal or audio duration limit",
            ));
        }
        if self.backend == BackendKind::Regression && self.model.is_none() {
            return Err(Error::invalid("model is required for Regression"));
        }
        if let Some(curve) = &self.mock_curve
            && !crate::inference::animation::CURVE_NAMES.contains(&curve.as_str())
        {
            return Err(Error::invalid("unknown mock curve"));
        }
        for value in [self.mock_value, self.mock_jaw_open].into_iter().flatten() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Error::invalid("mock weights must be finite and in 0..1"));
            }
        }
        Ok(())
    }
}
/// Current Regression/mock profile: PCM16 mono, converted to 16 kHz before inference.
pub fn validate_format(format: AudioFormat) -> Result<()> {
    if format.sample_format() != SampleFormat::Pcm16Le
        || format.channels() != 1
        || ![16000, 44100, 48000].contains(&format.sample_rate())
    {
        return Err(Error::invalid(
            "expected PCM16 little-endian, mono, 16000/44100/48000 Hz",
        ));
    }
    Ok(())
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
    pub fn max_audio_seconds(mut self, value: u32) -> Self {
        self.config.max_audio_seconds = value;
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
    pub fn build(self) -> Result<Config> {
        self.config.validate()?;
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
    pub fn max_audio_seconds(&self) -> u32 {
        self.max_audio_seconds
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
}
