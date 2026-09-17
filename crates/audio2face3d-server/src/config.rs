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
    pub backend: BackendKind,
    pub model: Option<PathBuf>,
    /// Optional Audio2Emotion classifier descriptor (16000 Hz).
    pub emotion_model: Option<PathBuf>,
    pub device: usize,
    pub mock_pattern: MockPattern,
    /// Select one of the 52 ACE curves instead of the preset pattern (mock only).
    pub mock_curve: Option<String>,
    /// Hold the selected curve at a constant weight; omit for a one-second pulse.
    pub mock_value: Option<f32>,
    /// Add a fixed JawOpen baseline to another selected diagnostic curve.
    pub mock_jaw_open: Option<f32>,
    pub max_streams: usize,
    /// Maximum requests waiting for an execution slot, excluding active streams.
    pub request_queue_capacity: usize,
    /// Maximum execution-slot wait in milliseconds; zero waits until cancellation.
    pub request_queue_timeout_ms: u64,
    pub max_message_bytes: usize,
    pub max_audio_seconds: u32,
    pub output_queue_capacity: usize,
    pub input_idle_timeout_ms: u64,
    pub output_timeout_ms: u64,
    pub shutdown_timeout_ms: u64,
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

impl Default for Config {
    fn default() -> Self {
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
