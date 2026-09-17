use audio2face3d_types::{AudioFormat, Error, Result, SampleFormat};
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
    pub backend: BackendKind,
    pub model: Option<PathBuf>,
    pub emotion_model: Option<PathBuf>,
    pub device: usize,
    pub max_audio_seconds: u32,
    pub mock_pattern: MockPattern,
    pub mock_curve: Option<String>,
    pub mock_value: Option<f32>,
    pub mock_jaw_open: Option<f32>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            backend: BackendKind::Mock,
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
            && !crate::animation::CURVE_NAMES.contains(&curve.as_str())
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
