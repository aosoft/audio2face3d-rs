use crate::{AudioBlock, AudioFormat, CurveFrame, CurveLayout, EmotionTrace, Progress};
use std::{sync::Arc, time::SystemTime};

#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct StreamInfo {
    pub audio_format: Option<AudioFormat>,
    pub curves: Option<Arc<CurveLayout>>,
    /// Correlation metadata, not the playback clock.
    pub started_at: Option<SystemTime>,
}
impl StreamInfo {
    pub fn new(audio_format: Option<AudioFormat>, curves: Option<Arc<CurveLayout>>) -> Self {
        Self {
            audio_format,
            curves,
            started_at: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Severity {
    Info,
    Warning,
    Error,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
}
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub progress: Progress,
}

/// A transport packet may contain zero, one or several frames independently of audio.
#[derive(Debug, Default, PartialEq)]
pub struct OutputBatch {
    pub audio: Option<AudioBlock>,
    pub curves: Vec<CurveFrame>,
    pub emotion: Option<EmotionTrace>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum OutputEvent {
    StreamInfo(StreamInfo),
    Audio(AudioBlock),
    Curves(CurveFrame),
    Emotion(EmotionTrace),
    Diagnostic(Diagnostic),
    ProcessingFinished,
    /// Only emitted after all data and the successful terminal condition.
    Completed(Summary),
}
