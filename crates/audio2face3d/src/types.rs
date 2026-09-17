//! Owned, transport- and runtime-independent inference data.
//!
//! This crate has no dependencies. PCM and curve buffers can be moved across
//! adapters without copying. Configuration validates explicitly before use.
mod animation;
mod audio;
mod emotion;
mod error;
mod event;
mod request;
mod time;

pub use animation::{CurveFrame, CurveLayout, EmotionLayout, LayoutId};
pub use audio::{AudioBlock, AudioFormat, InputChunk, PcmBuffer, SampleFormat};
pub use emotion::{EmotionKeyframe, EmotionTrace, EmotionValues};
pub use error::{Error, ErrorKind, Progress, RequestId, Result};
pub use event::{Diagnostic, OutputBatch, OutputEvent, Severity, StreamInfo, Summary};
pub use request::{
    BlendshapeParameters, EmotionParameters, EmotionPostProcessing, FaceParameters, RequestOptions,
    RequestOptionsBuilder,
};
pub use time::{MediaTime, SamplePosition};
