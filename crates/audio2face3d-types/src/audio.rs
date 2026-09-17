use crate::{EmotionKeyframe, Error, Result, SamplePosition};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SampleFormat {
    Pcm16Le,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AudioFormat {
    sample_rate: u32,
    channels: u16,
    sample_format: SampleFormat,
}
impl AudioFormat {
    pub const MONO_16KHZ: Self = Self {
        sample_rate: 16_000,
        channels: 1,
        sample_format: SampleFormat::Pcm16Le,
    };
    pub fn pcm16(sample_rate: u32, channels: u16) -> Result<Self> {
        if sample_rate == 0 || channels == 0 {
            return Err(Error::invalid(
                "audio rate and channel count must be positive",
            ));
        }
        Ok(Self {
            sample_rate,
            channels,
            sample_format: SampleFormat::Pcm16Le,
        })
    }
    pub const fn sample_rate(self) -> u32 {
        self.sample_rate
    }
    pub const fn channels(self) -> u16 {
        self.channels
    }
    pub const fn sample_format(self) -> SampleFormat {
        self.sample_format
    }
    pub fn bytes_per_frame(self) -> usize {
        usize::from(self.channels) * 2
    }
}

/// Owned PCM16LE. No implicit clone of audio; use Arc around a block to share it.
#[derive(Debug, PartialEq, Eq)]
pub struct PcmBuffer(Vec<u8>);
impl PcmBuffer {
    pub fn from_vec(bytes: Vec<u8>) -> Result<Self> {
        if !bytes.len().is_multiple_of(2) {
            return Err(Error::invalid("PCM16 needs complete samples"));
        }
        Ok(Self(bytes))
    }
    /// Copies borrowed bytes. from_vec transfers ownership without copying.
    pub fn copy_from_slice(bytes: &[u8]) -> Result<Self> {
        Self::from_vec(bytes.to_vec())
    }
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    pub fn into_vec(self) -> Vec<u8> {
        self.0
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn sample_frames(&self, format: AudioFormat) -> Result<u64> {
        if !self.0.len().is_multiple_of(format.bytes_per_frame()) {
            return Err(Error::invalid("PCM needs complete channel frames"));
        }
        u64::try_from(self.0.len() / format.bytes_per_frame())
            .map_err(|_| Error::invalid("PCM length overflow"))
    }
}

#[derive(Debug, PartialEq)]
pub struct InputChunk {
    pcm: PcmBuffer,
    emotions: Vec<EmotionKeyframe>,
}
impl InputChunk {
    pub fn new(pcm: PcmBuffer, emotions: Vec<EmotionKeyframe>) -> Self {
        Self { pcm, emotions }
    }
    pub fn pcm(&self) -> &PcmBuffer {
        &self.pcm
    }
    pub fn emotions(&self) -> &[EmotionKeyframe] {
        &self.emotions
    }
    pub fn validate(&self, format: AudioFormat) -> Result<()> {
        self.pcm.sample_frames(format)?;
        for emotion in &self.emotions {
            emotion.validate_input()?;
        }
        Ok(())
    }
    pub fn into_parts(self) -> (PcmBuffer, Vec<EmotionKeyframe>) {
        (self.pcm, self.emotions)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct AudioBlock {
    format: AudioFormat,
    position: SamplePosition,
    pcm: PcmBuffer,
}
impl AudioBlock {
    pub fn new(format: AudioFormat, position: SamplePosition, pcm: PcmBuffer) -> Result<Self> {
        position.checked_add(pcm.sample_frames(format)?)?;
        Ok(Self {
            format,
            position,
            pcm,
        })
    }
    pub fn format(&self) -> AudioFormat {
        self.format
    }
    pub fn position(&self) -> SamplePosition {
        self.position
    }
    pub fn pcm(&self) -> &PcmBuffer {
        &self.pcm
    }
    pub fn into_parts(self) -> (AudioFormat, SamplePosition, PcmBuffer) {
        (self.format, self.position, self.pcm)
    }
}
