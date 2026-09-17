use crate::types::{Error, ErrorKind};
use std::collections::VecDeque;

pub const SAMPLE_RATE: u64 = 16_000;
pub const FPS: u64 = 30;

#[derive(Default)]
pub struct FrameBuffer {
    bytes: VecDeque<u8>,
    received_samples: u64,
    emitted_samples: u64,
    frame: u64,
}

impl FrameBuffer {
    #[cfg(test)]
    pub(crate) fn first_pointer(&self) -> *const u8 {
        self.bytes.as_slices().0.as_ptr()
    }
    pub fn push_owned(&mut self, bytes: Vec<u8>, max_samples: u64) -> Result<(), Error> {
        self.accept(bytes.len(), max_samples)?;
        if self.bytes.is_empty() {
            self.bytes = bytes.into();
        } else {
            self.bytes.extend(bytes);
        }
        Ok(())
    }
    fn accept(&mut self, len: usize, max_samples: u64) -> Result<(), Error> {
        if !len.is_multiple_of(2) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "PCM chunk must contain whole 16-bit samples",
            ));
        }
        let new_total = self.received_samples + (len / 2) as u64;
        if new_total > max_samples {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "audio duration limit exceeded",
            ));
        }
        self.received_samples = new_total;
        Ok(())
    }
    pub fn push(&mut self, bytes: &[u8], max_samples: u64) -> Result<(), Error> {
        self.accept(bytes.len(), max_samples)?;
        self.bytes.extend(bytes);
        Ok(())
    }

    pub fn pop(&mut self, finished: bool) -> Option<(u64, Vec<u8>)> {
        let boundary = (self.frame + 1) * SAMPLE_RATE / FPS;
        if self.received_samples < boundary && !finished {
            return None;
        }
        let end = boundary.min(self.received_samples);
        if end == self.emitted_samples {
            return None;
        }
        let start = self.emitted_samples;
        let count = ((end - start) * 2) as usize;
        let pcm = if count == self.bytes.len() {
            std::mem::take(&mut self.bytes).into()
        } else {
            self.bytes.drain(..count).collect()
        };
        self.emitted_samples = end;
        self.frame += 1;
        Some((start, pcm))
    }

    pub fn is_empty(&self) -> bool {
        self.received_samples == 0
    }
}

/// Converts directly into reusable caller storage without a temporary PCM vector.
#[cfg(any(feature = "native", test))]
pub(crate) fn normalized_samples(bytes: &VecDeque<u8>, offset: usize, output: &mut [f32]) {
    let mut bytes = bytes.iter().skip(offset);
    for value in output {
        *value = f32::from(i16::from_le_bytes([
            *bytes.next().expect("PCM range"),
            *bytes.next().expect("PCM range"),
        ])) / 32768.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn render(pcm: &[u8], chunk_samples: usize) -> Vec<(u64, Vec<u8>)> {
        let mut buffer = FrameBuffer::default();
        let mut frames = Vec::new();
        for chunk in pcm.chunks(chunk_samples * 2) {
            buffer.push(chunk, 100_000).unwrap();
            while let Some(frame) = buffer.pop(false) {
                frames.push(frame);
            }
        }
        while let Some(frame) = buffer.pop(true) {
            frames.push(frame);
        }
        assert!(buffer.pop(true).is_none());
        frames
    }
    #[test]
    fn boundaries_and_chunking_preserve_every_sample() {
        for samples in [1, 532, 533, 534, 1065, 1066, 1067, 16_000, 16_001] {
            let pcm: Vec<u8> = (0..samples * 2).map(|i| (i % 251) as u8).collect();
            let expected = render(&pcm, samples);
            for chunk in [1, 17, 533, 560, 16_000] {
                assert_eq!(render(&pcm, chunk), expected);
            }
            assert_eq!(
                expected
                    .iter()
                    .flat_map(|(_, b)| b.iter().copied())
                    .collect::<Vec<_>>(),
                pcm
            );
            assert!(expected.windows(2).all(|w| w[0].0 < w[1].0));
        }
    }
    #[test]
    fn rejects_partial_samples_and_duration_overflow() {
        let mut buffer = FrameBuffer::default();
        assert!(buffer.push(&[], 1).is_ok());
        assert!(buffer.is_empty());
        assert_eq!(
            buffer.push(&[0], 1).unwrap_err().kind(),
            ErrorKind::InvalidInput
        );
        buffer.push(&[0, 0], 1).unwrap();
        assert_eq!(
            buffer.push(&[0, 0], 1).unwrap_err().kind(),
            ErrorKind::LimitExceeded
        );
    }
}
