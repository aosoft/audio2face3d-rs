//! Streaming, band-limited PCM conversion with a bounded FIR history.
use super::Backend;
use crate::proto::{a2f::AudioWithEmotion, animation::AnimationData};
use std::collections::VecDeque;
use tokio_util::sync::CancellationToken;
use tonic::Status;

const RADIUS: i64 = 32;

pub struct Resampler {
    rate: u64,
    samples: VecDeque<f64>,
    base: u64,
    received: u64,
    emitted: u64,
    limit: u64,
}
impl Resampler {
    pub fn new(rate: u32, seconds: u32) -> Self {
        Self {
            rate: u64::from(rate),
            samples: VecDeque::new(),
            base: 0,
            received: 0,
            emitted: 0,
            limit: u64::from(rate) * u64::from(seconds),
        }
    }
    pub fn push(&mut self, pcm: &[u8]) -> Result<Vec<u8>, Status> {
        if !pcm.len().is_multiple_of(2) {
            return Err(Status::invalid_argument(
                "PCM chunk must contain whole 16-bit samples",
            ));
        }
        let count = (pcm.len() / 2) as u64;
        if count > self.limit - self.received {
            return Err(Status::resource_exhausted("audio duration limit exceeded"));
        }
        self.received += count;
        if self.rate == 16_000 {
            return Ok(pcm.to_vec());
        }
        self.samples.extend(
            pcm.chunks_exact(2)
                .map(|v| f64::from(i16::from_le_bytes([v[0], v[1]]))),
        );
        Ok(self.render(false))
    }
    pub fn finish(&mut self) -> Vec<u8> {
        if self.rate == 16_000 {
            Vec::new()
        } else {
            self.render(true)
        }
    }
    fn render(&mut self, finished: bool) -> Vec<u8> {
        let target = (self.received * 16_000).div_ceil(self.rate);
        let cutoff = 0.94 * 8_000.0 / self.rate as f64;
        let mut pcm = Vec::new();
        while self.emitted < target {
            let position = (self.emitted * self.rate) as f64 / 16_000.0;
            let center = position.floor() as i64;
            if !finished && center + RADIUS >= self.received as i64 {
                break;
            }
            let mut value = 0.0;
            let mut sum = 0.0;
            for index in center - RADIUS + 1..=center + RADIUS {
                let distance = index as f64 - position;
                if distance.abs() >= RADIUS as f64 {
                    continue;
                }
                let x = 2.0 * cutoff * distance;
                let sinc = if x.abs() < 1e-12 {
                    1.0
                } else {
                    (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
                };
                let window = 0.5 + 0.5 * (std::f64::consts::PI * distance / RADIUS as f64).cos();
                let weight = 2.0 * cutoff * sinc * window;
                let source = index.clamp(0, self.received as i64 - 1) as u64;
                value += self.samples[(source - self.base) as usize] * weight;
                sum += weight;
            }
            let sample = (value / sum).round().clamp(-32768.0, 32767.0) as i16;
            pcm.extend_from_slice(&sample.to_le_bytes());
            self.emitted += 1;
            let keep = ((self.emitted * self.rate / 16_000) as i64 - RADIUS).max(0) as u64;
            let drop = keep.min(self.received).saturating_sub(self.base) as usize;
            self.samples.drain(..drop);
            self.base += drop as u64;
        }
        pcm
    }
}

pub struct ResamplingBackend {
    inner: Box<dyn Backend>,
    resampler: Resampler,
}
impl ResamplingBackend {
    pub fn new(inner: Box<dyn Backend>, rate: u32, seconds: u32) -> Self {
        Self {
            inner,
            resampler: Resampler::new(rate, seconds),
        }
    }
}
#[tonic::async_trait]
impl Backend for ResamplingBackend {
    fn push(&mut self, mut input: AudioWithEmotion) -> Result<(), Status> {
        input.audio_buffer = self.resampler.push(&input.audio_buffer)?;
        self.inner.push(input)
    }
    async fn next_frame(
        &mut self,
        cancel: &CancellationToken,
    ) -> Result<Option<AnimationData>, Status> {
        self.inner.next_frame(cancel).await
    }
    fn finish(&mut self) -> Result<(), Status> {
        let tail = self.resampler.finish();
        if !tail.is_empty() {
            self.inner.push(AudioWithEmotion {
                audio_buffer: tail,
                emotions: vec![],
            })?;
        }
        self.inner.finish()
    }
    async fn close(&mut self) -> Result<(), Status> {
        self.inner.close().await
    }
    fn success_message(&self) -> &'static str {
        self.inner.success_message()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn render(rate: u32, pcm: &[u8], chunk: usize) -> Vec<u8> {
        let mut r = Resampler::new(rate, 2);
        let mut output = Vec::new();
        for v in pcm.chunks(chunk * 2) {
            output.extend(r.push(v).unwrap());
        }
        output.extend(r.finish());
        assert!(r.finish().is_empty());
        output
    }
    #[test]
    fn partition_invariant_length_dc_and_passthrough() {
        for rate in [16_000, 44_100, 48_000] {
            for count in [1, 17, 533, rate as usize + 1] {
                let pcm = vec![1000i16; count]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>();
                let expected = render(rate, &pcm, count);
                assert_eq!(expected.len() / 2, (count * 16000).div_ceil(rate as usize));
                assert!(
                    expected
                        .chunks_exact(2)
                        .all(|v| i16::from_le_bytes([v[0], v[1]]) == 1000)
                );
                for chunk in [1, 37, 560, 4410] {
                    assert_eq!(render(rate, &pcm, chunk), expected);
                }
                if rate == 16000 {
                    assert_eq!(expected, pcm);
                }
            }
        }
    }
    #[test]
    fn rejects_alias_band() {
        for rate in [44100, 48000] {
            let energy = |frequency: f64| {
                let pcm = (0..rate)
                    .flat_map(|i| {
                        ((10000.0
                            * (2.0 * std::f64::consts::PI * frequency * i as f64 / rate as f64)
                                .sin()) as i16)
                            .to_le_bytes()
                    })
                    .collect::<Vec<_>>();
                let output = render(rate, &pcm, 560);
                output
                    .chunks_exact(2)
                    .skip(100)
                    .take(15800)
                    .map(|v| f64::from(i16::from_le_bytes([v[0], v[1]])).powi(2))
                    .sum::<f64>()
            };
            assert!(energy(12000.0) / energy(1000.0) < 0.001);
        }
    }
}
