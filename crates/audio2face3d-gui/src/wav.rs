//! Streaming WAV decode/downmix and windowed-sinc conversion to mono PCM16/16 kHz.
use crate::core::{Error, MAX_SECONDS, Result, SAMPLE_RATE};
use std::{collections::VecDeque, path::Path};
pub fn load(path: &Path) -> Result<Vec<u8>> {
    let mut reader = hound::WavReader::open(path).map_err(|e| Error(e.to_string()))?;
    let spec = reader.spec();
    let frames = reader.duration() as usize;
    if !(1..=2).contains(&spec.channels)
        || !(8000..=192000).contains(&spec.sample_rate)
        || frames == 0
        || frames as f64 / spec.sample_rate as f64 > MAX_SECONDS
    {
        return Err(Error(
            "WAV requires mono/stereo, 8..192 kHz, and 0..600 seconds".into(),
        ));
    }
    if !reader.len().is_multiple_of(spec.channels as u32) {
        return Err(Error("incomplete WAV channel frame".into()));
    }
    let samples: Box<dyn Iterator<Item = Result<f32>> + '_> =
        match (spec.sample_format, spec.bits_per_sample) {
            (hound::SampleFormat::Float, 32) => Box::new(
                reader
                    .samples::<f32>()
                    .map(|v| v.map_err(|e| Error(e.to_string()))),
            ),
            (hound::SampleFormat::Int, bits @ (16 | 24 | 32)) => {
                Box::new(reader.samples::<i32>().map(move |v| {
                    v.map(|v| v as f32 / (2f32).powi(bits as i32 - 1))
                        .map_err(|e| Error(e.to_string()))
                }))
            }
            _ => return Err(Error("WAV requires PCM16/24/32 or IEEE float32".into())),
        };
    let mut samples = samples;
    let mut mono = || -> Result<f32> {
        let mut sum = 0.;
        for _ in 0..spec.channels {
            let value = samples
                .next()
                .ok_or_else(|| Error("truncated WAV".into()))??;
            if !value.is_finite() {
                return Err(Error("nonfinite WAV sample".into()));
            }
            sum += value / spec.channels as f32;
        }
        Ok(sum)
    };
    let output_frames = (frames as u64 * SAMPLE_RATE as u64 / spec.sample_rate as u64) as usize;
    let mut output = Vec::with_capacity(output_frames * 2);
    let mut ring = VecDeque::<f32>::with_capacity(130);
    let mut base = 0usize;
    let mut read = 0usize;
    let cutoff = (SAMPLE_RATE as f64 / spec.sample_rate as f64).min(1.) * 0.94;
    for frame in 0..output_frames {
        let value = if spec.sample_rate == SAMPLE_RATE {
            mono()?
        } else {
            let position = frame as f64 * spec.sample_rate as f64 / SAMPLE_RATE as f64;
            let first = (position.floor() as isize - 63).max(0) as usize;
            let last = (position.floor() as usize + 64).min(frames - 1);
            while base < first && !ring.is_empty() {
                ring.pop_front();
                base += 1;
            }
            while read <= last {
                let value = mono()?;
                if read >= first {
                    ring.push_back(value);
                } else {
                    base = read + 1;
                }
                read += 1;
            }
            let mut sum = 0.;
            let mut norm = 0.;
            for source in first..=last {
                let distance = position - source as f64;
                let x = std::f64::consts::PI * distance * cutoff;
                let sinc = if x.abs() < 1e-12 { 1. } else { x.sin() / x };
                let coefficient =
                    cutoff * sinc * 0.5 * (1. + (std::f64::consts::PI * distance / 64.).cos());
                sum += ring[source - base] as f64 * coefficient;
                norm += coefficient;
            }
            (sum / norm) as f32
        };
        let value = (value.clamp(-1., 1.) * 32768.)
            .round()
            .clamp(-32768., 32767.) as i16;
        output.extend(value.to_le_bytes());
    }
    // Validate remaining tail samples too, including a malformed trailing frame.
    if spec.sample_rate == SAMPLE_RATE {
        read = output_frames;
    }
    while read < frames {
        mono()?;
        read += 1;
    }
    Ok(output)
}
