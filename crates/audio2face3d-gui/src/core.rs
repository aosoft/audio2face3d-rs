//! Owned media data and peak-preserving timeline queries, independent of UI/audio.
use std::collections::BTreeSet;

pub const SAMPLE_RATE: u32 = 16_000;
pub const MAX_SECONDS: f64 = 600.;
pub const MAX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct Error(pub String);
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, PartialEq)]
pub enum SessionState {
    Idle,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}
#[derive(Debug)]
pub struct CurveSample {
    pub time: f64,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub struct Clip {
    pub names: Vec<String>,
    pub frames: Vec<CurveSample>,
    pub audio: Vec<f32>,
    pub session: SessionState,
    pub revision: u64,
    curve_end: f64,
    curve_gap: bool,
    retained: usize,
}
impl Default for Clip {
    fn default() -> Self {
        Self {
            names: vec![],
            frames: vec![],
            audio: vec![],
            session: SessionState::Idle,
            revision: 0,
            curve_end: 0.,
            curve_gap: false,
            retained: 0,
        }
    }
}
impl Clip {
    pub fn running() -> Self {
        Self {
            session: SessionState::Running,
            ..Self::default()
        }
    }
    pub fn set_names(&mut self, names: Vec<String>) -> Result<()> {
        if names.is_empty()
            || names.len() > 256
            || names.iter().any(|n| n.is_empty() || n.len() > 256)
            || names.iter().collect::<BTreeSet<_>>().len() != names.len()
        {
            return Err(Error("invalid channel layout".into()));
        }
        if !self.frames.is_empty() && self.names != names {
            return Err(Error("channel layout changed within session".into()));
        }
        self.names = names;
        self.revision += 1;
        Ok(())
    }
    fn check_budget(&self, bytes: usize) -> Result<()> {
        // Leave room for Vec growth and per-frame bookkeeping.
        if self.retained.saturating_add(bytes).saturating_mul(2) > MAX_BYTES - 256 * 1024 {
            Err(Error(
                "result memory limit exceeded; partial result retained".into(),
            ))
        } else {
            Ok(())
        }
    }
    pub fn push_audio(&mut self, start: u64, samples: &[f32]) -> Result<()> {
        if start != self.audio.len() as u64
            || self.audio.len().saturating_add(samples.len())
                > SAMPLE_RATE as usize * MAX_SECONDS as usize
            || samples.iter().any(|v| !v.is_finite())
        {
            return Err(Error("invalid, noncontiguous, or over-limit audio".into()));
        }
        self.check_budget(samples.len() * 4)?;
        self.retained += samples.len() * 4;
        self.audio.extend_from_slice(samples);
        self.revision += 1;
        Ok(())
    }
    pub fn push_frame(&mut self, time: f64, values: Vec<f32>) -> Result<()> {
        if !time.is_finite()
            || !(0.0..=MAX_SECONDS).contains(&time)
            || values.len() != self.names.len()
            || values.iter().any(|v| !v.is_finite())
            || self.frames.last().is_some_and(|f| f.time >= time)
        {
            return Err(Error("invalid or out-of-order curve frame".into()));
        }
        let bytes = values.capacity() * 4 + std::mem::size_of::<CurveSample>();
        self.check_budget(bytes)?;
        self.retained += bytes;
        if self
            .frames
            .last()
            .map_or(time > 0.001, |f| time - f.time > 0.05)
        {
            self.curve_gap = true;
        }
        if !self.curve_gap {
            self.curve_end = time;
        }
        self.frames.push(CurveSample { time, values });
        self.revision += 1;
        Ok(())
    }
    pub fn duration(&self) -> f64 {
        self.audio.len() as f64 / SAMPLE_RATE as f64
    }
    pub fn ready_until(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.;
        }
        let end = if self.session == SessionState::Completed && !self.curve_gap {
            self.curve_end + 0.05
        } else {
            self.curve_end
        };
        self.duration().min(end)
    }
    pub fn values_at(&self, time: f64) -> (Vec<f32>, Vec<f32>) {
        if self.frames.is_empty() {
            return (vec![0.; self.names.len()], vec![0.; self.names.len()]);
        }
        let index = self
            .frames
            .partition_point(|f| f.time <= time)
            .saturating_sub(1);
        let first = &self.frames[index];
        let raw = first.values.clone();
        let Some(second) = self.frames.get(index + 1) else {
            return (raw.clone(), raw);
        };
        let mix = ((time - first.time) / (second.time - first.time)).clamp(0., 1.) as f32;
        (
            first
                .values
                .iter()
                .zip(&second.values)
                .map(|(a, b)| a + (b - a) * mix)
                .collect(),
            raw,
        )
    }
    /// One min/max pair per pixel bucket, retaining short peaks during zoom-out.
    pub fn envelope(&self, channel: usize, buckets: usize) -> Vec<Option<[f32; 2]>> {
        let duration = self
            .duration()
            .max(self.frames.last().map_or(0., |f| f.time))
            .max(1e-6);
        self.envelope_range(channel, buckets, 0., duration)
    }
    /// Aggregate only the requested media window without changing source samples.
    pub fn envelope_range(
        &self,
        channel: usize,
        buckets: usize,
        start: f64,
        end: f64,
    ) -> Vec<Option<[f32; 2]>> {
        let mut result = vec![None::<[f32; 2]>; buckets.min(4096)];
        if channel >= self.names.len()
            || result.is_empty()
            || !start.is_finite()
            || !end.is_finite()
            || end <= start
        {
            return result;
        }
        let first = self.frames.partition_point(|f| f.time < start);
        let last = self.frames.partition_point(|f| f.time <= end);
        for frame in &self.frames[first..last] {
            let index = (((frame.time - start) / (end - start) * result.len() as f64) as usize)
                .min(result.len() - 1);
            let value = frame.values[channel];
            match &mut result[index] {
                Some(range) => {
                    range[0] = range[0].min(value);
                    range[1] = range[1].max(value);
                }
                slot @ None => *slot = Some([value, value]),
            }
        }
        result
    }
}

pub fn demo_clip() -> Clip {
    let mut clip = Clip::default();
    clip.set_names(
        audio2face3d_gui_core::rig::CHANNELS
            .iter()
            .map(|n| n.to_string())
            .collect(),
    )
    .unwrap();
    let seconds = 6;
    let audio: Vec<_> = (0..SAMPLE_RATE * seconds)
        .map(|i| {
            let phase = i % SAMPLE_RATE;
            if phase < 160 {
                0.2 * (std::f32::consts::TAU * 880. * i as f32 / SAMPLE_RATE as f32).sin()
            } else {
                0.
            }
        })
        .collect();
    clip.push_audio(0, &audio).unwrap();
    for i in 0..seconds * 30 {
        let mut values = vec![0.; 52];
        values[17] = ((i as f32 / 30. * std::f32::consts::TAU).cos() * 0.5 + 0.5).powi(8);
        values[0] = if i % 30 < 3 {
            1. - (i % 30) as f32 / 3.
        } else {
            0.
        };
        values[7] = values[0];
        clip.push_frame(i as f64 / 30., values).unwrap();
    }
    clip.session = SessionState::Completed;
    clip
}
