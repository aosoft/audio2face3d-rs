//! Owned media data and peak-preserving timeline queries, independent of UI/audio.
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

mod audio_track;
pub(crate) use audio_track::AudioBlock;
pub use audio_track::{AudioTrack, MAX_AUDIO_BLOCKS};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ClipId(u64);

fn next_clip_id() -> ClipId {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    ClipId(
        NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .expect("clip identity exhausted"),
    )
}

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
    time: f64,
    values: Vec<f32>,
}
impl CurveSample {
    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClipRevision {
    pub layout: u64,
    pub audio: u64,
    pub curves: u64,
    pub emotions: u64,
    pub collection: u64,
}
enum Change {
    Layout,
    Audio,
    Curves,
    #[cfg(feature = "emotion")]
    Emotions,
    Collection,
}

#[derive(Debug)]
/// Validated media. Mutation is restricted to the collection methods.
///
/// ```compile_fail
/// let mut clip = audio2face3d_gui::core::Clip::default();
/// clip.names.push("unvalidated".into());
/// ```
/// ```compile_fail
/// fn corrupt(clip: &audio2face3d_gui::core::Clip) {
///     clip.frames()[0].values.clear();
/// }
/// ```
pub struct Clip {
    names: Vec<String>,
    frames: Vec<CurveSample>,
    audio: AudioTrack,
    #[cfg(feature = "emotion")]
    emotions: Vec<audio2face3d::types::EmotionKeyframe>,
    session: SessionState,
    revision: u64,
    revisions: ClipRevision,
    id: ClipId,
    curve_end: f64,
    curve_gap: bool,
    retained: usize,
    names_bytes: usize,
}
impl Default for Clip {
    fn default() -> Self {
        Self {
            names: vec![],
            frames: vec![],
            audio: AudioTrack::default(),
            #[cfg(feature = "emotion")]
            emotions: vec![],
            session: SessionState::Idle,
            revision: 0,
            revisions: ClipRevision::default(),
            id: next_clip_id(),
            curve_end: 0.,
            curve_gap: false,
            retained: 0,
            names_bytes: 0,
        }
    }
}
impl Clip {
    pub fn audio(&self) -> &AudioTrack {
        &self.audio
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn revisions(&self) -> ClipRevision {
        self.revisions
    }
    #[cfg(feature = "emotion")]
    pub fn emotions(&self) -> &[audio2face3d::types::EmotionKeyframe] {
        &self.emotions
    }
    fn ensure_receiving(&self) -> Result<()> {
        if matches!(self.session, SessionState::Idle | SessionState::Running) {
            Ok(())
        } else {
            Err(Error(
                "collection already ended; replace the clip before adding media".into(),
            ))
        }
    }
    fn next_revision(&self, change: Change) -> Result<(u64, ClipRevision)> {
        let error = || Error("clip revision exhausted".into());
        let sequence = self.revision.checked_add(1).ok_or_else(error)?;
        let mut revisions = self.revisions;
        let counter = match change {
            Change::Layout => &mut revisions.layout,
            Change::Audio => &mut revisions.audio,
            Change::Curves => &mut revisions.curves,
            #[cfg(feature = "emotion")]
            Change::Emotions => &mut revisions.emotions,
            Change::Collection => &mut revisions.collection,
        };
        *counter = counter.checked_add(1).ok_or_else(error)?;
        Ok((sequence, revisions))
    }
    fn commit_revision(&mut self, revision: (u64, ClipRevision)) {
        (self.revision, self.revisions) = revision;
    }
    pub fn id(&self) -> ClipId {
        self.id
    }
    pub fn names(&self) -> &[String] {
        &self.names
    }
    pub fn frames(&self) -> &[CurveSample] {
        &self.frames
    }
    pub fn session(&self) -> &SessionState {
        &self.session
    }
    /// Compatibility setter for collection state. Job results belong to the controller.
    pub fn set_session(&mut self, value: SessionState) -> Result<()> {
        if value == SessionState::Completed {
            return self.finish_collection();
        }
        if self.session == value {
            return Ok(());
        }
        self.ensure_receiving()?;
        if value == SessionState::Idle {
            return Err(Error("replace the clip to reset collection".into()));
        }
        let revision = self.next_revision(Change::Collection)?;
        self.session = value;
        self.commit_revision(revision);
        Ok(())
    }
    pub fn finish_collection(&mut self) -> Result<()> {
        if self.session == SessionState::Completed {
            return Ok(());
        }
        self.ensure_receiving()?;
        if self.curve_gap
            || (self.duration() > 0. && self.frames.is_empty())
            || self.curve_end + 0.05 + 1e-6 < self.duration()
        {
            return Err(Error("incomplete curve coverage".into()));
        }
        let revision = self.next_revision(Change::Collection)?;
        self.session = SessionState::Completed;
        self.commit_revision(revision);
        Ok(())
    }
    #[cfg(feature = "emotion")]
    pub fn push_emotions(
        &mut self,
        frames: Vec<audio2face3d::types::EmotionKeyframe>,
    ) -> Result<()> {
        self.ensure_receiving()?;
        let mut previous = self.emotions.last().map_or(0., |f| f.time().as_seconds());
        let mut bytes = 0usize;
        for frame in &frames {
            let time = frame.time().as_seconds();
            if time < previous || !(0.0..=MAX_SECONDS).contains(&time) {
                return Err(Error("invalid or out-of-order emotion frame".into()));
            }
            previous = time;
            // Conservative allowance for the sparse map's allocations and tree nodes.
            bytes = bytes.saturating_add(std::mem::size_of_val(frame));
            for name in frame.values().keys() {
                bytes = bytes.saturating_add(name.capacity()).saturating_add(128);
            }
        }
        self.check_budget(bytes)?;
        let revision = self.next_revision(Change::Emotions)?;
        self.retained += bytes;
        self.emotions.extend(frames);
        self.commit_revision(revision);
        Ok(())
    }
    #[cfg(feature = "emotion")]
    pub fn emotion_at(&self, time: f64) -> Option<&audio2face3d::types::EmotionValues> {
        let index = self
            .emotions
            .partition_point(|frame| frame.time().as_seconds() <= time);
        index.checked_sub(1).map(|i| self.emotions[i].values())
    }

    pub fn running() -> Self {
        Self {
            session: SessionState::Running,
            ..Self::default()
        }
    }
    pub fn set_names(&mut self, names: Vec<String>) -> Result<()> {
        self.ensure_receiving()?;
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
        if self.names == names {
            return Ok(());
        }
        let bytes = names.iter().fold(
            names
                .capacity()
                .saturating_mul(std::mem::size_of::<String>()),
            |total, name| total.saturating_add(name.capacity()),
        );
        let retained = self.retained - self.names_bytes;
        if retained.saturating_add(bytes).saturating_mul(2) > MAX_BYTES - 256 * 1024 {
            return Err(Error(
                "result memory limit exceeded; partial result retained".into(),
            ));
        }
        let revision = self.next_revision(Change::Layout)?;
        self.names = names;
        self.names_bytes = bytes;
        self.retained = retained + bytes;
        self.commit_revision(revision);
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
        let bytes = self.validate_audio(start, samples)?;
        self.store_audio(samples.to_vec().into_boxed_slice(), bytes)
    }
    pub fn push_audio_owned(&mut self, start: u64, samples: Vec<f32>) -> Result<()> {
        let bytes = self.validate_audio(start, &samples)?;
        self.store_audio(samples.into_boxed_slice(), bytes)
    }
    fn validate_audio(&self, start: u64, samples: &[f32]) -> Result<usize> {
        self.ensure_receiving()?;
        if start != self.audio.len() as u64
            || self.audio.len().saturating_add(samples.len())
                > SAMPLE_RATE as usize * MAX_SECONDS as usize
            || samples.iter().any(|v| !v.is_finite())
        {
            return Err(Error("invalid, noncontiguous, or over-limit audio".into()));
        }
        if !samples.is_empty() && self.audio.block_count() >= MAX_AUDIO_BLOCKS {
            return Err(Error(
                "audio block limit exceeded; partial result retained".into(),
            ));
        }
        let bytes = AudioTrack::block_cost(samples.len());
        self.check_budget(bytes)?;
        Ok(bytes)
    }
    fn store_audio(&mut self, samples: Box<[f32]>, bytes: usize) -> Result<()> {
        let revision = self.next_revision(Change::Audio)?;
        self.retained += bytes;
        self.audio.push(samples);
        self.commit_revision(revision);
        Ok(())
    }
    pub fn push_frame(&mut self, time: f64, values: Vec<f32>) -> Result<()> {
        self.ensure_receiving()?;
        if self.names.is_empty()
            || !time.is_finite()
            || !(0.0..=MAX_SECONDS).contains(&time)
            || values.len() != self.names.len()
            || values.iter().any(|v| !v.is_finite())
            || self.frames.last().is_some_and(|f| f.time >= time)
        {
            return Err(Error("invalid or out-of-order curve frame".into()));
        }
        let bytes = values.capacity() * 4 + std::mem::size_of::<CurveSample>();
        self.check_budget(bytes)?;
        let revision = self.next_revision(Change::Curves)?;
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
        self.commit_revision(revision);
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
    clip.set_names(crate::rig::CHANNELS.iter().map(|n| n.to_string()).collect())
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
    clip.finish_collection().unwrap();
    clip
}

#[cfg(test)]
mod tests;
