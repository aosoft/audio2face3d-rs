//! Audio callback driven media clock. Wall time alone never advances playback.
use crate::core::{AudioBlock, AudioTrack, Clip, ClipId, Error, Result, SAMPLE_RATE, SessionState};
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlaybackState {
    Paused,
    Playing,
    Buffering,
    Ended,
}
#[derive(Clone, Copy, Debug)]
pub enum Command {
    Play,
    Pause,
    Seek(f64),
    SetLoop(bool),
}
pub struct Snapshot {
    pub time: f64,
    pub duration: f64,
    pub ready_until: f64,
    pub state: PlaybackState,
    pub values: Vec<f32>,
    pub raw_values: Vec<f32>,
    pub underruns: u64,
    pub looping: bool,
}

/// A small, allocation-free audio clock snapshot. Curve lookup belongs to the controller.
#[derive(Clone, Copy, Debug)]
pub struct PlaybackClockSnapshot {
    pub clip_id: ClipId,
    pub time: f64,
    pub duration: f64,
    pub ready_until: f64,
    pub state: PlaybackState,
    pub underruns: u64,
    pub looping: bool,
}
impl PlaybackClockSnapshot {
    pub fn with_clip(self, clip: &Clip) -> Result<Snapshot> {
        if self.clip_id != clip.id() {
            return Err(Error("stale playback clock for another clip".into()));
        }
        let (values, raw_values) = clip.values_at(self.time);
        Ok(Snapshot {
            time: self.time,
            duration: self.duration,
            ready_until: self.ready_until,
            state: self.state,
            values,
            raw_values,
            underruns: self.underruns,
            looping: self.looping,
        })
    }
}

#[derive(Clone, Copy)]
struct PlaybackMedia {
    clip_id: ClipId,
    revision: u64,
    sample_count: usize,
    ready_until: f64,
    complete: bool,
    job_ready: bool,
    unavailable: bool,
    has_frames: bool,
}
impl PlaybackMedia {
    fn from_clip(clip: &Clip) -> Self {
        Self {
            clip_id: clip.id(),
            revision: clip.revision(),
            sample_count: clip.audio().len(),
            ready_until: clip.ready_until(),
            complete: clip.session() == &SessionState::Completed,
            job_ready: clip.session() == &SessionState::Completed,
            unavailable: matches!(
                clip.session(),
                SessionState::Failed(_) | SessionState::Cancelled
            ),
            has_frames: !clip.frames().is_empty(),
        }
    }
    fn duration(self) -> f64 {
        self.sample_count as f64 / SAMPLE_RATE as f64
    }
}

struct DetachedPlayback {
    media: PlaybackMedia,
    audio: AudioTrack,
}
enum PlaybackSource {
    Owned(Clip),
    Controller(DetachedPlayback),
}

/// A prepared update: Arc handles are collected before taking the audio lock.
pub(crate) struct PlaybackUpdate {
    media: PlaybackMedia,
    first_block: usize,
    blocks: Vec<std::sync::Arc<AudioBlock>>,
}
impl PlaybackUpdate {
    pub(crate) fn from_clip(clip: &Clip, first_block: usize) -> Result<Self> {
        Ok(Self {
            media: PlaybackMedia::from_clip(clip),
            first_block,
            blocks: clip.audio().shared_from(first_block)?,
        })
    }
    pub(crate) fn session(&mut self, revision: u64, ready: bool, unavailable: bool) {
        self.media.revision = revision;
        self.media.job_ready = ready;
        self.media.unavailable |= unavailable;
    }
}
struct Anchor {
    at: Instant,
    media: f64,
    frames: usize,
    rate: u32,
}
/// Playback state changes through commands and validated media updates.
///
/// ```compile_fail
/// let mut player = audio2face3d_gui::playback::Player::default();
/// player.state = audio2face3d_gui::playback::PlaybackState::Playing;
/// ```
pub struct Player {
    source: PlaybackSource,
    state: PlaybackState,
    looping: bool,
    streaming: bool,
    /// Media to accumulate before streaming starts or resumes after an underrun.
    stream_buffer: Duration,
    underruns: u64,
    submitted: f64,
    position: f64,
    anchors: VecDeque<Anchor>,
    audio_cursor: usize,
}
impl Default for Player {
    fn default() -> Self {
        Self {
            source: PlaybackSource::Owned(Clip::default()),
            state: PlaybackState::Paused,
            looping: false,
            streaming: false,
            stream_buffer: Duration::from_millis(100),
            underruns: 0,
            submitted: 0.,
            position: 0.,
            anchors: VecDeque::with_capacity(64),
            audio_cursor: 0,
        }
    }
}
impl Player {
    pub fn state(&self) -> PlaybackState {
        self.state
    }
    pub fn underruns(&self) -> u64 {
        self.underruns
    }
    pub fn looping(&self) -> bool {
        self.looping
    }
    pub fn streaming(&self) -> bool {
        self.streaming
    }
    pub fn stream_buffer(&self) -> Duration {
        self.stream_buffer
    }
    pub fn set_streaming(&mut self, enabled: bool, buffer: Duration) -> Result<()> {
        if buffer.as_secs_f64() > crate::core::MAX_SECONDS {
            return Err(Error("stream buffer exceeds media limit".into()));
        }
        self.streaming = enabled;
        self.stream_buffer = buffer;
        Ok(())
    }
    /// Read-only compatibility access for the player-owned path.
    pub fn clip(&self) -> Option<&Clip> {
        match &self.source {
            PlaybackSource::Owned(clip) => Some(clip),
            _ => None,
        }
    }
    fn owned_clip(&mut self) -> Result<&mut Clip> {
        match &mut self.source {
            PlaybackSource::Owned(clip) => Ok(clip),
            _ => Err(Error(
                "update controller-owned media through SessionController".into(),
            )),
        }
    }
    pub fn append_audio(&mut self, start: u64, samples: &[f32]) -> Result<()> {
        self.owned_clip()?.push_audio(start, samples)
    }
    pub fn append_frame(&mut self, time: f64, values: Vec<f32>) -> Result<()> {
        self.owned_clip()?.push_frame(time, values)
    }
    pub fn finish_collection(&mut self) -> Result<()> {
        self.owned_clip()?.finish_collection()
    }
    /// Prepare shared audio and constant-size media metadata outside the audio lock.
    pub(crate) fn from_clip(clip: &Clip) -> Self {
        Self {
            source: PlaybackSource::Controller(DetachedPlayback {
                media: PlaybackMedia::from_clip(clip),
                audio: clip.audio().clone(),
            }),
            ..Self::default()
        }
    }
    pub(crate) fn for_controller(clip: &Clip, ready: bool, unavailable: bool) -> Self {
        let mut player = Self::from_clip(clip);
        if let PlaybackSource::Controller(playback) = &mut player.source {
            playback.media.job_ready = ready;
            playback.media.unavailable |= unavailable;
        }
        player
    }
    pub(crate) fn apply_update(&mut self, update: PlaybackUpdate) -> Result<()> {
        let PlaybackSource::Controller(playback) = &mut self.source else {
            return Err(Error("player has no controller-owned clip".into()));
        };
        if playback.media.clip_id != update.media.clip_id
            || update.media.revision < playback.media.revision
            || update.first_block != playback.audio.block_count()
            || playback
                .audio
                .len()
                .saturating_add(update.blocks.iter().map(|block| block.len()).sum::<usize>())
                != update.media.sample_count
        {
            return Err(Error("stale or noncontiguous playback update".into()));
        }
        playback.audio.append_shared(update.blocks)?;
        let unavailable = playback.media.unavailable || update.media.unavailable;
        playback.media = update.media;
        playback.media.unavailable = unavailable;
        if unavailable {
            self.command(Command::Pause, Instant::now())?;
        }
        Ok(())
    }
    fn media(&self) -> PlaybackMedia {
        match &self.source {
            PlaybackSource::Owned(clip) => PlaybackMedia::from_clip(clip),
            PlaybackSource::Controller(playback) => playback.media,
        }
    }
    #[cfg(test)]
    pub(crate) fn audio_track(&self) -> &AudioTrack {
        match &self.source {
            PlaybackSource::Owned(clip) => clip.audio(),
            PlaybackSource::Controller(playback) => &playback.audio,
        }
    }
    pub fn replace(&mut self, clip: Clip) {
        *self = Self {
            source: PlaybackSource::Owned(clip),
            ..Default::default()
        };
    }
    pub fn audible_position(&self, now: Instant) -> f64 {
        self.anchors
            .iter()
            .rev()
            .find(|a| a.at <= now)
            .map_or(self.position, |a| {
                a.media
                    + now
                        .duration_since(a.at)
                        .as_secs_f64()
                        .min(a.frames as f64 / a.rate as f64)
            })
    }
    pub fn command(&mut self, command: Command, now: Instant) -> Result<()> {
        let media = self.media();
        match command {
            Command::Play => {
                if media.unavailable {
                    return Err(Error(
                        "failed/cancelled result: seek to inspect partial data".into(),
                    ));
                }
                if !self.streaming && !media.job_ready {
                    return Err(Error("wait for successful inference completion".into()));
                }
                if media.sample_count == 0 || !media.has_frames {
                    return Err(Error("no playable result".into()));
                }
                if self.state == PlaybackState::Ended {
                    self.submitted = 0.;
                    self.position = 0.;
                    self.anchors.clear();
                }
                self.state = if self.streaming
                    && !media.complete
                    && media.ready_until - self.submitted < self.stream_buffer.as_secs_f64()
                {
                    PlaybackState::Buffering
                } else {
                    PlaybackState::Playing
                };
            }
            Command::Pause => {
                self.position = self.audible_position(now);
                self.submitted = self.position;
                self.anchors.clear();
                self.state = PlaybackState::Paused;
            }
            Command::Seek(time) => {
                if !time.is_finite() || time < 0. || time > media.ready_until {
                    return Err(Error("seek outside received media".into()));
                }
                self.submitted = time;
                self.position = time;
                self.anchors.clear();
                if self.state == PlaybackState::Ended {
                    self.state = PlaybackState::Paused;
                }
            }
            Command::SetLoop(value) => self.looping = value,
        }
        Ok(())
    }
    pub fn snapshot(&mut self, now: Instant) -> Snapshot {
        let clock = self.clock_snapshot(now);
        // Compatibility path for an owned Clip. Detached users obtain curves
        // through PlaybackClockSnapshot::with_clip after releasing the lock.
        let (values, raw_values) = self
            .clip()
            .map_or_else(|| (vec![], vec![]), |clip| clip.values_at(clock.time));
        Snapshot {
            time: clock.time,
            duration: clock.duration,
            ready_until: clock.ready_until,
            state: clock.state,
            values,
            raw_values,
            underruns: clock.underruns,
            looping: clock.looping,
        }
    }
    pub fn clock_snapshot(&mut self, now: Instant) -> PlaybackClockSnapshot {
        let media = self.media();
        let time = self.audible_position(now).min(media.duration());
        if media.complete
            && !self.looping
            && time + 1e-8 >= media.duration()
            && self.state == PlaybackState::Playing
        {
            self.state = PlaybackState::Ended;
        }
        PlaybackClockSnapshot {
            clip_id: media.clip_id,
            time,
            duration: media.duration(),
            ready_until: media.ready_until,
            state: self.state,
            underruns: self.underruns,
            looping: self.looping,
        }
    }
    /// Host audio interface. `audible_at` is when the first output sample is heard.
    /// Output is mono duplicated by the host as needed; no allocation per sample.
    pub fn render_audio(
        &mut self,
        frames: usize,
        rate: u32,
        audible_at: Instant,
        mut write: impl FnMut(usize, f32),
    ) {
        if rate < SAMPLE_RATE
            || !matches!(
                self.state,
                PlaybackState::Playing | PlaybackState::Buffering
            )
        {
            for i in 0..frames {
                write(i, 0.);
            }
            return;
        }
        let media = self.media();
        let ready = media.ready_until;
        if media.unavailable {
            self.state = PlaybackState::Paused;
            for i in 0..frames {
                write(i, 0.);
            }
            return;
        }
        let complete = media.complete;
        if self.state == PlaybackState::Buffering {
            if ready - self.submitted >= self.stream_buffer.as_secs_f64()
                || (complete && ready > self.submitted)
            {
                self.state = PlaybackState::Playing;
            } else {
                for i in 0..frames {
                    write(i, 0.);
                }
                return;
            }
        }
        let mut segment = 0;
        let mut segment_media = self.submitted;
        for i in 0..frames {
            if self.submitted + 1e-9 >= ready {
                self.anchor(
                    audible_at + Duration::from_secs_f64(segment as f64 / rate as f64),
                    segment_media,
                    i - segment,
                    rate,
                );
                if complete && self.looping && ready > 0. && ready + 1e-6 >= media.duration() {
                    self.submitted = 0.;
                    segment = i;
                    segment_media = 0.;
                } else {
                    if !complete {
                        self.underruns += 1;
                        self.state = PlaybackState::Buffering;
                    }
                    for j in i..frames {
                        write(j, 0.);
                    }
                    return;
                }
            }
            let sample = self.submitted * SAMPLE_RATE as f64;
            let index = sample.floor() as usize;
            let audio = match &self.source {
                PlaybackSource::Owned(clip) => clip.audio(),
                PlaybackSource::Controller(playback) => &playback.audio,
            };
            let a = audio
                .sample_at(index, &mut self.audio_cursor)
                .copied()
                .unwrap_or(0.);
            let b = audio
                .sample_at(index + 1, &mut self.audio_cursor)
                .copied()
                .unwrap_or(a);
            write(i, a + (b - a) * sample.fract() as f32);
            self.submitted += 1. / rate as f64;
        }
        self.anchor(
            audible_at + Duration::from_secs_f64(segment as f64 / rate as f64),
            segment_media,
            frames - segment,
            rate,
        );
    }
    fn anchor(&mut self, at: Instant, media: f64, frames: usize, rate: u32) {
        if frames == 0 {
            return;
        }
        if self.anchors.len() == 64 {
            self.anchors.pop_front();
        }
        self.anchors.push_back(Anchor {
            at,
            media,
            frames,
            rate,
        });
    }
}
