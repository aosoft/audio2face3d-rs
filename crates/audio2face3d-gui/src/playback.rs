//! Audio callback driven media clock. Wall time alone never advances playback.
use crate::core::{Clip, Error, Result, SAMPLE_RATE, SessionState};
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
struct Anchor {
    at: Instant,
    media: f64,
    frames: usize,
    rate: u32,
}
pub struct Player {
    pub clip: Clip,
    pub state: PlaybackState,
    pub looping: bool,
    pub streaming: bool,
    pub underruns: u64,
    submitted: f64,
    position: f64,
    anchors: VecDeque<Anchor>,
}
impl Default for Player {
    fn default() -> Self {
        Self {
            clip: Clip::default(),
            state: PlaybackState::Paused,
            looping: false,
            streaming: false,
            underruns: 0,
            submitted: 0.,
            position: 0.,
            anchors: VecDeque::with_capacity(64),
        }
    }
}
impl Player {
    pub fn replace(&mut self, clip: Clip) {
        *self = Self {
            clip,
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
        match command {
            Command::Play => {
                if matches!(
                    self.clip.session,
                    SessionState::Failed(_) | SessionState::Cancelled
                ) {
                    return Err(Error(
                        "failed/cancelled result: seek to inspect partial data".into(),
                    ));
                }
                if !self.streaming && self.clip.session != SessionState::Completed {
                    return Err(Error("wait for successful inference completion".into()));
                }
                if self.clip.audio.is_empty() || self.clip.frames.is_empty() {
                    return Err(Error("no playable result".into()));
                }
                if self.state == PlaybackState::Ended {
                    self.submitted = 0.;
                    self.position = 0.;
                    self.anchors.clear();
                }
                self.state = if self.streaming
                    && self.clip.session != SessionState::Completed
                    && self.clip.ready_until() - self.submitted < 0.1
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
                if !time.is_finite() || time < 0. || time > self.clip.ready_until() {
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
        let time = self.audible_position(now).min(self.clip.duration());
        if self.clip.session == SessionState::Completed
            && !self.looping
            && time + 1e-8 >= self.clip.duration()
            && self.state == PlaybackState::Playing
        {
            self.state = PlaybackState::Ended;
        }
        let (values, raw_values) = self.clip.values_at(time);
        Snapshot {
            time,
            duration: self.clip.duration(),
            ready_until: self.clip.ready_until(),
            state: self.state,
            values,
            raw_values,
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
        let ready = self.clip.ready_until();
        if matches!(
            self.clip.session,
            SessionState::Failed(_) | SessionState::Cancelled
        ) {
            self.state = PlaybackState::Paused;
            for i in 0..frames {
                write(i, 0.);
            }
            return;
        }
        let complete = self.clip.session == SessionState::Completed;
        if self.state == PlaybackState::Buffering {
            if ready - self.submitted >= 0.1 || (complete && ready > self.submitted) {
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
                if complete && self.looping && ready > 0. && ready + 1e-6 >= self.clip.duration() {
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
            let a = self.clip.audio.get(index).copied().unwrap_or(0.);
            let b = self.clip.audio.get(index + 1).copied().unwrap_or(a);
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
