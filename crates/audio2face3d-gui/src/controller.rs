//! Controller-owned graph history; shared audio updates and clock snapshots stay small.
use crate::{
    core::{Clip, Result, SessionState},
    playback::{PlaybackClockSnapshot, PlaybackUpdate, Player, Snapshot},
};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

pub struct SessionController {
    clip: Clip,
    player: Arc<Mutex<Player>>,
    audio_blocks: usize,
}
impl SessionController {
    pub fn new(player: Arc<Mutex<Player>>) -> Self {
        let mut controller = Self {
            clip: Clip::default(),
            player,
            audio_blocks: 0,
        };
        controller.replace(Clip::default());
        controller
    }
    pub fn clip(&self) -> &Clip {
        &self.clip
    }

    pub fn replace(&mut self, clip: Clip) {
        // Clone only block handles before locking; graph and emotion history
        // remain owned here and are never installed in the callback's Player.
        let playback = Player::from_clip(&clip);
        *self.player.lock().unwrap() = playback;
        self.audio_blocks = clip.audio.block_count();
        self.clip = clip;
    }

    /// Apply a batch outside the audio lock, then call publish once.
    pub fn apply_event(&mut self, event: audio2face3d::types::OutputEvent) -> Result<()> {
        crate::inference::apply_event(&mut self.clip, event)
    }
    pub fn set_session(&mut self, state: SessionState) {
        self.clip.set_session(state);
    }

    pub fn publish(&mut self) -> Result<()> {
        let update = PlaybackUpdate::from_clip(&self.clip, self.audio_blocks)?;
        self.player.lock().unwrap().apply_update(update)?;
        self.audio_blocks = self.clip.audio.block_count();
        Ok(())
    }
    pub fn clock_snapshot(&self, now: Instant) -> PlaybackClockSnapshot {
        self.player.lock().unwrap().clock_snapshot(now)
    }
    pub fn snapshot(&self, now: Instant) -> Result<Snapshot> {
        self.with_view(now, |clip, clock| clock.with_clip(clip))?
    }
    /// The supplied graph/curve work always runs after the audio lock is released.
    pub fn with_view<T>(
        &self,
        now: Instant,
        view: impl FnOnce(&Clip, PlaybackClockSnapshot) -> T,
    ) -> Result<T> {
        let clock = self.clock_snapshot(now);
        if clock.clip_id != self.clip.id() {
            return Err(crate::core::Error(
                "stale playback clock for another clip".into(),
            ));
        }
        Ok(view(&self.clip, clock))
    }
}

#[cfg(test)]
mod tests;
