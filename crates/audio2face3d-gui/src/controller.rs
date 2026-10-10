//! Controller-owned graph history; shared audio updates and clock snapshots stay small.
use crate::{
    core::{Clip, ClipId, Error, Result, SessionState},
    playback::{PlaybackClockSnapshot, PlaybackUpdate, Player, Snapshot},
};
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionControllerState {
    Idle,
    Initializing,
    Receiving,
    Finalizing,
    Ready,
    Cancelled,
    Failed(String),
}

pub struct SessionController {
    clip: Clip,
    player: Arc<Mutex<Player>>,
    audio_blocks: usize,
    state: SessionControllerState,
    revision: u64,
}
impl SessionController {
    pub fn new(player: Arc<Mutex<Player>>) -> Self {
        let mut controller = Self {
            clip: Clip::default(),
            player,
            audio_blocks: 0,
            state: SessionControllerState::Idle,
            revision: 0,
        };
        controller.replace(Clip::default());
        controller
    }
    pub fn clip(&self) -> &Clip {
        &self.clip
    }
    pub fn state(&self) -> &SessionControllerState {
        &self.state
    }
    pub fn is_unavailable(&self) -> bool {
        matches!(
            self.state,
            SessionControllerState::Failed(_) | SessionControllerState::Cancelled
        )
    }
    pub fn begin(&mut self) {
        self.replace(Clip::running());
        self.state = SessionControllerState::Initializing;
    }

    pub fn replace(&mut self, clip: Clip) {
        let state = match clip.session() {
            SessionState::Idle => SessionControllerState::Idle,
            SessionState::Running => SessionControllerState::Receiving,
            SessionState::Completed => SessionControllerState::Ready,
            SessionState::Failed(error) => SessionControllerState::Failed(error.clone()),
            SessionState::Cancelled => SessionControllerState::Cancelled,
        };
        self.install(clip, state);
    }
    fn install(&mut self, clip: Clip, state: SessionControllerState) {
        // Prepare both PCM handles and final job flags before installing the
        // player, so a completed collection cannot briefly enable offline Play.
        let unavailable = matches!(
            state,
            SessionControllerState::Failed(_) | SessionControllerState::Cancelled
        );
        let playback =
            Player::for_controller(&clip, state == SessionControllerState::Ready, unavailable);
        *self.player.lock().unwrap() = playback;
        self.audio_blocks = clip.audio().block_count();
        self.revision = clip.revision();
        self.state = state;
        self.clip = clip;
    }
    /// Install collected output while the owning job's final result is pending.
    pub fn replace_output(&mut self, clip: Clip) -> Result<()> {
        if self.is_unavailable() {
            return Err(Error("session already ended".into()));
        }
        let state = if clip.session() == &SessionState::Completed {
            SessionControllerState::Finalizing
        } else {
            SessionControllerState::Receiving
        };
        self.install(clip, state);
        Ok(())
    }

    /// Apply a batch outside the audio lock, then call publish once.
    pub fn apply_event(&mut self, event: audio2face3d::types::OutputEvent) -> Result<()> {
        if self.is_unavailable() {
            return Err(Error("session already ended".into()));
        }
        if let Err(error) = crate::inference::apply_event(&mut self.clip, event) {
            self.fail(error.to_string())?;
            return Err(error);
        }
        self.state = if self.clip.session() == &SessionState::Completed {
            SessionControllerState::Finalizing
        } else {
            SessionControllerState::Receiving
        };
        Ok(())
    }
    pub fn apply_event_for(
        &mut self,
        id: ClipId,
        event: audio2face3d::types::OutputEvent,
    ) -> Result<()> {
        self.check_id(id)?;
        self.apply_event(event)
    }
    fn check_id(&self, id: ClipId) -> Result<()> {
        if self.clip.id() == id {
            Ok(())
        } else {
            Err(Error("stale update for another clip".into()))
        }
    }
    pub fn fail(&mut self, message: String) -> Result<()> {
        if !self.is_unavailable() {
            if matches!(
                self.clip.session(),
                SessionState::Idle | SessionState::Running
            ) {
                self.clip
                    .set_session(SessionState::Failed(message.clone()))?;
            }
            self.state = SessionControllerState::Failed(message);
        }
        self.publish()
    }
    pub fn cancel(&mut self) -> Result<()> {
        if !self.is_unavailable() {
            if matches!(
                self.clip.session(),
                SessionState::Idle | SessionState::Running
            ) {
                self.clip.set_session(SessionState::Cancelled)?;
            }
            self.state = SessionControllerState::Cancelled;
        }
        self.publish()
    }
    pub fn job_finished_for(&mut self, id: ClipId, result: Result<()>) -> Result<()> {
        self.check_id(id)?;
        self.job_finished(result)
    }
    pub fn job_finished(&mut self, result: Result<()>) -> Result<()> {
        if self.is_unavailable() || self.state == SessionControllerState::Ready {
            return self.publish();
        }
        match result {
            Err(error) => self.fail(error.to_string()),
            Ok(()) if self.clip.session() == &SessionState::Completed => {
                self.state = SessionControllerState::Ready;
                self.publish()
            }
            Ok(()) => self.fail("missing completion".into()),
        }
    }

    pub fn publish(&mut self) -> Result<()> {
        let mut update = PlaybackUpdate::from_clip(&self.clip, self.audio_blocks)?;
        let revision = self
            .revision
            .checked_add(1)
            .ok_or_else(|| Error("playback revision exhausted".into()))?
            .max(self.clip.revision());
        update.session(
            revision,
            self.state == SessionControllerState::Ready,
            self.is_unavailable(),
        );
        self.player.lock().unwrap().apply_update(update)?;
        self.audio_blocks = self.clip.audio().block_count();
        self.revision = revision;
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
