use crate::{inference::Request, playback::PlaybackState};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Action {
    Initialize,
    Loading,
    Stop,
    Play,
    Pause,
}
impl Action {
    pub fn label(self) -> &'static str {
        match self {
            Self::Initialize => "Initialize & Start",
            Self::Loading => "Abort Initializing",
            Self::Stop => "Stop",
            Self::Play => "Play",
            Self::Pause => "Pause",
        }
    }
    pub fn editable(self) -> bool {
        matches!(self, Self::Initialize | Self::Play)
    }
}

#[derive(Default)]
pub(super) struct Transport {
    prepared: bool,
}
impl Transport {
    pub fn invalidate(&mut self) {
        self.prepared = false;
    }
    /// Only a successful, fully collected offline result can be reused.
    pub fn finished(&mut self, success: bool, streaming: bool) -> bool {
        self.prepared = success && !streaming;
        self.prepared
    }
    pub fn action(&self, streaming: bool, busy: bool, state: PlaybackState) -> Action {
        let playing = matches!(state, PlaybackState::Playing | PlaybackState::Buffering);
        if busy && (!streaming || !playing) {
            Action::Loading
        } else if busy || (streaming && playing) {
            Action::Stop
        } else if !self.prepared || streaming {
            Action::Initialize
        } else if playing {
            Action::Pause
        } else {
            Action::Play
        }
    }
    pub fn can_seek(&self, streaming: bool, busy: bool) -> bool {
        self.prepared && !streaming && !busy
    }
}

pub(super) fn inputs_changed(a: &Request, b: &Request) -> bool {
    a.wav != b.wav
        || a.model != b.model
        || a.mode != b.mode
        || a.endpoint != b.endpoint
        || a.api_key != b.api_key
        || a.device != b.device
        || a.pace_input != b.pace_input
}

#[cfg(test)]
mod tests {
    use super::*;
    use PlaybackState::*;

    #[test]
    fn offline_requires_completion_then_supports_pause_resume_and_invalidation() {
        let mut controls = Transport::default();
        assert_eq!(controls.action(false, false, Paused), Action::Initialize);
        assert_eq!(controls.action(false, true, Paused), Action::Loading);
        assert!(!controls.action(false, true, Paused).editable());
        assert!(!controls.can_seek(false, true));
        assert!(
            controls.finished(true, false),
            "completed offline inference starts playback"
        );
        assert_eq!(controls.action(false, false, Playing), Action::Pause);
        assert!(!controls.action(false, false, Playing).editable());
        assert_eq!(controls.action(false, false, Paused), Action::Play);
        assert_eq!(controls.action(false, false, Ended), Action::Play);
        assert!(controls.can_seek(false, false));
        controls.invalidate();
        assert_eq!(controls.action(false, false, Paused), Action::Initialize);
        assert!(!controls.can_seek(false, false));
    }

    #[test]
    fn streaming_never_reuses_a_result_or_allows_seeking() {
        let mut controls = Transport::default();
        assert_eq!(controls.action(true, true, Paused), Action::Loading);
        assert!(!controls.action(true, true, Paused).editable());
        assert_eq!(controls.action(true, true, Buffering), Action::Stop);
        assert!(!controls.finished(true, true));
        assert_eq!(controls.action(true, false, Playing), Action::Stop);
        assert_eq!(controls.action(true, false, Ended), Action::Initialize);
        assert_eq!(controls.action(true, false, Paused), Action::Initialize);
        assert!(!controls.can_seek(true, false));
        assert!(
            !controls.finished(false, false),
            "failure/cancellation must not autoplay"
        );
        assert_eq!(controls.action(false, false, Paused), Action::Initialize);
    }

    #[test]
    fn every_editable_inference_input_invalidates_the_result() {
        let original = Request::default();
        assert!(!inputs_changed(&original, &original.clone()));
        let changes: [fn(&mut Request); 7] = [
            |r| r.wav = "other.wav".into(),
            |r| r.model = "other.json".into(),
            |r| r.endpoint.push('x'),
            |r| r.api_key.push('x'),
            |r| r.device += 1,
            |r| r.pace_input = !r.pace_input,
            |r| r.mode = crate::inference::Mode::Mock,
        ];
        for change in changes {
            let mut edited = original.clone();
            change(&mut edited);
            assert!(inputs_changed(&original, &edited));
        }
    }
}
