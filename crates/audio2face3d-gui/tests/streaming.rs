use audio2face3d_gui::{
    core::{Clip, SessionState},
    playback::{Command, PlaybackState, Player},
};
use std::time::{Duration, Instant};
fn partial(seconds: f64) -> Clip {
    let mut clip = Clip::running();
    clip.set_names(vec!["JawOpen".into()]).unwrap();
    clip.push_audio(0, &vec![0.5; (seconds * 16000.) as usize])
        .unwrap();
    for i in 0..=(seconds * 30.) as usize {
        clip.push_frame(i as f64 / 30., vec![i as f32 / 30.])
            .unwrap();
    }
    clip
}
#[test]
fn start_threshold_underflow_freeze_and_rebuffer_preserve_media_time() {
    let now = Instant::now();
    let mut player = Player::default();
    player.replace(partial(0.05));
    player.streaming = true;
    player.command(Command::Play, now).unwrap();
    assert_eq!(player.state, PlaybackState::Buffering);
    player.render_audio(1600, 16000, now, |_, v| assert_eq!(v, 0.));
    assert_eq!(player.snapshot(now + Duration::from_secs(1)).time, 0.);
    player.clip.push_audio(800, &vec![0.5; 2400]).unwrap();
    for i in 2..=6 {
        player.clip.push_frame(i as f64 / 30., vec![0.]).unwrap();
    }
    player.render_audio(4000, 16000, now, |_, _| {});
    assert_eq!(player.state, PlaybackState::Buffering);
    assert_eq!(player.underruns, 1);
    assert!((player.snapshot(now + Duration::from_secs(1)).time - 0.2).abs() < 1e-6);
    player.render_audio(1600, 16000, now + Duration::from_secs(1), |_, v| {
        assert_eq!(v, 0.)
    });
    assert!((player.snapshot(now + Duration::from_secs(2)).time - 0.2).abs() < 1e-6);
    player.clip.push_audio(3200, &vec![0.5; 3200]).unwrap();
    for i in 7..=12 {
        player.clip.push_frame(i as f64 / 30., vec![0.]).unwrap();
    }
    player.render_audio(1600, 16000, now + Duration::from_secs(2), |_, v| {
        assert_eq!(v, 0.5)
    });
    assert!((player.snapshot(now + Duration::from_millis(2050)).time - 0.25).abs() < 1e-6);
    assert_eq!(player.clip.frames[6].time, 0.2);
}
#[test]
fn completed_short_clip_flushes_but_failed_stream_does_not_autoplay() {
    let now = Instant::now();
    let mut player = Player::default();
    player.replace(partial(0.02));
    player.streaming = true;
    player.clip.session = SessionState::Completed;
    player.command(Command::Play, now).unwrap();
    let mut nonzero = 0;
    player.render_audio(1600, 16000, now, |_, v| {
        if v != 0. {
            nonzero += 1;
        }
    });
    assert_eq!(nonzero, 320);
    player.clip.session = SessionState::Failed("disconnected".into());
    assert!(player.command(Command::Play, now).is_err());
}
#[test]
fn a_gap_is_not_confused_with_the_largest_received_timestamp() {
    let mut clip = partial(0.1);
    clip.push_frame(0.5, vec![1.]).unwrap();
    clip.push_audio(1600, &vec![0.; 8000]).unwrap();
    assert!((clip.ready_until() - 0.1).abs() < 1e-6);
}
