#![cfg(feature = "session")]
use audio2face3d_gui::{
    core::{Clip, SessionState},
    playback::{Command, PlaybackState, Player},
};
use std::time::{Duration, Instant};
fn clip() -> Clip {
    let mut clip = Clip::default();
    clip.set_names(vec!["Test".into()]).unwrap();
    clip.push_audio(0, &vec![0.25; 16000]).unwrap();
    for i in 0..30 {
        clip.push_frame(i as f64 / 30., vec![if i == 15 { 2. } else { 0. }])
            .unwrap();
    }
    clip.session = SessionState::Completed;
    clip
}
#[test]
fn audible_clock_accounts_for_latency_pause_seek_and_loop() {
    let mut player = Player::default();
    player.replace(clip());
    let now = Instant::now();
    player.command(Command::Play, now).unwrap();
    let mut output = vec![0.; 4800];
    player.render_audio(4800, 48000, now + Duration::from_millis(20), |i, v| {
        output[i] = v
    });
    assert_eq!(player.snapshot(now).time, 0.);
    assert!((player.snapshot(now + Duration::from_millis(70)).time - 0.05).abs() < 1e-7);
    assert!(output.iter().all(|&v| v == 0.25));
    player
        .command(Command::Pause, now + Duration::from_millis(70))
        .unwrap();
    assert!((player.snapshot(now + Duration::from_secs(3)).time - 0.05).abs() < 1e-7);
    player.command(Command::Seek(0.5), now).unwrap();
    let snapshot = player.snapshot(now);
    assert_eq!(snapshot.values, vec![2.]);
    assert_eq!(snapshot.raw_values, vec![2.]);
    player.command(Command::Seek(0.99), now).unwrap();
    player.command(Command::SetLoop(true), now).unwrap();
    player.command(Command::Play, now).unwrap();
    player.render_audio(320, 16000, now, |_, _| {});
    assert!((player.snapshot(now + Duration::from_millis(15)).time - 0.005).abs() < 0.0001);
}
#[test]
fn peak_aggregation_does_not_clamp_or_mutate_raw_values() {
    let clip = clip();
    let envelope = clip.envelope(0, 3);
    assert!(envelope.iter().flatten().any(|v| v[1] == 2.));
    assert!(
        clip.envelope_range(0, 2, 0.49, 0.51)
            .iter()
            .flatten()
            .any(|v| v[1] == 2.)
    );
    assert!(
        clip.envelope_range(0, 2, 0.0, 0.4)
            .iter()
            .flatten()
            .all(|v| v[1] == 0.)
    );
    assert_eq!(clip.frames[15].values[0], 2.);
    assert!(clip.values_at(0.49).0[0] > 1.);
}
#[test]
fn invalid_media_and_future_seek_are_rejected() {
    let mut clip = clip();
    assert!(clip.push_audio(0, &[0.]).is_err());
    assert!(clip.push_frame(f64::NAN, vec![0.]).is_err());
    let mut player = Player::default();
    player.replace(clip);
    assert!(player.command(Command::Seek(2.), Instant::now()).is_err());
    assert_eq!(player.state, PlaybackState::Paused);
}

#[test]
fn streaming_buffer_controls_start_recovery_and_short_completed_clips() {
    let now = Instant::now();
    let mut player = Player::default();
    player.replace(clip());
    player.clip.session = SessionState::Running;
    player.streaming = true;
    player.stream_buffer = Duration::from_millis(1500);
    player.command(Command::Play, now).unwrap();
    assert_eq!(player.state, PlaybackState::Buffering);
    player.render_audio(1, 16000, now, |_, v| assert_eq!(v, 0.));
    assert_eq!(player.state, PlaybackState::Buffering);
    // Complete clips shorter than the target must not wait forever.
    player.clip.session = SessionState::Completed;
    player.render_audio(1, 16000, now, |_, v| assert_eq!(v, 0.25));
    assert_eq!(player.state, PlaybackState::Playing);

    player.replace(clip());
    player.clip.session = SessionState::Running;
    player.streaming = true;
    player.stream_buffer = Duration::from_millis(300);
    player.command(Command::Play, now).unwrap();
    assert_eq!(player.state, PlaybackState::Playing);
    player.render_audio(16001, 16000, now, |_, _| {});
    assert_eq!(player.state, PlaybackState::Buffering);
    player.clip.push_audio(16000, &vec![0.25; 4000]).unwrap();
    for i in 30..38 {
        player.clip.push_frame(i as f64 / 30., vec![0.]).unwrap();
    }
    player.render_audio(1, 16000, now, |_, v| assert_eq!(v, 0.));
    assert_eq!(player.state, PlaybackState::Buffering);
    player.clip.push_audio(20000, &vec![0.25; 1600]).unwrap();
    for i in 38..41 {
        player.clip.push_frame(i as f64 / 30., vec![0.]).unwrap();
    }
    player.render_audio(1, 16000, now, |_, v| assert_eq!(v, 0.25));
    assert_eq!(player.state, PlaybackState::Playing);
    assert_eq!(player.underruns, 1);
}
