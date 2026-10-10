use super::*;
use crate::playback::{Command, PlaybackState};
use std::{sync::Barrier, time::Duration};

fn clip(samples: &[f32], chunks: &[usize]) -> Clip {
    let mut clip = Clip::default();
    clip.set_names(vec!["JawOpen".into()]).unwrap();
    let mut start = 0;
    for &len in chunks {
        clip.push_audio(start as u64, &samples[start..start + len])
            .unwrap();
        start += len;
    }
    assert_eq!(start, samples.len());
    for index in 0..=(samples.len() as f64 / 16000. * 30.) as usize {
        clip.push_frame(index as f64 / 30., vec![index as f32 / 30.])
            .unwrap();
    }
    clip.set_session(SessionState::Completed);
    clip
}

#[test]
fn graph_aggregation_stopped_at_a_barrier_does_not_hold_the_audio_lock() {
    let player = Arc::new(Mutex::new(Player::default()));
    let mut controller = SessionController::new(player.clone());
    controller.replace(crate::core::demo_clip());
    let now = Instant::now();
    player.lock().unwrap().command(Command::Play, now).unwrap();
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let ui_entered = entered.clone();
    let ui_release = release.clone();
    let ui = std::thread::spawn(move || {
        controller
            .with_view(now, |clip, clock| {
                ui_entered.wait();
                ui_release.wait();
                assert_eq!(clock.clip_id, clip.id());
                for channel in 0..clip.names().len() {
                    let _ = clip.envelope(channel, 1024);
                }
                clock.with_clip(clip).unwrap()
            })
            .unwrap()
    });
    entered.wait();
    let mut audio = player
        .try_lock()
        .expect("controller graph work must not retain the audio lock");
    let mut samples = vec![0.; 160];
    audio.render_audio(samples.len(), 16000, now, |i, value| samples[i] = value);
    assert!(samples.iter().any(|value| *value != 0.));
    assert!(audio.clip.frames().is_empty());
    assert!(audio.clip.names().is_empty());
    drop(audio);
    release.wait();
    let view = ui.join().unwrap();
    assert_eq!(view.time, 0.);
}

#[test]
fn block_and_contiguous_audio_match_across_seek_pause_loop_and_rates() {
    let samples: Vec<_> = (0..257).map(|i| i as f32 / 512.).collect();
    for rate in [16000, 48000] {
        let blocked = clip(&samples, &[7, 1, 71, 3, 175]);
        let mut block_player = Player::from_clip(&blocked);
        let mut reference = Player::default();
        reference.replace(clip(&samples, &[257]));
        let now = Instant::now();
        for command in [
            Command::Play,
            Command::Pause,
            Command::Seek(6.5 / 16000.),
            Command::Play,
            Command::SetLoop(true),
            Command::Seek(255. / 16000.),
        ] {
            reference.command(command, now).unwrap();
            block_player.command(command, now).unwrap();
            let mut flat = vec![0.; 40];
            let mut blocks = vec![0.; 40];
            reference.render_audio(40, rate, now, |i, value| flat[i] = value);
            block_player.render_audio(40, rate, now, |i, value| blocks[i] = value);
            assert_eq!(blocks, flat, "rate={rate}, command={command:?}");
            let later = now + Duration::from_micros(250);
            assert_eq!(
                block_player.clock_snapshot(later).time,
                reference.clock_snapshot(later).time
            );
        }
    }
}

#[test]
fn controller_shares_payload_and_rejects_stale_clock_and_readiness_updates() {
    let player = Arc::new(Mutex::new(Player::default()));
    let mut controller = SessionController::new(player.clone());
    controller.replace(crate::core::demo_clip());
    let old_clock = controller.clock_snapshot(Instant::now());
    let old_update = PlaybackUpdate::from_clip(controller.clip(), controller.audio_blocks).unwrap();
    assert!(
        controller
            .clip()
            .audio
            .shares_payload_with(player.lock().unwrap().audio_track())
    );
    let next = crate::core::demo_clip();
    assert!(old_clock.with_clip(&next).is_err());
    controller.replace(next);
    let prior_state =
        PlaybackUpdate::from_clip(controller.clip(), controller.audio_blocks).unwrap();
    controller.set_session(SessionState::Failed("newer failure".into()));
    controller.publish().unwrap();
    let mut playback = player.lock().unwrap();
    assert!(playback.apply_update(old_update).is_err());
    assert!(playback.apply_update(prior_state).is_err());
    assert_eq!(
        playback.clock_snapshot(Instant::now()).clip_id,
        controller.clip().id()
    );
}

#[test]
fn incremental_blocks_recover_from_underrun_without_replacing_the_clock() {
    let samples = vec![0.5; 1600];
    let mut clip = clip(&samples, &[800, 800]);
    clip.set_session(SessionState::Running);
    let player = Arc::new(Mutex::new(Player::default()));
    let mut controller = SessionController::new(player.clone());
    controller.replace(clip);
    let now = Instant::now();
    {
        let mut playback = player.lock().unwrap();
        playback.streaming = true;
        playback.stream_buffer = Duration::from_millis(10);
        playback.command(Command::Play, now).unwrap();
        playback.render_audio(1601, 16000, now, |_, _| {});
        assert_eq!(playback.state, PlaybackState::Buffering);
        assert_eq!(playback.underruns, 1);
    }
    controller.clip.push_audio(1600, &vec![0.5; 1600]).unwrap();
    for index in 4..=6 {
        controller
            .clip
            .push_frame(index as f64 / 30., vec![0.])
            .unwrap();
    }
    controller.set_session(SessionState::Completed);
    controller.publish().unwrap();
    {
        let mut playback = player.lock().unwrap();
        assert!(
            controller
                .clip()
                .audio
                .shares_payload_with(playback.audio_track())
        );
        playback.render_audio(1, 16000, now + Duration::from_secs(1), |_, value| {
            assert_eq!(value, 0.5)
        });
        assert_eq!(playback.state, PlaybackState::Playing);
        assert_eq!(playback.underruns, 1);
    }
    controller.set_session(SessionState::Failed("disconnected".into()));
    controller.publish().unwrap();
    player
        .lock()
        .unwrap()
        .render_audio(1, 16000, now, |_, value| assert_eq!(value, 0.));
    assert_eq!(player.lock().unwrap().state, PlaybackState::Paused);
}

#[test]
fn tiny_audio_blocks_are_bounded_and_rejection_preserves_partial_data() {
    let mut clip = Clip::running();
    for index in 0..crate::core::MAX_AUDIO_BLOCKS {
        clip.push_audio(index as u64, &[0.]).unwrap();
    }
    let revision = clip.revision;
    assert!(clip.push_audio(clip.audio.len() as u64, &[0.]).is_err());
    assert_eq!(clip.audio.len(), crate::core::MAX_AUDIO_BLOCKS);
    assert_eq!(clip.revision, revision);
}
