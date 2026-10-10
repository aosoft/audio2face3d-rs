use super::*;
use crate::playback::{Command, PlaybackState};
use std::{sync::Barrier, time::Duration};

#[test]
fn completed_collection_waits_for_job_and_late_failure_or_cancel_stops_playback() {
    for cancelled in [false, true] {
        let player = Arc::new(Mutex::new(Player::default()));
        let mut controller = SessionController::new(player.clone());
        controller.begin();
        controller.replace_output(crate::core::demo_clip()).unwrap();
        assert_eq!(controller.state(), &SessionControllerState::Finalizing);
        let now = Instant::now();
        assert!(player.lock().unwrap().command(Command::Play, now).is_err());
        {
            let mut playback = player.lock().unwrap();
            playback
                .set_streaming(true, Duration::from_millis(100))
                .unwrap();
            playback.command(Command::Play, now).unwrap();
            playback.render_audio(160, 16000, now, |_, _| {});
        }
        if cancelled {
            controller.cancel().unwrap();
        } else {
            controller
                .job_finished(Err(Error("cleanup failed".into())))
                .unwrap();
        }
        assert_eq!(controller.clip().session(), &SessionState::Completed);
        assert!(controller.is_unavailable());
        let mut playback = player.lock().unwrap();
        assert_eq!(playback.state(), PlaybackState::Paused);
        assert!(playback.command(Command::Play, now).is_err());
        playback.render_audio(160, 16000, now, |_, value| assert_eq!(value, 0.));
        drop(playback);
        controller.job_finished(Ok(())).unwrap();
        assert!(
            controller.is_unavailable(),
            "later success must not clear failure/cancel"
        );
    }
}

#[test]
fn successful_job_enables_offline_play_and_old_clip_updates_do_not_change_new_session() {
    let player = Arc::new(Mutex::new(Player::default()));
    let mut controller = SessionController::new(player.clone());
    controller.begin();
    let obsolete = controller.clip().id();
    controller.replace_output(crate::core::demo_clip()).unwrap();
    let current = controller.clip().id();
    let before = controller.clip().revision();
    assert!(
        controller
            .job_finished_for(obsolete, Err(Error("old failure".into())))
            .is_err()
    );
    assert!(
        controller
            .apply_event_for(
                obsolete,
                audio2face3d::types::OutputEvent::Completed(audio2face3d::types::Summary {
                    progress: Default::default()
                })
            )
            .is_err()
    );
    assert_eq!(controller.clip().revision(), before);
    assert_eq!(controller.state(), &SessionControllerState::Finalizing);
    controller.job_finished_for(current, Ok(())).unwrap();
    assert_eq!(controller.state(), &SessionControllerState::Ready);
    player
        .lock()
        .unwrap()
        .command(Command::Play, Instant::now())
        .unwrap();
}

fn clip(samples: &[f32], chunks: &[usize]) -> Clip {
    let mut clip = partial_clip(samples, chunks);
    clip.finish_collection().unwrap();
    clip
}
fn partial_clip(samples: &[f32], chunks: &[usize]) -> Clip {
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
    assert!(audio.clip().is_none());
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
            .audio()
            .shares_payload_with(player.lock().unwrap().audio_track())
    );
    let next = crate::core::demo_clip();
    assert!(old_clock.with_clip(&next).is_err());
    controller.replace(next);
    let prior_state =
        PlaybackUpdate::from_clip(controller.clip(), controller.audio_blocks).unwrap();
    controller.fail("newer failure".into()).unwrap();
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
    let clip = partial_clip(&samples, &[800, 800]);
    let player = Arc::new(Mutex::new(Player::default()));
    let mut controller = SessionController::new(player.clone());
    controller.replace(clip);
    let now = Instant::now();
    {
        let mut playback = player.lock().unwrap();
        playback
            .set_streaming(true, Duration::from_millis(10))
            .unwrap();
        playback.command(Command::Play, now).unwrap();
        playback.render_audio(1601, 16000, now, |_, _| {});
        assert_eq!(playback.state(), PlaybackState::Buffering);
        assert_eq!(playback.underruns(), 1);
    }
    controller.clip.push_audio(1600, &vec![0.5; 1600]).unwrap();
    for index in 4..=6 {
        controller
            .clip
            .push_frame(index as f64 / 30., vec![0.])
            .unwrap();
    }
    controller.clip.finish_collection().unwrap();
    controller.state = SessionControllerState::Finalizing;
    controller.publish().unwrap();
    {
        let mut playback = player.lock().unwrap();
        assert!(
            controller
                .clip()
                .audio()
                .shares_payload_with(playback.audio_track())
        );
        playback.render_audio(1, 16000, now + Duration::from_secs(1), |_, value| {
            assert_eq!(value, 0.5)
        });
        assert_eq!(playback.state(), PlaybackState::Playing);
        assert_eq!(playback.underruns(), 1);
    }
    controller
        .job_finished(Err(Error("disconnected".into())))
        .unwrap();
    player
        .lock()
        .unwrap()
        .render_audio(1, 16000, now, |_, value| assert_eq!(value, 0.));
    assert_eq!(player.lock().unwrap().state(), PlaybackState::Paused);
}

#[test]
fn tiny_audio_blocks_are_bounded_and_rejection_preserves_partial_data() {
    let mut clip = Clip::running();
    for index in 0..crate::core::MAX_AUDIO_BLOCKS {
        clip.push_audio(index as u64, &[0.]).unwrap();
    }
    let revision = clip.revision();
    assert!(clip.push_audio(clip.audio().len() as u64, &[0.]).is_err());
    assert_eq!(clip.audio().len(), crate::core::MAX_AUDIO_BLOCKS);
    assert_eq!(clip.revision(), revision);
}
