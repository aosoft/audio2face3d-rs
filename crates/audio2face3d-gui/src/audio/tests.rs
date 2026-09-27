use super::*;
use std::{sync::mpsc, time::Duration};

fn player() -> SharedPlayer {
    let mut player = Player::default();
    player.replace(crate::core::demo_clip());
    Arc::new(Mutex::new(player))
}

#[test]
fn paused_seek_never_opens_a_device_and_updates_values_immediately() {
    let player = player();
    let (opened, received) = mpsc::channel();
    let mut audio = AudioOutput::with_stream_factory(player.clone(), move |_, _| {
        opened.send(()).unwrap();
        Ok(())
    });
    for time in [0.5, 1., 2.5, 0.] {
        audio.command(Command::Seek(time)).unwrap();
        let snapshot = player.lock().unwrap().snapshot(Instant::now());
        assert_eq!(snapshot.time, time);
        assert_eq!(snapshot.state, PlaybackState::Paused);
        assert_eq!(
            snapshot.values,
            player.lock().unwrap().clip.values_at(time).0
        );
    }
    // Join the worker so all pending work finishes before checking device calls.
    drop(audio);
    assert!(
        received.try_recv().is_err(),
        "paused inspection must not open an audio device"
    );
}

#[test]
fn blocked_device_start_does_not_block_seek_and_obsolete_callbacks_are_silent() {
    let player = player();
    let (started, starts) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let mut audio = AudioOutput::with_stream_factory(player.clone(), move |device, generation| {
        started.send((device.clone(), generation)).unwrap();
        gate.recv_timeout(Duration::from_secs(5))
            .map_err(|e| Error(e.to_string()))?;
        Ok(())
    });
    audio.command(Command::Play).unwrap();
    let (old_device, old_generation) = starts.recv_timeout(Duration::from_secs(2)).unwrap();
    // These calls must return while device setup is still blocked on the gate.
    audio.command(Command::Seek(1.)).unwrap();
    audio.command(Command::Seek(2.)).unwrap();
    audio.command(Command::Pause).unwrap();
    audio.command(Command::Seek(3.)).unwrap();
    audio.command(Command::Play).unwrap();
    let latest = audio.generation.load(Ordering::SeqCst);
    assert_eq!(player.lock().unwrap().snapshot(Instant::now()).time, 3.);
    let mut samples = [1_f32; 160];
    old_device.render(old_generation, &mut samples, 1, 16000, Instant::now());
    assert!(samples.iter().all(|&v| v == 0.));
    assert_eq!(player.lock().unwrap().snapshot(Instant::now()).time, 3.);
    release.send(()).unwrap();
    let (_, next_generation) = starts.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(
        next_generation, latest,
        "rapid requests must coalesce to the latest seek"
    );
    release.send(()).unwrap();
    audio.stop();
}

#[test]
fn obsolete_device_failure_does_not_pause_the_new_generation() {
    let player = player();
    let (started, starts) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let mut first = true;
    let mut audio = AudioOutput::with_stream_factory(player.clone(), move |_, generation| {
        started.send(generation).unwrap();
        if first {
            first = false;
            gate.recv_timeout(Duration::from_secs(5))
                .map_err(|e| Error(e.to_string()))?;
            return Err(Error("obsolete device failure".into()));
        }
        Ok(())
    });
    audio.command(Command::Play).unwrap();
    starts.recv_timeout(Duration::from_secs(2)).unwrap();
    audio.command(Command::Seek(2.)).unwrap();
    release.send(()).unwrap();
    assert_eq!(
        starts.recv_timeout(Duration::from_secs(2)).unwrap(),
        audio.generation.load(Ordering::SeqCst)
    );
    assert!(audio.errors.lock().unwrap().is_none());
    let snapshot = player.lock().unwrap().snapshot(Instant::now());
    assert_eq!(snapshot.time, 2.);
    assert_eq!(snapshot.state, PlaybackState::Playing);
}

#[test]
fn current_device_failure_is_reported_and_pauses_playback() {
    let player = player();
    let mut audio = AudioOutput::with_stream_factory(player.clone(), |_, _| -> Result<()> {
        Err(Error("test device unavailable".into()))
    });
    audio.command(Command::Seek(1.)).unwrap();
    audio.command(Command::Play).unwrap();
    let until = Instant::now() + Duration::from_secs(2);
    while audio.errors.lock().unwrap().is_none() && Instant::now() < until {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        audio.errors.lock().unwrap().as_deref(),
        Some("test device unavailable")
    );
    let snapshot = player.lock().unwrap().snapshot(Instant::now());
    assert_eq!(snapshot.state, PlaybackState::Paused);
    assert_eq!(snapshot.time, 1.);
}
