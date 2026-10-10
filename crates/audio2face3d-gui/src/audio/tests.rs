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
            player.lock().unwrap().clip().unwrap().values_at(time).0
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
    let attempted = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let opened = attempted.clone();
    let (handled, handler) = mpsc::channel();
    let mut audio = AudioOutput::with_stream_factory_and_before_wait(
        player.clone(),
        move |_, _| -> Result<()> {
            opened.store(true, Ordering::Release);
            Err(Error("test device unavailable".into()))
        },
        move || {
            if attempted.swap(false, Ordering::AcqRel) {
                handled.send(()).unwrap();
            }
        },
    );
    audio.command(Command::Seek(1.)).unwrap();
    audio.command(Command::Play).unwrap();
    handler.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(
        audio.errors.lock().unwrap().as_deref(),
        Some("test device unavailable")
    );
    let snapshot = player.lock().unwrap().snapshot(Instant::now());
    assert_eq!(snapshot.state, PlaybackState::Paused);
    assert_eq!(snapshot.time, 1.);
}

struct FakeStream {
    generation: u64,
    player: SharedPlayer,
    dropped: mpsc::Sender<(u64, std::thread::ThreadId, bool)>,
}
impl Drop for FakeStream {
    fn drop(&mut self) {
        let _ = self.dropped.send((
            self.generation,
            std::thread::current().id(),
            self.player.try_lock().is_ok(),
        ));
    }
}
struct Harness {
    audio: AudioOutput,
    player: SharedPlayer,
    devices: mpsc::Receiver<Device>,
    dropped: mpsc::Receiver<(u64, std::thread::ThreadId, bool)>,
    observed: mpsc::Receiver<(u64, u8, u64)>,
    starts: Arc<AtomicU64>,
}
fn harness() -> Harness {
    let player = player();
    let (devices, started) = mpsc::channel();
    let (dropped, drops) = mpsc::channel();
    let (observed, observations) = mpsc::channel();
    let epoch = Arc::new(Mutex::new(None::<Arc<StreamEpoch>>));
    let running_epoch = epoch.clone();
    let starts = Arc::new(AtomicU64::new(0));
    let calls = starts.clone();
    let audio = AudioOutput::with_stream_factory_and_before_wait(
        player.clone(),
        move |device, generation| {
            calls.fetch_add(1, Ordering::Relaxed);
            *epoch.lock().unwrap() = Some(device.epoch.clone());
            devices.send(device.clone()).unwrap();
            Ok(FakeStream {
                generation,
                player: device.state.player.clone(),
                dropped: dropped.clone(),
            })
        },
        move || {
            if let Some(epoch) = running_epoch.lock().unwrap().clone() {
                let _ = observed.send((
                    epoch.generation,
                    epoch.state.load(Ordering::Acquire),
                    epoch.repeated_faults.load(Ordering::Acquire),
                ));
            }
        },
    );
    Harness {
        audio,
        player,
        devices: started,
        dropped: drops,
        observed: observations,
        starts,
    }
}
fn fault(message: &str) -> cpal::StreamError {
    cpal::BackendSpecificError {
        description: message.into(),
    }
    .into()
}
impl Harness {
    fn device(&self) -> Device {
        self.devices.recv_timeout(Duration::from_secs(2)).unwrap()
    }
    fn wait_state(&self, generation: u64, state: u8, repeats: u64) {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let observation = self
                .observed
                .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                .unwrap();
            if observation.0 == generation && observation.1 == state && observation.2 >= repeats {
                return;
            }
        }
    }
}

#[test]
fn runtime_fault_reporting_never_needs_player_or_status_locks() {
    let mut harness = harness();
    harness.audio.command(Command::Seek(1.)).unwrap();
    harness.audio.command(Command::Play).unwrap();
    let device = harness.device();
    let player = harness.player.lock().unwrap();
    let errors = harness.audio.errors.lock().unwrap();
    let status = harness.audio.failure.lock().unwrap();
    let epoch = device.epoch.clone();
    let (returned, return_notification) = mpsc::channel();
    let callback = std::thread::spawn(move || {
        epoch.report_fault(fault("runtime failure"));
        returned.send(()).unwrap();
    });
    return_notification
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    let mut output = [1_f32; 16];
    device.render(
        device.epoch.generation,
        &mut output,
        1,
        16000,
        Instant::now(),
    );
    assert_eq!(output, [0.; 16]);
    assert_eq!(harness.audio.callback_contention.load(Ordering::Relaxed), 0);
    drop(status);
    drop(errors);
    drop(player);
    callback.join().unwrap();
    let (generation, thread, lock_free) = harness
        .dropped
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    assert_eq!(generation, device.epoch.generation);
    assert_eq!(thread, device.epoch.worker_thread.id());
    assert!(lock_free, "stream must be dropped outside the player lock");
    assert_eq!(
        harness.player.lock().unwrap().state(),
        PlaybackState::Paused
    );
    assert_eq!(
        harness.player.lock().unwrap().snapshot(Instant::now()).time,
        1.
    );
    assert!(
        harness
            .audio
            .take_error()
            .unwrap()
            .contains("runtime failure")
    );
    assert!(harness.audio.take_error().is_none());
}

#[test]
fn simultaneous_callbacks_keep_one_fault_and_count_all_repeats() {
    let epoch = Arc::new(StreamEpoch::new(17));
    let barrier = Arc::new(std::sync::Barrier::new(9));
    let threads: Vec<_> = (0..8)
        .map(|index| {
            let epoch = epoch.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                epoch.report_fault(fault(&format!("fault-{index}")));
            })
        })
        .collect();
    barrier.wait();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(epoch.state.load(Ordering::Acquire), FAULT_READY);
    assert_eq!(epoch.repeated_faults.load(Ordering::Acquire), 7);
    let original = epoch.first_fault.get().unwrap().to_string();
    assert!(original.contains("fault-"));
    assert_eq!(epoch.take_fault(), Some(original.clone()));
    epoch.report_fault(fault("must not replace the first fault"));
    assert_eq!(epoch.first_fault.get().unwrap().to_string(), original);
    assert_eq!(epoch.repeated_faults.load(Ordering::Acquire), 8);
    assert!(epoch.take_fault().is_none());
}

#[test]
fn handled_fault_is_not_restarted_and_old_notification_cannot_stop_resume() {
    let mut harness = harness();
    harness.audio.command(Command::Play).unwrap();
    let old = harness.device();
    old.epoch.report_fault(fault("first failure"));
    harness.wait_state(old.epoch.generation, FAULT_HANDLED, 0);
    harness
        .dropped
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    old.epoch.report_fault(fault("repeated failure"));
    harness.wait_state(old.epoch.generation, FAULT_HANDLED, 1);
    assert_eq!(harness.starts.load(Ordering::Relaxed), 1);
    assert!(harness.dropped.try_recv().is_err());
    harness.audio.command(Command::Play).unwrap();
    let new = harness.device();
    assert_ne!(new.epoch.generation, old.epoch.generation);
    old.epoch.report_fault(fault("late failure"));
    // Exercise the same final generation check even if an old READY was read
    // just before the new command won the player lock.
    let mut sentinel = Some(ActiveStream {
        _stream: (),
        epoch: new.epoch.clone(),
    });
    assert!(
        !old.state
            .handle_failure(old.epoch.generation, "stale status".into(), &mut sentinel)
    );
    assert!(sentinel.is_some());
    assert_eq!(
        harness.player.lock().unwrap().state(),
        PlaybackState::Playing
    );
    assert!(harness.audio.take_error().is_none());
    assert!(harness.dropped.try_recv().is_err());
    assert_eq!(harness.starts.load(Ordering::Relaxed), 2);
}

#[test]
fn fault_during_start_is_handled_before_the_worker_parks() {
    let player = player();
    let (dropped, drops) = mpsc::channel();
    let (started, starts) = mpsc::channel();
    let mut audio = AudioOutput::with_stream_factory(player.clone(), move |device, generation| {
        started.send(device.clone()).unwrap();
        device.epoch.report_fault(fault("opening stream failure"));
        Ok(FakeStream {
            generation,
            player: device.state.player.clone(),
            dropped: dropped.clone(),
        })
    });
    audio.command(Command::Play).unwrap();
    let device = starts.recv_timeout(Duration::from_secs(2)).unwrap();
    let (generation, thread, lock_free) = drops.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(generation, device.epoch.generation);
    assert_eq!(thread, device.epoch.worker_thread.id());
    assert!(lock_free);
    assert_eq!(player.lock().unwrap().state(), PlaybackState::Paused);
    assert!(
        audio
            .take_error()
            .unwrap()
            .contains("opening stream failure")
    );
}

#[test]
fn fault_before_park_keeps_a_wake_token() {
    let player = player();
    let (started, starts) = mpsc::channel();
    let (parked, parks) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let (dropped, drops) = mpsc::channel();
    let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let opened = armed.clone();
    let mut audio = AudioOutput::with_stream_factory_and_before_wait(
        player.clone(),
        move |device, generation| {
            opened.store(true, Ordering::Release);
            started.send(device.clone()).unwrap();
            Ok(FakeStream {
                generation,
                player: device.state.player.clone(),
                dropped: dropped.clone(),
            })
        },
        move || {
            if armed.swap(false, Ordering::AcqRel) {
                parked.send(()).unwrap();
                released.recv_timeout(Duration::from_secs(5)).unwrap();
            }
        },
    );
    audio.command(Command::Play).unwrap();
    let device = starts.recv_timeout(Duration::from_secs(2)).unwrap();
    parks.recv_timeout(Duration::from_secs(2)).unwrap();
    device.epoch.report_fault(fault("before park"));
    release.send(()).unwrap();
    drops.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(player.lock().unwrap().state(), PlaybackState::Paused);
}

#[test]
fn writing_fault_wake_is_followed_by_ready_publication_and_another_wake() {
    let mut harness = harness();
    harness.audio.command(Command::Play).unwrap();
    let device = harness.device();
    let epoch = device.epoch.clone();
    let (writing, writes) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let callback = std::thread::spawn(move || {
        epoch.publish_fault(fault("first incomplete publication"), || {
            writing.send(()).unwrap();
            released.recv_timeout(Duration::from_secs(5)).unwrap();
        });
    });
    writes.recv_timeout(Duration::from_secs(2)).unwrap();
    device.epoch.report_fault(fault("repeat while writing"));
    harness.wait_state(device.epoch.generation, WRITING_FAULT, 1);
    assert_eq!(
        harness.player.lock().unwrap().state(),
        PlaybackState::Playing
    );
    assert!(harness.dropped.try_recv().is_err());
    release.send(()).unwrap();
    callback.join().unwrap();
    harness.wait_state(device.epoch.generation, FAULT_HANDLED, 1);
    harness
        .dropped
        .recv_timeout(Duration::from_secs(2))
        .unwrap();
    assert!(
        harness
            .audio
            .take_error()
            .unwrap()
            .contains("first incomplete publication")
    );
}
