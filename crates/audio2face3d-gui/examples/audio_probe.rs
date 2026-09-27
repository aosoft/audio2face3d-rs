//! Explicit audio-device test. Exercises real callback timestamps and queue reset.
use audio2face3d_gui::{
    audio::AudioOutput,
    core::demo_clip,
    playback::{Command, Player},
};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut player = Player::default();
    player.replace(demo_clip());
    let shared = Arc::new(Mutex::new(player));
    let mut audio = AudioOutput::new(shared.clone());
    audio.command(Command::Play)?;
    std::thread::sleep(Duration::from_millis(600));
    let position = shared.lock().unwrap().snapshot(Instant::now()).time;
    assert!(
        position > 0.3 && position < 1.,
        "device failed to advance: {position}"
    );
    audio.command(Command::Pause)?;
    let paused = shared.lock().unwrap().snapshot(Instant::now()).time;
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(paused, shared.lock().unwrap().snapshot(Instant::now()).time);
    let seek_started = Instant::now();
    audio.command(Command::Seek(5.9))?;
    let paused_seek = seek_started.elapsed();
    assert_eq!(shared.lock().unwrap().snapshot(Instant::now()).time, 5.9);
    audio.command(Command::SetLoop(true))?;
    audio.command(Command::Play)?;
    std::thread::sleep(Duration::from_millis(400));
    let looped = shared.lock().unwrap().snapshot(Instant::now()).time;
    assert!(looped < 1., "loop failed: {looped}");
    let seek_started = Instant::now();
    audio.command(Command::Seek(2.))?;
    let playing_seek = seek_started.elapsed();
    let immediate = shared.lock().unwrap().snapshot(Instant::now()).time;
    assert!(
        (immediate - 2.).abs() < 0.05,
        "seek not immediately reflected: {immediate}"
    );
    std::thread::sleep(Duration::from_millis(400));
    let resumed = shared.lock().unwrap().snapshot(Instant::now()).time;
    assert!(
        resumed > 2.1 && resumed < 2.6,
        "playback failed to resume after seek: {resumed}"
    );
    audio.stop();
    if let Some(error) = audio.errors.lock().unwrap().take() {
        return Err(error.into());
    }
    println!(
        "CPAL: audible position {position:.3}s after 600ms; pause held {paused:.3}s; seek/loop {looped:.3}s"
    );
    println!(
        "Seek command latency: paused {paused_seek:?}, playing {playing_seek:?}; resumed at {resumed:.3}s"
    );
    Ok(())
}
