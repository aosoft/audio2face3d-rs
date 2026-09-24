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
    audio.command(Command::Seek(5.9))?;
    audio.command(Command::SetLoop(true))?;
    audio.command(Command::Play)?;
    std::thread::sleep(Duration::from_millis(400));
    let looped = shared.lock().unwrap().snapshot(Instant::now()).time;
    assert!(looped < 1., "loop failed: {looped}");
    audio.stop();
    if let Some(error) = audio.errors.lock().unwrap().take() {
        return Err(error.into());
    }
    println!(
        "CPAL: audible position {position:.3}s after 600ms; pause held {paused:.3}s; seek/loop {looped:.3}s"
    );
    Ok(())
}
