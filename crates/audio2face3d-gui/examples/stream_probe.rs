//! Exercise actual audio output while the shared inference worker still sends input.
use audio2face3d_gui::{
    audio::AudioOutput,
    core::{Clip, SessionState},
    inference::{Event, Job, Mode, Request, apply_event},
    playback::{Command, Player},
};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mode = match args.get(1).map(String::as_str) {
        Some("local") => Mode::Local,
        Some("grpc") => Mode::Grpc,
        Some("mock") => Mode::Mock,
        _ => return Err("mode required".into()),
    };
    let request = Request {
        mode,
        wav: args.get(2).ok_or("WAV required")?.into(),
        model: args.get(3).cloned().unwrap_or_default().into(),
        endpoint: args
            .get(4)
            .cloned()
            .unwrap_or_else(|| "http://127.0.0.1:52000".into()),
        cuda_root: std::env::var_os("CUDA_PATH").unwrap_or_default().into(),
        tensorrt_root: std::env::var_os("TENSORRT_ROOT_DIR")
            .unwrap_or_default()
            .into(),
        pace_input: true,
        ..Default::default()
    };
    let mut job = Job::start(1, request, Arc::new(audio2face3d::logging::NoopLogger))?;
    let mut player = Player::default();
    player.replace(Clip::running());
    player.streaming = true;
    let shared = Arc::new(Mutex::new(player));
    let mut audio = AudioOutput::new(shared.clone());
    let start = Instant::now();
    let mut sent = false;
    let mut first = None;
    let mut play = None;
    let mut heard_before_end = false;
    let mut finished = false;
    while !finished {
        let mut events: Vec<_> = job.events.try_iter().collect();
        let result = job.try_finish();
        if result.is_some() {
            events.extend(job.events.try_iter());
        }
        for event in events {
            match event {
                Event::InputFinished => sent = true,
                Event::Output(event) => {
                    if matches!(&event, audio2face3d::types::OutputEvent::Curves(_))
                        && first.is_none()
                    {
                        first = Some(start.elapsed().as_secs_f64());
                    }
                    apply_event(&mut shared.lock().unwrap().clip, event)?;
                }
            }
        }
        let ready = {
            let p = shared.lock().unwrap();
            p.clip.ready_until() >= 0.1
                || (p.clip.session == SessionState::Completed && p.clip.ready_until() > 0.)
        };
        if play.is_none() && ready {
            audio.command(Command::Play)?;
            play = Some(start.elapsed().as_secs_f64());
        }
        if !sent && shared.lock().unwrap().snapshot(Instant::now()).time > 0.01 {
            heard_before_end = true;
        }
        if let Some(result) = result {
            result?;
            finished = true;
        }
        if start.elapsed() > Duration::from_secs(180) {
            return Err("streaming timeout".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let duration = shared.lock().unwrap().clip.duration();
    while shared.lock().unwrap().snapshot(Instant::now()).time + 0.001 < duration {
        if start.elapsed() > Duration::from_secs(180) {
            return Err("playback timeout".into());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    let underruns = shared.lock().unwrap().underruns;
    audio.stop();
    println!(
        "{mode:?}: first frame {first:?}s, playback start {play:?}s, total {:.3}s, duration {duration:.3}s, underruns {underruns}, audible before EndOfAudio: {heard_before_end}",
        start.elapsed().as_secs_f64()
    );
    assert!(heard_before_end, "input ended before playback began");
    Ok(())
}
