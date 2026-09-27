//! Invoke the same worker as the GUI, optionally saving results for path comparison.
use audio2face3d_gui::{
    core::{Clip, SessionState},
    inference::{Event, Job, Mode, Request, apply_event},
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let mode = match args.get(1).map(String::as_str) {
        Some("local") => Mode::Local,
        Some("grpc") => Mode::Grpc,
        Some("mock") => Mode::Mock,
        _ => return Err("mode: local/grpc/mock".into()),
    };
    let request = Request {
        mode,
        wav: args.get(2).ok_or("WAV required")?.into(),
        model: args.get(3).cloned().unwrap_or_default().into(),
        endpoint: args
            .get(4)
            .cloned()
            .unwrap_or_else(|| "http://127.0.0.1:50051".into()),
        cuda_root: std::env::var_os("CUDA_PATH").unwrap_or_default().into(),
        tensorrt_root: std::env::var_os("TENSORRT_ROOT_DIR")
            .unwrap_or_default()
            .into(),
        ..Default::default()
    };
    let (logger, mut logs) =
        audio2face3d_gui::logging::channel(2048, 10000, audio2face3d::logging::LogLevel::Info);
    let mut job = Job::start(1, request, logger as Arc<dyn audio2face3d::logging::Logger>)?;
    let start = Instant::now();
    let mut clip = Clip::running();
    let mut input_finished = false;
    loop {
        let mut events: Vec<_> = job.events.try_iter().collect();
        let finished = job.try_finish();
        if finished.is_some() {
            events.extend(job.events.try_iter());
        }
        for event in events {
            match event {
                Event::InputFinished => input_finished = true,
                Event::Output(event) => apply_event(&mut clip, event)?,
            }
        }
        logs.drain();
        if let Some(result) = finished {
            result?;
            break;
        }
        if start.elapsed() > Duration::from_secs(180) {
            job.cancel();
            return Err("inference timeout".into());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(clip.session, SessionState::Completed);
    assert!(input_finished);
    assert!(!clip.frames.is_empty());
    println!(
        "{mode:?}: {} channels, {} frames, {:.3}s audio, {:.3}s elapsed; resources released",
        clip.names.len(),
        clip.frames.len(),
        clip.duration(),
        start.elapsed().as_secs_f64()
    );
    if let Some(path) = std::env::var_os("A2F_PROBE_OUTPUT") {
        let data = serde_json::json!({"names":clip.names,"frames":clip.frames.iter().map(|f|serde_json::json!({"time":f.time,"values":f.values})).collect::<Vec<_>>(),"audio":clip.audio});
        std::fs::write(path, serde_json::to_vec(&data)?)?;
    }
    Ok(())
}
