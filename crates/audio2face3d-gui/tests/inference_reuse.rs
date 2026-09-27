#![cfg(all(feature = "session", any(feature = "mock", feature = "local")))]
use audio2face3d_gui::{
    core::{Clip, SessionState},
    inference::{Engine, Event, Job, Mode, Request, apply_event},
};
use std::time::{Duration, Instant};

fn collect(mut job: Job) -> Clip {
    let mut clip = Clip::running();
    let start = Instant::now();
    loop {
        let mut events: Vec<_> = job.events.try_iter().collect();
        let finished = job.try_finish();
        if finished.is_some() {
            events.extend(job.events.try_iter());
        }
        for event in events {
            if let Event::Output(event) = event {
                apply_event(&mut clip, event).unwrap();
            }
        }
        if let Some(result) = finished {
            result.unwrap();
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(180),
            "inference timed out"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(clip.session, SessionState::Completed);
    clip
}
fn compare(a: &Clip, b: &Clip) {
    assert_eq!(a.audio, b.audio);
    assert_eq!(a.names, b.names);
    assert_eq!(a.frames.len(), b.frames.len());
    for (a, b) in a.frames.iter().zip(&b.frames) {
        assert_eq!(a.time, b.time);
        for (a, b) in a.values.iter().zip(&b.values) {
            assert!((a - b).abs() < 1e-4, "{a} != {b}");
        }
    }
}
#[cfg(feature = "mock")]
#[test]
fn completed_jobs_reuse_backend_and_cancelled_jobs_are_replaced() {
    let folder = tempfile::tempdir().unwrap();
    let wav = folder.path().join("input.wav");
    let mut writer = hound::WavWriter::create(
        &wav,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    for _ in 0..32000 {
        writer.write_sample(500i16).unwrap();
    }
    writer.finalize().unwrap();
    let mut request = Request {
        mode: Mode::Mock,
        wav,
        ..Default::default()
    };
    let engine = Engine::default();
    let (logger, mut logs) =
        audio2face3d_gui::logging::channel(2048, 10000, audio2face3d::logging::LogLevel::Info);
    let first = collect(engine.start(1, request.clone(), logger.clone()).unwrap());
    request.pace_input = true;
    let second = collect(engine.start(2, request.clone(), logger.clone()).unwrap());
    compare(&first, &second);
    logs.drain();
    assert_eq!(
        logs.entries
            .iter()
            .filter(|e| e.record.message == "Initializing inference backend")
            .count(),
        1
    );
    assert_eq!(
        logs.entries
            .iter()
            .filter(|e| e.record.message == "Reusing inference backend")
            .count(),
        1
    );
    let job = engine.start(3, request.clone(), logger.clone()).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    job.cancel();
    drop(job);
    request.pace_input = false;
    compare(
        &first,
        &collect(engine.start(4, request, logger.clone()).unwrap()),
    );
    logs.drain();
    assert_eq!(
        logs.entries
            .iter()
            .filter(|e| e.record.message == "Initializing inference backend")
            .count(),
        2
    );
}
#[cfg(feature = "local")]
#[test]
#[ignore = "requires A2F_MODEL, A2F_WAV, CUDA_PATH and TENSORRT_ROOT_DIR"]
fn native_model_reuse_matches_fresh_inference() {
    let request = Request {
        mode: Mode::Local,
        wav: std::env::var_os("A2F_WAV").expect("A2F_WAV").into(),
        model: std::env::var_os("A2F_MODEL").expect("A2F_MODEL").into(),
        cuda_root: std::env::var_os("CUDA_PATH").expect("CUDA_PATH").into(),
        tensorrt_root: std::env::var_os("TENSORRT_ROOT_DIR")
            .expect("TENSORRT_ROOT_DIR")
            .into(),
        ..Default::default()
    };
    let engine = Engine::default();
    let (logger, mut logs) =
        audio2face3d_gui::logging::channel(4096, 10000, audio2face3d::logging::LogLevel::Info);
    let start = Instant::now();
    let first = collect(engine.start(1, request.clone(), logger.clone()).unwrap());
    println!("First request: {:?}", start.elapsed());
    let start = Instant::now();
    let second = collect(engine.start(2, request.clone(), logger.clone()).unwrap());
    println!("Reused request: {:?}", start.elapsed());
    compare(&first, &second);
    logs.drain();
    assert_eq!(
        logs.entries
            .iter()
            .filter(|e| e.record.message == "regression model and host solver ready")
            .count(),
        1
    );
    assert_eq!(
        logs.entries
            .iter()
            .filter(|e| e.record.message == "Reusing loaded regression model")
            .count(),
        1
    );
    drop(engine);
    let fresh = collect(
        Job::start(
            3,
            request,
            std::sync::Arc::new(audio2face3d::logging::NoopLogger),
        )
        .unwrap(),
    );
    compare(&second, &fresh);
}
