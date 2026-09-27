#![cfg(all(feature = "session", any(feature = "mock", feature = "local")))]
use audio2face3d_gui::{
    core::{Clip, SessionState},
    inference::{Engine, Event, Job, Mode, Request, apply_event},
};
use std::time::{Duration, Instant};

fn collect(mut job: Job) -> Clip {
    let mut clip = Clip::running();
    let start = Instant::now();
    let mut ready = false;
    loop {
        let mut events: Vec<_> = job.events.try_iter().collect();
        let finished = job.try_finish();
        if finished.is_some() {
            events.extend(job.events.try_iter());
        }
        for event in events {
            match event {
                Event::Output(event) => apply_event(&mut clip, event).unwrap(),
                Event::Ready(result) => clip = *result,
                Event::InputFinished => {}
            }
        }
        if !ready && clip.ready_until() >= 0.1 {
            ready = true;
            println!("Playback ready: {:?}", start.elapsed());
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
// Stop only after real output has arrived, while paced input is still active.
fn stop_during_playback(mut job: Job) {
    let start = Instant::now();
    let mut clip = Clip::running();
    loop {
        for event in job.events.try_iter() {
            match event {
                Event::Output(event) => apply_event(&mut clip, event).unwrap(),
                Event::Ready(result) => clip = *result,
                Event::InputFinished => {}
            }
        }
        assert!(job.try_finish().is_none(), "request finished before Stop");
        if clip.ready_until() >= 0.1 {
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(180));
        std::thread::sleep(Duration::from_millis(1));
    }
    job.cancel();
    loop {
        if let Some(result) = job.try_finish() {
            assert!(result.is_err());
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(180));
        std::thread::sleep(Duration::from_millis(1));
    }
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
fn completed_and_stopped_jobs_reuse_backend_but_errors_invalidate_it() {
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
    request.pace_input = true;
    let second = collect(engine.start(2, request.clone(), logger.clone()).unwrap());
    compare(&first, &second);
    let start = Instant::now();
    let third = collect(engine.start(5, request.clone(), logger.clone()).unwrap());
    println!("Repeated streaming request: {:?}", start.elapsed());
    compare(&second, &third);
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
        2
    );
    stop_during_playback(engine.start(3, request.clone(), logger.clone()).unwrap());
    request.pace_input = false;
    compare(
        &first,
        &collect(engine.start(4, request.clone(), logger.clone()).unwrap()),
    );
    logs.drain();
    assert_eq!(
        logs.entries
            .iter()
            .filter(|e| e.record.message == "Initializing inference backend")
            .count(),
        1
    );
    let mut invalid = request.clone();
    invalid.wav = folder.path().join("missing.wav");
    let mut job = engine.start(6, invalid, logger.clone()).unwrap();
    let start = Instant::now();
    loop {
        if let Some(result) = job.try_finish() {
            assert!(result.is_err());
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(5));
        std::thread::sleep(Duration::from_millis(1));
    }
    compare(
        &first,
        &collect(engine.start(7, request, logger.clone()).unwrap()),
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
    let mut request = Request {
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
    request.pace_input = true;
    let second = collect(engine.start(2, request.clone(), logger.clone()).unwrap());
    println!("Reused request: {:?}", start.elapsed());
    compare(&first, &second);
    let start = Instant::now();
    let third = collect(engine.start(5, request.clone(), logger.clone()).unwrap());
    println!("Repeated streaming request: {:?}", start.elapsed());
    compare(&second, &third);
    for id in 10..13 {
        stop_during_playback(engine.start(id, request.clone(), logger.clone()).unwrap());
        let start = Instant::now();
        let restarted = collect(
            engine
                .start(id + 10, request.clone(), logger.clone())
                .unwrap(),
        );
        println!("After Stop: {:?}", start.elapsed());
        compare(&first, &restarted);
    }
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
        8
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

#[cfg(all(feature = "local", feature = "emotion"))]
#[test]
#[ignore = "requires A2F_MODEL, A2F_EMOTION_MODEL, A2F_WAV and CUDA/TensorRT"]
fn native_emotion_model_is_optional_and_reusable() {
    let mut request = Request {
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
    let without = collect(engine.start(1, request.clone(), logger.clone()).unwrap());
    assert!(without.emotions.is_empty());
    request.emotion.model = Some(
        std::env::var_os("A2F_EMOTION_MODEL")
            .expect("A2F_EMOTION_MODEL")
            .into(),
    );
    request.emotion.beginning.insert("joy".into(), 0.8);
    let first = collect(engine.start(2, request.clone(), logger.clone()).unwrap());
    assert!(!first.emotions.is_empty());
    let reused = collect(engine.start(3, request.clone(), logger.clone()).unwrap());
    compare(&first, &reused);
    assert_eq!(first.emotions, reused.emotions);
    stop_during_playback(
        engine
            .start(
                4,
                Request {
                    pace_input: true,
                    ..request.clone()
                },
                logger.clone(),
            )
            .unwrap(),
    );
    let after_stop = collect(engine.start(5, request.clone(), logger.clone()).unwrap());
    compare(&first, &after_stop);
    assert_eq!(first.emotions, after_stop.emotions);
    request.emotion.model = None;
    let disabled = collect(engine.start(6, request, logger.clone()).unwrap());
    compare(&without, &disabled);
    logs.drain();
    assert_eq!(
        logs.entries
            .iter()
            .filter(|e| e.record.message == "Initializing inference backend")
            .count(),
        3
    );
}
