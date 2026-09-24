#![cfg(feature = "mock")]
use audio2face3d_gui::{
    core::{Clip, SessionState},
    inference::{Event, Job, Mode, Request, apply_event},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
static SEQUENCE: AtomicU64 = AtomicU64::new(0);
#[cfg(feature = "grpc")]
#[test]
fn unreachable_server_finishes_with_error_and_releases_worker() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut request = request();
    let path = request.wav.clone();
    request.mode = Mode::Grpc;
    request.endpoint = format!("http://{address}");
    let mut job = Job::start(4, request, Arc::new(audio2face3d::logging::NoopLogger)).unwrap();
    let start = Instant::now();
    loop {
        if let Some(result) = job.try_finish() {
            assert!(result.is_err());
            break;
        }
        assert!(start.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(job.events.try_iter().all(|event| !matches!(
        event,
        Event::Output(audio2face3d::types::OutputEvent::Completed(_))
    )));
    drop(job);
    std::fs::remove_file(path).unwrap();
}
fn request() -> Request {
    let path = std::env::temp_dir().join(format!(
        "a2f-job-{}-{}.wav",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut writer = hound::WavWriter::create(
        &path,
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
    Request {
        mode: Mode::Mock,
        wav: path,
        ..Default::default()
    }
}
#[test]
fn repeated_jobs_finish_with_audio_curves_and_released_resources() {
    let request = request();
    for id in 1..=2 {
        let mut job = Job::start(
            id,
            request.clone(),
            Arc::new(audio2face3d::logging::NoopLogger),
        )
        .unwrap();
        let mut clip = Clip::running();
        let start = Instant::now();
        loop {
            let mut events: Vec<_> = job.events.try_iter().collect();
            let done = job.try_finish();
            if done.is_some() {
                events.extend(job.events.try_iter());
            }
            for event in events {
                if let Event::Output(event) = event {
                    apply_event(&mut clip, event).unwrap();
                }
            }
            if let Some(result) = done {
                result.unwrap();
                break;
            }
            assert!(start.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(job.id, id);
        assert_eq!(clip.session, SessionState::Completed);
        assert_eq!(clip.audio.len(), 32000);
        assert_eq!(clip.names.len(), 52);
        assert!(clip.frames.len() >= 59);
    }
    std::fs::remove_file(request.wav).unwrap();
}
#[test]
fn cancellation_releases_a_worker_with_an_unconsumed_bounded_queue() {
    let request = request();
    let path = request.wav.clone();
    let job = Job::start(3, request, Arc::new(audio2face3d::logging::NoopLogger)).unwrap();
    std::thread::sleep(Duration::from_millis(100));
    let start = Instant::now();
    job.cancel();
    drop(job);
    assert!(start.elapsed() < Duration::from_secs(3));
    std::fs::remove_file(path).unwrap();
}
