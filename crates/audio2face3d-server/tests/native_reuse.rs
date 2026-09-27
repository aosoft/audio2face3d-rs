#![cfg(feature = "native")]
mod support;
use audio2face3d::{
    Audio2Face3DContext,
    client::{Client, ServerConfig, types::*},
    logging::{LogLevel, LogRecord, Logger},
    runtime::NativeRuntimeConfig,
};
use audio2face3d_server::{
    Server,
    config::{BackendKind, Config},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};
use support::*;
#[derive(Default)]
struct Loads {
    loaded: AtomicUsize,
    reused: AtomicUsize,
}
impl Logger for Loads {
    fn log_level(&self) -> LogLevel {
        LogLevel::Info
    }
    fn write_log(&self, _: LogLevel, record: LogRecord) {
        match record.message.as_str() {
            "regression model and host solver ready" => {
                self.loaded.fetch_add(1, Ordering::Relaxed);
            }
            "Reusing loaded regression model" => {
                self.reused.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}
fn curves(events: &[OutputEvent]) -> Vec<(u64, Vec<f32>)> {
    events
        .iter()
        .filter_map(|e| match e {
            OutputEvent::Curves(frame) => Some((frame.time().as_nanos(), frame.values().to_vec())),
            _ => None,
        })
        .collect()
}
#[test]
#[ignore = "requires A2F_MODEL, CUDA_PATH and TENSORRT_ROOT_DIR"]
fn grpc_reuses_model_across_requests_and_client_cancellation() {
    run_reuse(false);
    run_reuse(true);
}

fn run_reuse(custom: bool) {
    let logs = Arc::new(Loads::default());
    let context = Audio2Face3DContext::builder()
        .logger(logs.clone())
        .native_runtime(
            NativeRuntimeConfig::builder()
                .cuda_root(std::env::var_os("CUDA_PATH").expect("CUDA_PATH"))
                .tensorrt_root(std::env::var_os("TENSORRT_ROOT_DIR").expect("TENSORRT_ROOT_DIR"))
                .build()
                .unwrap(),
        )
        .build();
    let server = Server::builder(
        Config::builder(BackendKind::Regression)
            .model(std::env::var_os("A2F_MODEL").expect("A2F_MODEL"))
            .build()
            .unwrap(),
    )
    .context(context)
    .build()
    .unwrap();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let task = runtime.spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let client = wait(Client::server(
        ServerConfig::builder(endpoint)
            .runtime(runtime.handle().clone())
            .connect_timeout(Duration::from_secs(180))
            .build()
            .unwrap(),
    ))
    .unwrap();
    let mut builder =
        RequestOptions::builder(AudioFormat::MONO_16KHZ).timeout(Duration::from_secs(180));
    if custom {
        let mut face = FaceParameters::default();
        face.lower_face_strength = Some(0.7);
        let mut blendshapes = BlendshapeParameters::default();
        blendshapes.multipliers.insert("JawOpen".into(), 0.6);
        let mut emotion = EmotionParameters::default();
        emotion.transition_time = Some(0.4);
        let mut post = EmotionPostProcessing::default();
        post.contrast = Some(1.0);
        post.smoothing = Some(0.7);
        builder = builder
            .face(face)
            .blendshapes(blendshapes)
            .emotion(emotion)
            .emotion_post_processing(post);
    }
    let options = builder.build().unwrap();
    let bytes = pcm(31, 16000);
    let first = collect(&client, options.clone(), bytes.clone()).unwrap();
    for iteration in 0..2 {
        if iteration == 1 {
            let (mut input, mut output, control) = client.start(options.clone()).unwrap().split();
            wait(input.send(InputChunk::new(
                PcmBuffer::from_vec(bytes.clone()).unwrap(),
                vec![],
            )))
            .unwrap();
            loop {
                if matches!(wait(output.recv()).unwrap(), Some(OutputEvent::Audio(_))) {
                    break;
                }
            }
            control.cancel();
            assert!(wait(control.closed()).is_err());
        }
        let start = Instant::now();
        let next = collect(&client, options.clone(), bytes.clone()).unwrap();
        println!(
            "gRPC request after {}: {:?}",
            if iteration == 0 { "completion" } else { "Stop" },
            start.elapsed()
        );
        assert_eq!(returned_pcm(&first), returned_pcm(&next));
        let a = curves(&first);
        let b = curves(&next);
        assert!(!a.is_empty());
        assert_eq!(a.len(), b.len());
        for ((ta, va), (tb, vb)) in a.iter().zip(&b) {
            assert_eq!(ta, tb);
            for (a, b) in va.iter().zip(vb) {
                assert!((a - b).abs() < 1e-4);
            }
        }
    }
    assert_eq!(
        logs.loaded.load(Ordering::Relaxed),
        if custom { 2 } else { 1 }
    );
    assert_eq!(logs.reused.load(Ordering::Relaxed), 3);
    if custom {
        // Changed settings must not reuse stale solver settings. Returning to the
        // original settings must reload and still match the first fresh result.
        let changed = RequestOptions::builder(AudioFormat::MONO_16KHZ)
            .timeout(Duration::from_secs(180))
            .build()
            .unwrap();
        let other = collect(&client, changed, bytes.clone()).unwrap();
        assert_eq!(logs.loaded.load(Ordering::Relaxed), 3);
        assert!(
            curves(&first).iter().zip(curves(&other)).any(|(a, b)| a
                .1
                .iter()
                .zip(b.1)
                .any(|(a, b)| (a - b).abs() > 1e-4))
        );
        let fresh = collect(&client, options, bytes).unwrap();
        assert_eq!(logs.loaded.load(Ordering::Relaxed), 4);
        for (a, b) in curves(&first).iter().zip(curves(&fresh)) {
            assert_eq!(a.0, b.0);
            for (a, b) in a.1.iter().zip(b.1) {
                assert!((a - b).abs() < 1e-4);
            }
        }
    }
    wait(client.shutdown()).unwrap();
    stop.send(()).unwrap();
    runtime.block_on(task).unwrap().unwrap();
    assert_eq!(logs.reused.load(Ordering::Relaxed), 3);
}
