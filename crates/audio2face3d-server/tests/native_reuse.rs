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
    let options = RequestOptions::builder(AudioFormat::MONO_16KHZ)
        .timeout(Duration::from_secs(180))
        .build()
        .unwrap();
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
    wait(client.shutdown()).unwrap();
    stop.send(()).unwrap();
    runtime.block_on(task).unwrap().unwrap();
    assert_eq!(logs.loaded.load(Ordering::Relaxed), 1);
    assert_eq!(logs.reused.load(Ordering::Relaxed), 3);
}
