#![cfg(feature = "mock")]
use audio2face3d::{
    Audio2Face3DContext,
    client::{Client, DirectConfig, ServerConfig},
    inference::{BackendKind as EngineKind, Config as EngineConfig},
    logging::{LogLevel, Logger},
    types::*,
};
use audio2face3d_server::{
    Server,
    config::{BackendKind, Config},
};
use std::sync::{Arc, Mutex};
#[derive(Default)]
struct Sink(Mutex<Vec<String>>);
impl Logger for Sink {
    fn log_level(&self) -> LogLevel {
        LogLevel::Info
    }
    fn write_log(&self, _: LogLevel, message: String) {
        self.0.lock().unwrap().push(message);
    }
}
fn context(sink: Arc<Sink>) -> Audio2Face3DContext {
    Audio2Face3DContext::builder().logger(sink).build()
}
async fn utterance(client: &Client) {
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    input
        .send(InputChunk::new(
            PcmBuffer::from_vec(vec![0; 3200]).unwrap(),
            vec![],
        ))
        .await
        .unwrap();
    input.finish().await.unwrap();
    let mut completed = false;
    while let Some(event) = output.recv().await.unwrap() {
        completed |= matches!(event, OutputEvent::Completed(_));
    }
    assert!(completed);
    control.closed().await.unwrap();
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_servers_remote_clients_and_direct_have_separate_contexts() {
    let mut running = vec![];
    let mut clients = vec![];
    let mut sinks = vec![];
    let mut addresses = vec![];
    for _ in 0..2 {
        let sink = Arc::new(Sink::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = Server::builder(Config {
            backend: BackendKind::Mock,
            ..Default::default()
        })
        .context(context(sink.clone()))
        .build()
        .unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(server.serve(listener, async {
            let _ = rx.await;
        }));
        let remote_sink = Arc::new(Sink::default());
        let client = Client::server_with_context(
            ServerConfig::new(format!("http://{addr}")),
            context(remote_sink.clone()),
        )
        .await
        .unwrap();
        utterance(&client).await;
        assert!(
            remote_sink
                .0
                .lock()
                .unwrap()
                .iter()
                .any(|s| s.contains("remote inference client connected"))
        );
        clients.push(client);
        running.push((tx, handle));
        sinks.push(sink);
        addresses.push(addr.to_string());
    }
    let direct_sink = Arc::new(Sink::default());
    let direct = Client::direct_with_context(
        DirectConfig {
            engine: EngineConfig {
                backend: EngineKind::Mock,
                ..Default::default()
            },
            ..Default::default()
        },
        context(direct_sink.clone()),
    )
    .await
    .unwrap();
    utterance(&direct).await;
    direct.shutdown().await.unwrap();
    for client in clients {
        client.shutdown().await.unwrap();
    }
    for (tx, handle) in running {
        tx.send(()).unwrap();
        handle.await.unwrap().unwrap();
    }
    for i in 0..2 {
        let lines = sinks[i].0.lock().unwrap();
        assert!(lines.iter().any(|s| s.contains(&addresses[i])));
        assert!(!lines.iter().any(|s| s.contains(&addresses[1 - i])));
    }
    assert!(
        direct_sink
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|s| s.contains("inference prepared"))
    );
}
