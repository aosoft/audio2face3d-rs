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
struct Sink(Mutex<Vec<audio2face3d::logging::LogRecord>>);
impl Logger for Sink {
    fn log_level(&self) -> LogLevel {
        LogLevel::Debug
    }
    fn write_log(&self, _: LogLevel, message: audio2face3d::logging::LogRecord) {
        self.0.lock().unwrap().push(message);
    }
}
fn context(sink: Arc<Sink>) -> Audio2Face3DContext {
    Audio2Face3DContext::builder().logger(sink).build()
}
async fn utterance(client: &Client) {
    let (mut input, mut output, control) = client
        .start(
            RequestOptions::builder(AudioFormat::MONO_16KHZ)
                .build()
                .unwrap(),
        )
        .unwrap()
        .split();
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
    for active in 1..=2 {
        let sink = Arc::new(Sink::default());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = Server::builder(
            Config::builder(BackendKind::Mock)
                .max_streams(active)
                .build()
                .unwrap(),
        )
        .context(context(sink.clone()))
        .build()
        .unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(server.serve(listener, async {
            let _ = rx.await;
        }));
        let remote_sink = Arc::new(Sink::default());
        let client = Client::server_with_context(
            ServerConfig::builder(format!("http://{addr}"))
                .build()
                .unwrap(),
            context(remote_sink.clone()),
        )
        .await
        .unwrap();
        tokio::join!(utterance(&client), utterance(&client));
        assert!(remote_sink.0.lock().unwrap().iter().any(|s| {
            format!("{} {:?}", s.message, s.fields).contains("remote connection finished")
        }));
        clients.push(client);
        running.push((tx, handle));
        sinks.push(sink);
        addresses.push(addr.to_string());
    }
    let direct_sink = Arc::new(Sink::default());
    let direct = Client::direct_with_context(
        DirectConfig::builder(EngineConfig::builder(EngineKind::Mock).build().unwrap())
            .build()
            .unwrap(),
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
        let mut completed = std::collections::BTreeSet::new();
        for line in lines.iter().filter(|r| r.message == "completed") {
            let id = line
                .fields
                .iter()
                .find(|(k, _)| k == "rpc_id")
                .expect("RPC field");
            let audio2face3d::logging::LogValue::U64(id) = id.1 else {
                panic!("typed RPC ID");
            };
            assert!(completed.insert(id), "duplicate completion");
            assert!(lines.iter().any(|r| {
                r.message == "received"
                    && r.fields.iter().any(|(k, v)| {
                        k == "rpc_id" && *v == audio2face3d::logging::LogValue::U64(id)
                    })
            }));
        }
        assert_eq!(completed.len(), 2);

        assert!(
            lines
                .iter()
                .any(|s| format!("{} {:?}", s.message, s.fields).contains(&addresses[i]))
        );
        assert!(
            !lines
                .iter()
                .any(|s| format!("{} {:?}", s.message, s.fields).contains(&addresses[1 - i]))
        );
    }
    assert!(direct_sink.0.lock().unwrap().iter().any(|s| {
        format!("{} {:?}", s.message, s.fields).contains("inference preparation finished")
    }));
}
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn queued_request_keeps_id_after_active_cancellation() {
    let sink = Arc::new(Sink::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = Server::builder(
        Config::builder(BackendKind::Mock)
            .max_streams(1)
            .build()
            .unwrap(),
    )
    .context(context(sink.clone()))
    .build()
    .unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let handle = tokio::spawn(server.serve(listener, async {
        let _ = rx.await;
    }));
    let client = Client::server(
        ServerConfig::builder(format!("http://{addr}"))
            .build()
            .unwrap(),
    )
    .await
    .unwrap();
    let options = || {
        RequestOptions::builder(AudioFormat::MONO_16KHZ)
            .build()
            .unwrap()
    };
    let (mut input1, output1, control1) = client.start(options()).unwrap().split();
    input1
        .send(InputChunk::new(
            PcmBuffer::from_vec(vec![0; 320]).unwrap(),
            vec![],
        ))
        .await
        .unwrap();
    async fn until(sink: &Sink, message: &str, count: usize) {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if sink
                    .0
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|r| r.message == message)
                    .count()
                    >= count
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
    }
    until(&sink, "started", 1).await;
    let (mut input2, mut output2, control2) = client.start(options()).unwrap().split();
    input2
        .send(InputChunk::new(
            PcmBuffer::from_vec(vec![0; 320]).unwrap(),
            vec![],
        ))
        .await
        .unwrap();
    input2.finish().await.unwrap();
    until(&sink, "waiting for execution slot", 2).await;
    control1.cancel();
    drop(output1);
    drop(input1);
    let mut completed = false;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while let Some(event) = output2.recv().await.unwrap() {
            completed |= matches!(event, OutputEvent::Completed(_));
        }
    })
    .await
    .unwrap();
    assert!(completed);
    control2.closed().await.unwrap();
    client.shutdown().await.unwrap();
    tx.send(()).unwrap();
    handle.await.unwrap().unwrap();
    let records = sink.0.lock().unwrap();
    let id = |r: &audio2face3d::logging::LogRecord| {
        r.fields
            .iter()
            .find_map(|(k, v)| if k == "rpc_id" { Some(v.clone()) } else { None })
            .unwrap()
    };
    let started: Vec<_> = records
        .iter()
        .filter(|r| r.message == "started")
        .map(id)
        .collect();
    assert_eq!(started.len(), 2);
    assert_ne!(started[0], started[1]);
    assert!(
        records
            .iter()
            .any(|r| r.message == "completed" && id(r) == started[1])
    );
    assert!(records.iter().any(|r| (r.message == "failed"
        || r.message.contains("response stream dropped"))
        && id(r) == started[0]));
}

#[tokio::test]
async fn early_rejections_and_input_timeout_have_exactly_one_terminal_record() {
    use audio2face3d::logging::LogValue;
    use audio2face3d_server::{
        auth::{AuthRequest, Principal},
        proto::{
            controller::AudioStream,
            nvidia_ace::services::a2f_controller::v1::a2f_controller_service_client::A2fControllerServiceClient,
        },
    };
    let sink = Arc::new(Sink::default());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = Server::builder(
        Config::builder(BackendKind::Mock)
            .input_idle_timeout_ms(50)
            .build()
            .unwrap(),
    )
    .context(context(sink.clone()))
    .authentication(Some(|_: AuthRequest<'_>| Principal::new("test")))
    .build()
    .unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(server.serve(listener, async {
        let _ = rx.await;
    }));
    let mut client = A2fControllerServiceClient::connect(format!("http://{addr}"))
        .await
        .unwrap();
    for case in 0..4 {
        let (input, stream) = tokio::sync::mpsc::channel::<AudioStream>(1);
        if case != 2 {
            input.send(AudioStream { stream_part: None }).await.unwrap();
        }
        let mut request = tonic::Request::new(tokio_stream::wrappers::ReceiverStream::new(stream));
        request.metadata_mut().insert(
            "authorization",
            if case == 0 {
                "Basic SentinelSecret"
            } else {
                "Bearer SentinelSecret"
            }
            .parse()
            .unwrap(),
        );
        if case == 3 {
            request
                .metadata_mut()
                .insert("grpc-timeout", "invalid".parse().unwrap());
        }
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.process_audio_stream(request),
        )
        .await
        .unwrap();
        assert!(result.is_err());
        drop(input);
    }
    let remote = Client::server(
        ServerConfig::builder(format!("http://{addr}"))
            .api_key("SentinelSecret")
            .build()
            .unwrap(),
    )
    .await
    .unwrap();
    utterance(&remote).await;
    remote.shutdown().await.unwrap();
    tx.send(()).unwrap();
    task.await.unwrap().unwrap();
    let logs = sink.0.lock().unwrap();
    let terminal: Vec<_> = logs
        .iter()
        .filter(|r| {
            r.fields.iter().any(|(k, _)| k == "rpc_id")
                && r.fields.iter().any(|(k, _)| k == "outcome")
        })
        .collect();
    assert_eq!(terminal.len(), 5);
    let mut ids = std::collections::HashSet::new();
    for record in &terminal {
        let (_, LogValue::U64(id)) = record.fields.iter().find(|(k, _)| k == "rpc_id").unwrap()
        else {
            panic!("typed id")
        };
        assert!(ids.insert(*id));
        assert!(record.fields.iter().any(|(k, _)| k == "elapsed_us"));
    }
    for stage in ["authentication", "input_header", "metadata"] {
        assert!(
            terminal
                .iter()
                .any(|r| r.fields.contains(&("stage".into(), stage.into())))
        );
    }
    assert!(
        terminal
            .iter()
            .any(|r| r.fields.contains(&("outcome".into(), "timeout".into())))
    );
    let success = terminal.iter().find(|r| r.message == "completed").unwrap();
    assert!(
        success
            .fields
            .contains(&("input_audio_bytes".into(), LogValue::U64(3200)))
    );
    assert!(
        success
            .fields
            .iter()
            .any(|(k, v)| k == "output_batches_enqueued" && matches!(v,LogValue::U64(n) if *n>0))
    );
    assert_eq!(
        logs.iter()
            .filter(|r| r.message == "closing inference request")
            .count(),
        1
    );
    assert!(!format!("{logs:?}").contains("SentinelSecret"));
}
