#![cfg(feature = "mock")]
use audio2face3d::{
    Audio2Face3DContext,
    client::{Client, ServerConfig, types::*},
    logging::{LogLevel, Logger},
};
use audio2face3d_server::{
    Server, ShutdownReport,
    auth::{AuthError, AuthRequest, Principal},
    config::{BackendKind, Config},
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use std::time::Duration;

const KEY: &str = "ClientE2ESentinel-._~+/==";
#[derive(Default)]
struct Logs(Mutex<Vec<String>>);
impl Logger for Logs {
    fn log_level(&self) -> LogLevel {
        LogLevel::Trace
    }
    fn write_log(&self, _: LogLevel, message: String) {
        self.0.lock().unwrap().push(message);
    }
}
struct Fixture {
    url: String,
    enabled: Arc<AtomicBool>,
    calls: Arc<AtomicUsize>,
    logs: Arc<Logs>,
    stop: tokio::sync::oneshot::Sender<()>,
    serving: tokio::task::JoinHandle<
        std::result::Result<ShutdownReport, audio2face3d_server::ServerError>,
    >,
}
impl Fixture {
    async fn new(auth: bool) -> Self {
        let enabled = Arc::new(AtomicBool::new(true));
        let calls = Arc::new(AtomicUsize::new(0));
        let logs = Arc::new(Logs::default());
        let context = Audio2Face3DContext::builder().logger(logs.clone()).build();
        let (allow, counter) = (enabled.clone(), calls.clone());
        let verifier = move |request: AuthRequest<'_>| {
            counter.fetch_add(1, Ordering::SeqCst);
            if request.api_key.expose() == KEY && allow.load(Ordering::SeqCst) {
                Principal::new("e2e-client")
            } else {
                Err(AuthError::InvalidCredential)
            }
        };
        let server = Server::builder(
            Config::builder(BackendKind::Mock)
                .max_streams(1)
                .build()
                .unwrap(),
        )
        .context(context)
        .authentication(auth.then_some(verifier))
        .build()
        .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (stop, stopped) = tokio::sync::oneshot::channel();
        let serving = tokio::spawn(server.serve(listener, async {
            let _ = stopped.await;
        }));
        Self {
            url,
            enabled,
            calls,
            logs,
            stop,
            serving,
        }
    }
    async fn client(&self, key: Option<&str>) -> Client {
        let config = ServerConfig::builder(&self.url)
            .optional_api_key(key.map(str::to_owned))
            .build()
            .unwrap();
        let debug = format!("{config:?}");
        assert!(!debug.contains(KEY));
        assert_eq!(config.clone().api_key(), config.api_key());
        Client::server_with_context(
            config,
            Audio2Face3DContext::builder()
                .logger(self.logs.clone())
                .build(),
        )
        .await
        .unwrap()
    }
    async fn close(self, requests: usize, rejected: usize) {
        self.stop.send(()).unwrap();
        let report = tokio::time::timeout(Duration::from_secs(5), self.serving)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(report.inference_requests as usize, requests);
        assert_eq!(report.authentication_rejections as usize, rejected);
        assert_eq!(
            report.inference_workers_started,
            report.inference_workers_finished
        );
        assert!(!self.logs.0.lock().unwrap().join("\n").contains(KEY));
    }
}
async fn success(client: &Client) {
    let (mut input, mut output, control) = client
        .start(
            RequestOptions::builder(AudioFormat::MONO_16KHZ)
                .build()
                .unwrap(),
        )
        .unwrap()
        .split();
    let pcm: Vec<_> = (0i16..1600).flat_map(i16::to_le_bytes).collect();
    input
        .send(InputChunk::new(
            PcmBuffer::from_vec(pcm.clone()).unwrap(),
            vec![],
        ))
        .await
        .unwrap();
    input.finish().await.unwrap();
    let mut returned = vec![];
    let mut frames = 0;
    let mut complete = false;
    let mut finished = false;
    while let Some(event) = output.recv().await.unwrap() {
        match event {
            OutputEvent::Audio(audio) => returned.extend_from_slice(audio.pcm().as_bytes()),
            OutputEvent::Curves(curves) => {
                assert_eq!(curves.values().len(), 52);
                frames += 1;
            }
            OutputEvent::ProcessingFinished => finished = true,
            OutputEvent::Completed(_) => complete = true,
            _ => (),
        }
    }
    control.closed().await.unwrap();
    assert_eq!(returned, pcm);
    assert!(frames > 0 && finished && complete);
}
async fn rejected(client: &Client, message: &str) {
    // Leave input open: authentication rejection must not wait for audio.
    let (_input, mut output, control) = client
        .start(
            RequestOptions::builder(AudioFormat::MONO_16KHZ)
                .build()
                .unwrap(),
        )
        .unwrap()
        .split();
    let error = output.recv().await.unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert!(error.message().contains(message));
    assert!(!format!("{error:?}").contains(KEY));
    assert!(control.closed().await.is_err());
}
#[tokio::test]
async fn common_client_authentication_over_real_grpc_recovers_on_same_client() {
    tokio::time::timeout(Duration::from_secs(15), async {
        let fixture = Fixture::new(true).await;
        let missing = fixture.client(None).await;
        rejected(&missing, "missing or invalid bearer credential").await;
        missing.shutdown().await.unwrap();
        let wrong = fixture.client(Some("wrong-key")).await;
        rejected(&wrong, "invalid credential").await;
        wrong.shutdown().await.unwrap();
        let client = fixture.client(Some(KEY)).await;
        success(&client).await;
        fixture.enabled.store(false, Ordering::SeqCst);
        rejected(&client, "invalid credential").await;
        fixture.enabled.store(true, Ordering::SeqCst);
        success(&client).await;
        success(&client.clone()).await;
        client.shutdown().await.unwrap();
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 5);
        fixture.close(6, 3).await;
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn common_client_without_api_key_works_with_authentication_disabled() {
    tokio::time::timeout(Duration::from_secs(10), async {
        let fixture = Fixture::new(false).await;
        let client = fixture.client(None).await;
        success(&client).await;
        client.shutdown().await.unwrap();
        assert_eq!(fixture.calls.load(Ordering::SeqCst), 0);
        fixture.close(1, 0).await;
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn invalid_client_api_keys_fail_before_connecting_and_never_echo_values() {
    for key in [
        "".to_owned(),
        " ".to_owned(),
        "a\r\nb".to_owned(),
        "日本".to_owned(),
        "a=b".to_owned(),
        "=".to_owned(),
        "a b".to_owned(),
        "a".repeat(4097),
    ] {
        let error = ServerConfig::builder("http://127.0.0.1:1")
            .api_key(key)
            .build()
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidInput);
        assert_eq!(error.message(), "invalid API key format");
    }
}
