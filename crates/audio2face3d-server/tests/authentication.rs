#![cfg(feature = "mock")]
use audio2face3d_server::proto::{
    self,
    controller::{
        AudioStream, AudioStreamHeader,
        audio_stream::{EndOfAudio, StreamPart},
    },
    nvidia_ace::services::a2f_controller::v1::a2f_controller_service_client::A2fControllerServiceClient,
};
use audio2face3d_server::{
    Server, ServerError,
    auth::*,
    config::{BackendKind, Config},
};
use std::{
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tonic::{Code, Request, transport::Channel};
struct Running {
    address: SocketAddr,
    client: A2fControllerServiceClient<Channel>,
    stop: tokio::sync::oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Result<audio2face3d_server::ShutdownReport, ServerError>>,
}
async fn start<A: Authenticator>(auth: Option<A>) -> Running {
    start_policy(auth, audio2face3d_server::HealthAuth::Public).await
}
async fn start_policy<A: Authenticator>(
    auth: Option<A>,
    policy: audio2face3d_server::HealthAuth,
) -> Running {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = Server::builder(
        Config::builder(BackendKind::Mock)
            .max_streams(1)
            .build()
            .unwrap(),
    )
    .authentication(auth)
    .health_auth(policy)
    .build()
    .unwrap();
    let task = tokio::spawn(server.serve(listener, async {
        let _ = stopped.await;
    }));
    let client = A2fControllerServiceClient::connect(format!("http://{address}"))
        .await
        .unwrap();
    Running {
        address,
        client,
        stop,
        task,
    }
}
impl Running {
    async fn close(self) {
        self.stop.send(()).unwrap();
        let report = tokio::time::timeout(Duration::from_secs(3), self.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(
            report.inference_workers_started,
            report.inference_workers_finished
        );
    }
}
fn request<S>(stream: S, key: Option<&str>) -> Request<S> {
    let mut r = Request::new(stream);
    if let Some(key) = key {
        r.metadata_mut()
            .insert("authorization", format!("Bearer {key}").parse().unwrap());
    }
    r
}
fn messages() -> Vec<AudioStream> {
    vec![
        AudioStream {
            stream_part: Some(StreamPart::AudioStreamHeader(AudioStreamHeader {
                audio_header: Some(proto::audio::AudioHeader {
                    audio_format: 0,
                    channel_count: 1,
                    samples_per_second: 16000,
                    bits_per_sample: 16,
                }),
                ..Default::default()
            })),
        },
        AudioStream {
            stream_part: Some(StreamPart::AudioWithEmotion(proto::a2f::AudioWithEmotion {
                audio_buffer: vec![0; 3200],
                emotions: vec![],
            })),
        },
        AudioStream {
            stream_part: Some(StreamPart::EndOfAudio(EndOfAudio {})),
        },
    ]
}
async fn success(client: &mut A2fControllerServiceClient<Channel>, key: Option<&str>) {
    let mut body = client
        .process_audio_stream(request(tokio_stream::iter(messages()), key))
        .await
        .unwrap()
        .into_inner();
    let mut completed = false;
    while let Some(m) = body.message().await.unwrap() {
        if let Some(proto::controller::animation_data_stream::StreamPart::Status(status)) =
            m.stream_part
        {
            assert_eq!(status.code, 0);
            completed = true;
        }
    }
    assert!(completed);
}
#[tokio::test]
async fn rejection_precedes_input_and_does_not_poison_next_request() {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let mut running = start(Some(move |r: AuthRequest<'_>| {
        counter.fetch_add(1, Ordering::SeqCst);
        assert_eq!(r.method, RpcMethod::ProcessAudioStream);
        assert!(r.peer_addr.is_some());
        match r.api_key.expose() {
            "accepted" => Principal::new("test"),
            "forbidden" => Err(AuthError::Forbidden),
            "unavailable" => Err(AuthError::Unavailable),
            "panic" => panic!("test verifier failure"),
            _ => Err(AuthError::InvalidCredential),
        }
    }))
    .await;
    for (key, code) in [
        (None, Code::Unauthenticated),
        (Some("wrong"), Code::Unauthenticated),
        (Some("forbidden"), Code::PermissionDenied),
        (Some("unavailable"), Code::Unavailable),
        (Some("panic"), Code::Internal),
    ] {
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            running
                .client
                .process_audio_stream(request(tokio_stream::pending::<AudioStream>(), key)),
        )
        .await
        .unwrap();
        assert_eq!(result.unwrap_err().code(), code); // no first audio message was required
        success(&mut running.client, Some("accepted")).await;
    }
    assert_eq!(calls.load(Ordering::SeqCst), 9);
    running.close().await;
}
struct DropCount(Arc<AtomicUsize>);
impl Drop for DropCount {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
async fn reached(counter: &AtomicUsize, n: usize) {
    tokio::time::timeout(Duration::from_secs(2), async {
        while counter.load(Ordering::SeqCst) < n {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
}
#[tokio::test]
async fn pending_auth_cancels_on_rst_tcp_deadline_and_stop_without_taking_inference_slot() {
    let entered = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let e = entered.clone();
    let d = dropped.clone();
    let mut running = start(Some(async_authenticator(move |r: AuthRequest<'_>| {
        let pending = r.api_key.expose() == "pending";
        let e = e.clone();
        let d = d.clone();
        async move {
            if pending {
                let _guard = DropCount(d);
                e.fetch_add(1, Ordering::SeqCst);
                std::future::pending::<()>().await;
            }
            Principal::new("test")
        }
    })))
    .await;
    let mut client = running.client.clone();
    let rpc = tokio::spawn(async move {
        client
            .process_audio_stream(request(
                tokio_stream::pending::<AudioStream>(),
                Some("pending"),
            ))
            .await
    });
    reached(&entered, 1).await;
    success(&mut running.client, Some("accepted")).await; // max_streams=1 is still available
    rpc.abort();
    let _ = rpc.await;
    reached(&dropped, 1).await;
    let relay_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let relay_addr = relay_listener.local_addr().unwrap();
    let address = running.address;
    let relay = tokio::spawn(async move {
        let (mut downstream, _) = relay_listener.accept().await.unwrap();
        let mut upstream = tokio::net::TcpStream::connect(address).await.unwrap();
        let _ = tokio::io::copy_bidirectional(&mut downstream, &mut upstream).await;
    });
    let mut client = A2fControllerServiceClient::connect(format!("http://{relay_addr}"))
        .await
        .unwrap();
    let rpc = tokio::spawn(async move {
        client
            .process_audio_stream(request(
                tokio_stream::pending::<AudioStream>(),
                Some("pending"),
            ))
            .await
    });
    reached(&entered, 2).await;
    relay.abort();
    let _ = relay.await;
    reached(&dropped, 2).await;
    let _ = rpc.await;
    let mut r = request(tokio_stream::pending::<AudioStream>(), Some("pending"));
    r.set_timeout(Duration::from_millis(60));
    assert!(running.client.process_audio_stream(r).await.is_err());
    reached(&dropped, 3).await;
    let mut client = running.client.clone();
    let rpc = tokio::spawn(async move {
        client
            .process_audio_stream(request(
                tokio_stream::pending::<AudioStream>(),
                Some("pending"),
            ))
            .await
    });
    reached(&entered, 4).await;
    running.close().await;
    reached(&dropped, 4).await;
    assert!(rpc.await.unwrap().is_err());
}
#[tokio::test]
async fn runtime_none_and_without_authentication_use_same_public_builder() {
    let enabled = std::hint::black_box(false);
    let auth = enabled.then_some(|_: AuthRequest<'_>| Err(AuthError::Forbidden));
    let mut running = start(auth).await;
    success(&mut running.client, None).await;
    running.close().await;
    assert!(
        Server::builder(Config::builder(BackendKind::Mock).build().unwrap())
            .authentication(Some(|_: AuthRequest<'_>| Err(AuthError::Forbidden)))
            .without_authentication()
            .build()
            .is_ok()
    );
}

#[tokio::test]
async fn slow_verifier_does_not_block_later_authorized_inference() {
    let release = Arc::new(tokio::sync::Notify::new());
    let entered = Arc::new(AtomicUsize::new(0));
    let ready = release.clone();
    let counter = entered.clone();
    let mut running = start(Some(async_authenticator(move |r: AuthRequest<'_>| {
        let slow = r.api_key.expose() == "slow";
        let release = ready.clone();
        let entered = counter.clone();
        async move {
            if slow {
                entered.fetch_add(1, Ordering::SeqCst);
                release.notified().await;
            }
            Principal::new("test")
        }
    })))
    .await;
    let mut first = running.client.clone();
    let pending = tokio::spawn(async move {
        success(&mut first, Some("slow")).await;
    });
    reached(&entered, 1).await;
    tokio::time::timeout(
        Duration::from_secs(1),
        success(&mut running.client, Some("fast")),
    )
    .await
    .unwrap();
    release.notify_one();
    pending.await.unwrap();
    running.close().await;
}

#[tokio::test]
async fn deadline_after_headers_and_unconsumed_response_releases_worker() {
    let mut running = start::<NoAuth>(None).await;
    let (tx, rx) = tokio::sync::mpsc::channel(2);
    tx.send(messages().remove(0)).await.unwrap();
    let mut r = request(tokio_stream::wrappers::ReceiverStream::new(rx), None);
    r.set_timeout(Duration::from_millis(60));
    let mut body = running
        .client
        .process_audio_stream(r)
        .await
        .unwrap()
        .into_inner();
    // Do not poll the response until after its deadline.
    tokio::time::sleep(Duration::from_millis(110)).await;
    let error = loop {
        match body.message().await {
            Err(e) => break e,
            Ok(Some(_)) => (),
            Ok(None) => panic!("expired request succeeded"),
        }
    };
    assert_eq!(error.code(), Code::DeadlineExceeded);
    tokio::time::timeout(Duration::from_secs(1), tx.closed())
        .await
        .unwrap();
    success(&mut running.client, None).await;
    running.close().await;
}
#[tokio::test]
async fn invalid_zero_and_queued_rpc_deadlines_are_distinct() {
    let mut running = start::<NoAuth>(None).await;
    for value in ["1s", "123456789S", "-1S"] {
        let mut r = request(tokio_stream::pending::<AudioStream>(), None);
        r.metadata_mut()
            .insert("grpc-timeout", value.parse().unwrap());
        let error = running.client.process_audio_stream(r).await.unwrap_err();
        assert_eq!(
            error.code(),
            if value == "0S" {
                Code::DeadlineExceeded
            } else {
                Code::InvalidArgument
            }
        );
    }
    let (tx, rx) = tokio::sync::mpsc::channel(1);
    tx.send(messages().remove(0)).await.unwrap();
    let active = running
        .client
        .process_audio_stream(tokio_stream::wrappers::ReceiverStream::new(rx))
        .await
        .unwrap();
    assert_eq!(raw_timeout_status(running.address, &["50m"]).await, "4");
    drop(active);
    drop(tx);
    success(&mut running.client, None).await;
    running.close().await;
}
#[tokio::test]
async fn health_authentication_watch_and_shutdown_share_policy() {
    use audio2face3d_server::HealthAuth;
    use tonic_health::pb::{HealthCheckRequest, health_client::HealthClient};
    assert!(
        Server::builder(Config::builder(BackendKind::Mock).build().unwrap())
            .health_auth(HealthAuth::SameAsInference)
            .build()
            .is_err()
    );
    for policy in [HealthAuth::Public, HealthAuth::SameAsInference] {
        let calls = Arc::new(AtomicUsize::new(0));
        let counter = calls.clone();
        let mut running = start_policy(
            Some(move |r: AuthRequest<'_>| {
                counter.fetch_add(1, Ordering::SeqCst);
                if r.api_key.expose() == "accepted" {
                    Principal::new("test")
                } else {
                    Err(AuthError::InvalidCredential)
                }
            }),
            policy,
        )
        .await;
        let channel =
            tonic::transport::Endpoint::from_shared(format!("http://{}", running.address))
                .unwrap()
                .connect()
                .await
                .unwrap();
        let mut health = HealthClient::new(channel);
        let req = || HealthCheckRequest {
            service: String::new(),
        };
        if policy == HealthAuth::SameAsInference {
            assert_eq!(
                health.check(req()).await.unwrap_err().code(),
                Code::Unauthenticated
            );
            assert_eq!(
                health
                    .watch(request(req(), Some("wrong")))
                    .await
                    .unwrap_err()
                    .code(),
                Code::Unauthenticated
            );
        } else {
            assert_eq!(health.check(req()).await.unwrap().into_inner().status, 1);
        }
        let key = if policy == HealthAuth::Public {
            None
        } else {
            Some("accepted")
        };
        let mut watches = vec![];
        for _ in 0..66 {
            let mut watch = health
                .watch(request(req(), key))
                .await
                .unwrap()
                .into_inner();
            assert_eq!(watch.message().await.unwrap().unwrap().status, 1);
            watches.push(watch);
        }
        // More watches than auth slots: each Watch released its auth permit.
        success(&mut running.client, Some("accepted")).await;
        assert_eq!(
            calls.load(Ordering::SeqCst),
            if policy == HealthAuth::Public { 1 } else { 68 }
        );
        running.close().await;
        for mut watch in watches {
            while watch.message().await.unwrap().is_some() {}
        }
    }
}

async fn raw_timeout_status(address: SocketAddr, values: &[&str]) -> String {
    let socket = tokio::net::TcpStream::connect(address).await.unwrap();
    let (mut client, connection) = h2::client::handshake(socket).await.unwrap();
    let connection = tokio::spawn(connection);
    let mut request = http::Request::builder().method("POST")
        .uri(format!("http://{address}/nvidia_ace.services.a2f_controller.v1.A2FControllerService/ProcessAudioStream"))
        .header("content-type","application/grpc").header("te","trailers");
    for v in values {
        request = request.header("grpc-timeout", *v);
    }
    let (response, _send) = client
        .send_request(request.body(()).unwrap(), false)
        .unwrap();
    let response = response.await.unwrap();
    let status = response
        .headers()
        .get("grpc-status")
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    connection.abort();
    status
}
#[tokio::test]
async fn raw_http2_zero_deadline_and_duplicate_timeout_return_wire_status() {
    let running = start::<NoAuth>(None).await;
    assert_eq!(raw_timeout_status(running.address, &["0S"]).await, "4");
    assert_eq!(
        raw_timeout_status(running.address, &["1S", "2S"]).await,
        "3"
    );
    running.close().await;
}
