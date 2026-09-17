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
    task: tokio::task::JoinHandle<Result<(), ServerError>>,
}
async fn start<A: Authenticator>(auth: Option<A>) -> Running {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let server = Server::builder(Config {
        backend: BackendKind::Mock,
        max_streams: 1,
        ..Default::default()
    })
    .authentication(auth)
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
        tokio::time::timeout(Duration::from_secs(3), self.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
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
        Server::builder(Config {
            backend: BackendKind::Mock,
            ..Default::default()
        })
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
