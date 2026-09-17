#![cfg(feature = "server")]
mod support;
use audio2face3d_client::{types::*, *};
use audio2face3d_protocol::{
    convert,
    wire::{self, A2fControllerService, A2fControllerServiceServer, animation, controller},
};
use std::{
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use support::*;
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{
    Stream,
    wrappers::{ReceiverStream, TcpListenerStream},
};
use tonic::{Request, Response, Status, Streaming};
type Reply = controller::AnimationDataStream;
#[derive(Clone, Copy)]
enum Mode {
    PartialHold,
    ProcessingEarly,
    StatusBeforeHeader,
    Normal,
    EarlySuccess,
    NoStatus,
    NoHeader,
    Trailers,
    MidInput,
    AfterInput,
    Hold,
    AudioOnly,
    CurvesOnly,
    DuplicateHeader,
    DuplicateEvent,
    ErrorStatus,
    ReverseTime,
}
#[derive(Clone)]
struct Service {
    mode: Mode,
    calls: Arc<AtomicUsize>,
}
fn wrap(part: controller::animation_data_stream::StreamPart) -> Reply {
    Reply {
        stream_part: Some(part),
    }
}
fn success() -> Reply {
    wrap(controller::animation_data_stream::StreamPart::Status(
        wire::status::Status {
            code: 0,
            message: "done".into(),
        },
    ))
}
impl A2fControllerService for Service {
    type ProcessAudioStreamStream =
        Pin<Box<dyn Stream<Item = std::result::Result<Reply, Status>> + Send>>;
    fn process_audio_stream<'borrow, 'future>(
        &'borrow self,
        request: Request<Streaming<controller::AudioStream>>,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = std::result::Result<Response<Self::ProcessAudioStreamStream>, Status>,
                > + Send
                + 'future,
        >,
    >
    where
        'borrow: 'future,
        Self: 'future,
    {
        let mode = if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            self.mode
        } else {
            Mode::Normal
        };
        Box::pin(async move {
            let mut input = request.into_inner();
            let mut pcm = vec![];
            // Deliberately withhold the HTTP response until input arrives/completes.
            while let Some(message) = input.message().await? {
                match message.stream_part {
                    Some(controller::audio_stream::StreamPart::AudioWithEmotion(a)) => {
                        pcm.extend(a.audio_buffer);
                        if matches!(mode, Mode::MidInput) {
                            return Err(Status::unavailable("reset during upload"));
                        }
                    }
                    Some(controller::audio_stream::StreamPart::EndOfAudio(_)) => break,
                    _ => {}
                }
            }
            if matches!(mode, Mode::Hold) {
                let (tx, rx) = mpsc::channel(1);
                tokio::spawn(async move {
                    tx.closed().await;
                });
                return Ok(Response::new(
                    Box::pin(ReceiverStream::new(rx)) as Self::ProcessAudioStreamStream
                ));
            }
            use controller::animation_data_stream::StreamPart as Part;
            let header = wrap(Part::AnimationDataStreamHeader(
                controller::AnimationDataStreamHeader {
                    audio_header: if matches!(mode, Mode::CurvesOnly) {
                        None
                    } else {
                        Some(convert::encode_audio_format(AudioFormat::MONO_16KHZ).unwrap())
                    },
                    skel_animation_header: if matches!(mode, Mode::AudioOnly) {
                        None
                    } else {
                        Some(animation::SkelAnimationHeader {
                            blend_shapes: vec!["JawOpen".into()],
                            joints: vec![],
                        })
                    },
                    start_time_code_since_epoch: 0.0,
                },
            ));
            let data = wrap(Part::AnimationData(animation::AnimationData {
                audio: if matches!(mode, Mode::CurvesOnly) {
                    None
                } else {
                    Some(animation::AudioWithTimeCode {
                        time_code: 0.0,
                        audio_buffer: pcm,
                    })
                },
                skel_animation: if matches!(mode, Mode::AudioOnly) {
                    None
                } else {
                    Some(animation::SkelAnimation {
                        blend_shape_weights: vec![
                            animation::FloatArrayWithTimeCode {
                                time_code: 0.0,
                                values: vec![0.2],
                            },
                            animation::FloatArrayWithTimeCode {
                                time_code: if matches!(mode, Mode::ReverseTime) {
                                    0.0
                                } else {
                                    0.033333333
                                },
                                values: vec![0.7],
                            },
                        ],
                        ..Default::default()
                    })
                },
                ..Default::default()
            }));
            if matches!(mode, Mode::PartialHold) {
                let (tx, rx) = mpsc::channel(2);
                tx.send(Ok(header)).await.unwrap();
                tx.send(Ok(data)).await.unwrap();
                tokio::spawn(async move {
                    tx.closed().await;
                });
                return Ok(Response::new(
                    Box::pin(ReceiverStream::new(rx)) as Self::ProcessAudioStreamStream
                ));
            }
            let mut replies = vec![];
            if matches!(mode, Mode::StatusBeforeHeader) {
                replies.push(Ok(success()));
            }
            if !matches!(mode, Mode::NoHeader) {
                replies.push(Ok(header.clone()));
            }
            if matches!(mode, Mode::DuplicateHeader) {
                replies.push(Ok(header));
            }
            if matches!(mode, Mode::ProcessingEarly) {
                replies.push(Ok(wrap(Part::Event(controller::Event {
                    event_type: 0,
                    metadata: None,
                }))));
            }
            if matches!(mode, Mode::EarlySuccess) {
                replies.push(Ok(success()));
            }
            replies.push(Ok(data));
            if matches!(mode, Mode::AfterInput) {
                replies.push(Err(Status::unavailable("reset after partial response")));
            }
            if matches!(mode, Mode::DuplicateEvent) {
                for _ in 0..2 {
                    replies.push(Ok(wrap(Part::Event(controller::Event {
                        event_type: 0,
                        metadata: None,
                    }))));
                }
            }
            if matches!(mode, Mode::ErrorStatus) {
                replies.push(Ok(wrap(Part::Status(wire::status::Status {
                    code: 3,
                    message: "inference failed".into(),
                }))));
            }
            if !matches!(mode, Mode::NoStatus) {
                replies.push(Ok(success()));
            }
            if matches!(mode, Mode::Trailers) {
                replies.push(Err(Status::internal("non-OK trailers after SUCCESS")));
            }
            Ok(Response::new(
                Box::pin(tokio_stream::iter(replies)) as Self::ProcessAudioStreamStream
            ))
        })
    }
}
struct Fixture {
    url: String,
    stop: Option<oneshot::Sender<()>>,
    task: tokio::task::JoinHandle<()>,
}
impl Fixture {
    async fn start(mode: Mode) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = oneshot::channel();
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(A2fControllerServiceServer::new(Service {
                    mode,
                    calls: Arc::new(AtomicUsize::new(0)),
                }))
                .serve_with_incoming_shutdown(TcpListenerStream::new(listener), async {
                    let _ = rx.await;
                })
                .await
                .unwrap();
        });
        Self {
            url,
            stop: Some(tx),
            task,
        }
    }
    async fn close(mut self) {
        let _ = self.stop.take().unwrap().send(());
        self.task.await.unwrap();
    }
}
async fn collect_async(client: &Client) -> Result<Vec<OutputEvent>> {
    let (mut input, mut output, control) = client.start(RequestOptions::default())?.split();
    let send = async {
        for _ in 0..4 {
            input
                .send(InputChunk::new(
                    PcmBuffer::from_vec(vec![1, 2]).unwrap(),
                    vec![],
                ))
                .await?;
        }
        input.finish().await
    };
    let recv = async {
        let mut events = vec![];
        while let Some(e) = output.recv().await? {
            events.push(e);
        }
        Ok(events)
    };
    let result = tokio::try_join!(send, recv);
    if result.is_err() {
        control.cancel();
    }
    let closed = control.closed().await;
    result.and_then(|(_, events)| {
        closed?;
        Ok(events)
    })
}
#[tokio::test]
async fn current_thread_runtime_validates_stream_terminals_and_recovers() {
    for (mode, error) in [
        (Mode::Normal, None),
        (Mode::ProcessingEarly, None),
        (Mode::StatusBeforeHeader, Some(ErrorKind::Protocol)),
        (Mode::EarlySuccess, None),
        (Mode::AudioOnly, None),
        (Mode::CurvesOnly, None),
        (Mode::NoStatus, Some(ErrorKind::IncompleteResponse)),
        (Mode::NoHeader, Some(ErrorKind::Protocol)),
        (Mode::Trailers, Some(ErrorKind::Transport)),
        (Mode::MidInput, Some(ErrorKind::Transport)),
        (Mode::AfterInput, Some(ErrorKind::Transport)),
        (Mode::DuplicateHeader, Some(ErrorKind::Protocol)),
        (Mode::DuplicateEvent, Some(ErrorKind::Protocol)),
        (Mode::ErrorStatus, Some(ErrorKind::Inference)),
        (Mode::ReverseTime, Some(ErrorKind::Protocol)),
    ] {
        let server = Fixture::start(mode).await;
        let client = Client::server(ServerConfig::new(&server.url))
            .await
            .unwrap();
        let result = tokio::time::timeout(Duration::from_secs(5), collect_async(&client))
            .await
            .unwrap();
        if let Some(kind) = error {
            assert_eq!(result.unwrap_err().kind(), kind);
        } else {
            let events = result.unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|e| matches!(e, OutputEvent::Completed(_)))
                    .count(),
                1
            );
        }
        assert!(collect_async(&client).await.is_ok());
        client.shutdown().await.unwrap();
        server.close().await;
    }
}
#[test]
fn server_requires_runtime_and_runs_on_explicit_runtime_from_standard_executor() {
    assert_eq!(
        wait(Client::server(ServerConfig::new("http://127.0.0.1:1")))
            .err()
            .unwrap()
            .kind(),
        ErrorKind::RuntimeUnavailable
    );
    let rt = tokio::runtime::Runtime::new().unwrap();
    let server = rt.block_on(Fixture::start(Mode::Normal));
    let mut config = ServerConfig::new(&server.url);
    config.runtime = Some(rt.handle().clone());
    let client = wait(Client::server(config)).unwrap();
    let events = collect(&client, RequestOptions::default(), pcm(2, 128000)).unwrap();
    assert_eq!(returned_pcm(&events), pcm(2, 128000));
    wait(client.shutdown()).unwrap();
    rt.block_on(server.close());
}
#[test]
fn stopping_runtime_completes_pending_handles() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let server = rt.block_on(Fixture::start(Mode::Hold));
    let mut config = ServerConfig::new(&server.url);
    config.runtime = Some(rt.handle().clone());
    let client = wait(Client::server(config)).unwrap();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    wait(input.send(InputChunk::new(
        PcmBuffer::from_vec(vec![0, 0]).unwrap(),
        vec![],
    )))
    .unwrap();
    wait(input.finish()).unwrap();
    drop(rt);
    assert!(matches!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::RuntimeUnavailable | ErrorKind::Transport
    ));
    assert!(wait(control.closed()).is_err());
    wait(client.shutdown()).unwrap();
    drop(server);
}
#[tokio::test]
async fn deadline_closes_rpc_with_no_response() {
    let fixture = Fixture::start(Mode::Hold).await;
    let client = Client::server(ServerConfig::new(&fixture.url))
        .await
        .unwrap();
    let mut options = RequestOptions::default();
    options.timeout = Some(Duration::from_millis(50));
    let (input, mut output, control) = client.start(options).unwrap().split();
    input.finish().await.unwrap();
    assert_eq!(
        output.recv().await.unwrap_err().kind(),
        ErrorKind::DeadlineExceeded
    );
    assert!(control.closed().await.is_err());
    client.shutdown().await.unwrap();
    fixture.close().await;
}

#[test]
fn stopped_current_thread_runtime_wakes_a_backpressured_sender() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let fixture = rt.block_on(Fixture::start(Mode::Normal));
    let client = rt
        .block_on(Client::server(ServerConfig::new(&fixture.url)))
        .unwrap();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    for _ in 0..16 {
        wait(input.send(InputChunk::new(
            PcmBuffer::from_vec(vec![0, 0]).unwrap(),
            vec![],
        )))
        .unwrap();
    }
    let mut pending = input.send(InputChunk::new(
        PcmBuffer::from_vec(vec![0, 0]).unwrap(),
        vec![],
    ));
    assert!(
        Pin::new(&mut pending)
            .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()))
            .is_pending()
    );
    drop(rt);
    assert!(wait(pending).is_err());
    assert!(wait(output.recv()).is_err());
    assert!(wait(control.closed()).is_err());
    wait(client.shutdown()).unwrap();
    drop(fixture);
}
#[tokio::test]
async fn full_response_queue_cancels_without_consumer_polling() {
    let fixture = Fixture::start(Mode::Normal).await;
    let mut config = ServerConfig::new(&fixture.url);
    config.limits.output_queue_items = 1;
    let client = Client::server(config).await.unwrap();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    input
        .send(InputChunk::new(
            PcmBuffer::from_vec(vec![1, 2]).unwrap(),
            vec![],
        ))
        .await
        .unwrap();
    input.finish().await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while control.progress().received_audio_bytes == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    control.cancel();
    assert_eq!(
        output.recv().await.unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert!(control.closed().await.is_err());
    client.shutdown().await.unwrap();
    fixture.close().await;
}
#[test]
fn unavailable_or_disabled_runtime_fails_initialization() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let error = rt
        .block_on(Client::server(ServerConfig::new("http://127.0.0.1:1")))
        .err()
        .unwrap();
    assert_eq!(error.kind(), ErrorKind::RuntimeUnavailable);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut config = ServerConfig::new("http://127.0.0.1:1");
    config.runtime = Some(rt.handle().clone());
    config.connect_timeout = Duration::from_millis(100);
    assert_eq!(
        wait(Client::server(config)).err().unwrap().kind(),
        ErrorKind::Transport
    );
}

#[tokio::test]
async fn tcp_disconnect_during_response_is_not_success_and_next_request_recovers() {
    let fixture = Fixture::start(Mode::PartialHold).await;
    let upstream = fixture.url.trim_start_matches("http://").to_string();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (cut, cut_rx) = oneshot::channel();
    let proxy = tokio::spawn(async move {
        let (mut downstream, _) = listener.accept().await.unwrap();
        let mut upstream_first = tokio::net::TcpStream::connect(&upstream).await.unwrap();
        tokio::select! {_ = tokio::io::copy_bidirectional(&mut downstream,&mut upstream_first)=>{},_ = cut_rx=>{}}
        drop(downstream);
        drop(upstream_first);
        let (mut downstream, _) = listener.accept().await.unwrap();
        let mut upstream_next = tokio::net::TcpStream::connect(&upstream).await.unwrap();
        let _ = tokio::io::copy_bidirectional(&mut downstream, &mut upstream_next).await;
    });
    let client = Client::server(ServerConfig::new(url)).await.unwrap();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    input
        .send(InputChunk::new(
            PcmBuffer::from_vec(vec![1, 2]).unwrap(),
            vec![],
        ))
        .await
        .unwrap();
    input.finish().await.unwrap();
    assert!(matches!(
        output.recv().await.unwrap(),
        Some(OutputEvent::StreamInfo(_))
    ));
    assert!(matches!(
        output.recv().await.unwrap(),
        Some(OutputEvent::Audio(_))
    ));
    cut.send(()).unwrap();
    let failure = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match output.recv().await {
                Err(e) => break e,
                Ok(Some(OutputEvent::Completed(_))) | Ok(None) => {
                    panic!("truncated response succeeded")
                }
                _ => {}
            }
        }
    })
    .await
    .unwrap();
    assert_eq!(failure.kind(), ErrorKind::Transport);
    assert!(control.closed().await.is_err());
    assert!(
        tokio::time::timeout(Duration::from_secs(5), collect_async(&client))
            .await
            .unwrap()
            .is_ok()
    );
    client.shutdown().await.unwrap();
    proxy.abort();
    let _ = proxy.await;
    fixture.close().await;
}

#[tokio::test]
async fn tcp_disconnect_during_upload_wakes_input_and_output() {
    let fixture = Fixture::start(Mode::Hold).await;
    let upstream = fixture.url.trim_start_matches("http://").to_string();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (connected, ready) = oneshot::channel();
    let (cut, cut_rx) = oneshot::channel();
    let proxy = tokio::spawn(async move {
        let (mut a, _) = listener.accept().await.unwrap();
        let mut b = tokio::net::TcpStream::connect(upstream).await.unwrap();
        connected.send(()).unwrap();
        tokio::select! {_ = tokio::io::copy_bidirectional(&mut a,&mut b)=>{},_ = cut_rx=>{}}
    });
    let client = Client::server(ServerConfig::new(url)).await.unwrap();
    ready.await.unwrap();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    input
        .send(InputChunk::new(
            PcmBuffer::from_vec(vec![0; 32000]).unwrap(),
            vec![],
        ))
        .await
        .unwrap();
    // Input remains open: this utterance has not sent EndOfAudio.
    cut.send(()).unwrap();
    proxy.await.unwrap();
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), output.recv())
            .await
            .unwrap()
            .unwrap_err()
            .kind(),
        ErrorKind::Transport
    );
    assert!(
        input
            .send(InputChunk::new(
                PcmBuffer::from_vec(vec![0, 0]).unwrap(),
                vec![]
            ))
            .await
            .is_err()
    );
    assert!(control.closed().await.is_err());
    drop(input);
    client.shutdown().await.unwrap();
    fixture.close().await;
}

#[test]
fn transport_windows_reject_values_outside_http2_range() {
    for value in [0, 65534, 0x80000000, u32::MAX] {
        let mut config = ServerConfig::new("http://127.0.0.1:1");
        config.http2_window_bytes = value;
        assert_eq!(
            wait(Client::server(config)).err().unwrap().kind(),
            ErrorKind::InvalidInput
        );
    }
}
