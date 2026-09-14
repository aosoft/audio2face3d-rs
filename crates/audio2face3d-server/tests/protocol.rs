use audio2face3d_server::{
    config::Config,
    proto::{
        self,
        controller::{
            AudioStream, AudioStreamHeader,
            audio_stream::{EndOfAudio, StreamPart},
        },
        nvidia_ace::services::a2f_controller::v1::a2f_controller_service_client::A2fControllerServiceClient,
    },
    server,
};
use clap::Parser;
use std::time::Duration;
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot},
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Code, transport::Channel};

fn header() -> AudioStream {
    AudioStream {
        stream_part: Some(StreamPart::AudioStreamHeader(AudioStreamHeader {
            audio_header: Some(proto::audio::AudioHeader {
                audio_format: 0,
                channel_count: 1,
                samples_per_second: 16_000,
                bits_per_sample: 16,
            }),
            ..Default::default()
        })),
    }
}
fn audio(pcm: Vec<u8>) -> AudioStream {
    AudioStream {
        stream_part: Some(StreamPart::AudioWithEmotion(proto::a2f::AudioWithEmotion {
            audio_buffer: pcm,
            emotions: vec![],
        })),
    }
}
fn end() -> AudioStream {
    AudioStream {
        stream_part: Some(StreamPart::EndOfAudio(EndOfAudio {})),
    }
}

struct Running {
    client: A2fControllerServiceClient<Channel>,
    stop: oneshot::Sender<()>,
    task: tokio::task::JoinHandle<Result<(), Box<dyn std::error::Error + Send + Sync>>>,
    addr: std::net::SocketAddr,
}
impl Running {
    async fn start(extra: &[&str]) -> Self {
        let mut args = vec![
            "test",
            "--input-idle-timeout-ms",
            "200",
            "--output-timeout-ms",
            "200",
            "--max-audio-seconds",
            "1",
        ];
        args.extend_from_slice(extra);
        let config = Config::parse_from(args);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(server::serve(config, listener, async {
            let _ = stopped.await;
        }));
        let client = A2fControllerServiceClient::connect(format!("http://{addr}"))
            .await
            .unwrap();
        Self {
            client,
            stop,
            task,
            addr,
        }
    }
    async fn stop(self) {
        self.stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), self.task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let rebound = TcpListener::bind(self.addr).await.unwrap();
        drop(rebound);
    }
    async fn stream(
        &mut self,
    ) -> (
        mpsc::Sender<AudioStream>,
        tonic::Streaming<proto::controller::AnimationDataStream>,
    ) {
        let (tx, rx) = mpsc::channel(8);
        tx.send(header()).await.unwrap();
        let stream = self
            .client
            .process_audio_stream(ReceiverStream::new(rx))
            .await
            .unwrap()
            .into_inner();
        (tx, stream)
    }
}

#[tokio::test]
async fn streams_before_end_and_finishes_without_half_close() {
    let mut running = Running::start(&[]).await;
    let (tx, mut stream) = running.stream().await;
    tx.send(audio(vec![7; 1066])).await.unwrap();
    assert!(matches!(
        stream.message().await.unwrap().unwrap().stream_part,
        Some(proto::controller::animation_data_stream::StreamPart::AnimationDataStreamHeader(_))
    ));
    let frame = tokio::time::timeout(Duration::from_secs(1), stream.message())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(
        frame.stream_part,
        Some(proto::controller::animation_data_stream::StreamPart::AnimationData(_))
    ));
    tx.send(end()).await.unwrap();
    let mut parts = Vec::new();
    while let Some(message) = stream.message().await.unwrap() {
        parts.push(message.stream_part.unwrap());
    }
    assert!(
        matches!(parts.as_slice(), [proto::controller::animation_data_stream::StreamPart::Event(_), proto::controller::animation_data_stream::StreamPart::Status(s)] if s.code == 0)
    );
    drop(tx);
    running.stop().await;
}

#[tokio::test]
async fn protocol_errors_are_finite_and_do_not_succeed() {
    let mut running = Running::start(&[]).await;
    for (messages, code) in [
        (vec![header()], Code::InvalidArgument),
        (
            vec![AudioStream { stream_part: None }],
            Code::InvalidArgument,
        ),
        (vec![audio(vec![1])], Code::InvalidArgument),
        (vec![audio(vec![]), end()], Code::InvalidArgument),
        (vec![audio(vec![0; 32_002])], Code::ResourceExhausted),
        (vec![], Code::InvalidArgument),
    ] {
        let (tx, mut stream) = running.stream().await;
        for message in messages {
            tx.send(message).await.unwrap();
        }
        drop(tx);
        let error = tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match stream.message().await {
                    Err(error) => break error,
                    Ok(Some(_)) => {}
                    Ok(None) => panic!("unexpected successful EOF"),
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(error.code(), code);
    }
    running.stop().await;
}

#[tokio::test]
async fn idle_timeout_concurrency_limit_and_shutdown_release_sessions() {
    let mut running = Running::start(&[]).await;
    let (_tx, mut stream) = running.stream().await;
    let error = running
        .client
        .process_audio_stream(tokio_stream::iter([header()]))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::ResourceExhausted);
    let error = loop {
        if let Err(error) = stream.message().await {
            break error;
        }
    };
    assert_eq!(error.code(), Code::DeadlineExceeded);
    let (_tx2, mut stream2) = running.stream().await;
    running.stop.send(()).unwrap();
    let error = loop {
        match stream2.message().await {
            Err(error) => break error,
            Ok(None) => panic!("shutdown returned OK"),
            _ => {}
        }
    };
    assert_eq!(error.code(), Code::Unavailable);
    tokio::time::timeout(Duration::from_secs(3), running.task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
}

#[tokio::test]
async fn rejects_bad_first_header_and_unsupported_backend() {
    let mut running = Running::start(&[]).await;
    for message in [
        end(),
        AudioStream {
            stream_part: Some(StreamPart::AudioStreamHeader(AudioStreamHeader::default())),
        },
    ] {
        let error = running
            .client
            .process_audio_stream(tokio_stream::iter([message]))
            .await
            .unwrap_err();
        assert_eq!(error.code(), Code::InvalidArgument);
    }
    let config = Config::parse_from(["test", "--backend", "regression"]);
    assert!(config.validate().unwrap_err().contains("regression"));
    running.stop().await;
}

#[tokio::test]
async fn dropping_response_cancels_idle_worker_and_allows_next_rpc() {
    let mut running = Running::start(&[]).await;
    let (tx, stream) = running.stream().await;
    drop(stream);
    // Keep the input sender open: response cancellation alone must release the worker.
    tokio::time::timeout(Duration::from_secs(2), tx.closed())
        .await
        .unwrap();
    let (next_tx, mut next_stream) = running.stream().await;
    next_tx.send(audio(vec![0; 2])).await.unwrap();
    next_tx.send(end()).await.unwrap();
    while next_stream.message().await.unwrap().is_some() {}
    running.stop().await;
}
