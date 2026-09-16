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

#[tokio::test]
async fn all_52_diagnostic_curves_survive_rpc_without_cross_talk() {
    use audio2face3d_server::animation::CURVE_NAMES;
    use proto::controller::animation_data_stream::StreamPart as Output;
    for (selected, name) in CURVE_NAMES.iter().enumerate() {
        let mut running = Running::start(&["--mock-curve", name, "--mock-value", "0.5"]).await;
        let (tx, mut stream) = running.stream().await;
        let pcm = vec![37; 1600];
        tx.send(audio(pcm.clone())).await.unwrap();
        tx.send(end()).await.unwrap();
        let mut returned_pcm = Vec::new();
        let mut frames = 0;
        let mut header_seen = false;
        let mut success = false;
        let mut previous = None;
        while let Some(message) = stream.message().await.unwrap() {
            match message.stream_part.unwrap() {
                Output::AnimationDataStreamHeader(header) => {
                    assert_eq!(
                        header.skel_animation_header.unwrap().blend_shapes,
                        CURVE_NAMES
                    );
                    header_seen = true;
                }
                Output::AnimationData(frame) => {
                    assert!(header_seen);
                    let audio = frame.audio.unwrap();
                    let weights = frame.skel_animation.unwrap().blend_shape_weights;
                    assert_eq!(weights.len(), 1);
                    let sample = &weights[0];
                    assert_eq!(sample.time_code, audio.time_code);
                    if let Some(last) = previous {
                        assert!(sample.time_code > last);
                    }
                    previous = Some(sample.time_code);
                    assert_eq!(sample.values.len(), 52);
                    for (index, value) in sample.values.iter().enumerate() {
                        assert_eq!(
                            *value,
                            if index == selected { 0.5 } else { 0.0 },
                            "{name} index {index}"
                        );
                    }
                    returned_pcm.extend(audio.audio_buffer);
                    frames += 1;
                }
                Output::Status(status) => {
                    assert_eq!(status.code, 0);
                    success = true;
                }
                _ => {}
            }
        }
        assert!(frames > 0 && success, "{name}");
        assert_eq!(returned_pcm, pcm, "{name}");
        drop(tx);
        running.stop().await;
    }
}

#[test]
fn diagnostic_cli_rejects_invalid_inputs() {
    for args in [
        vec!["test", "--mock-curve", "NotACurve"],
        vec!["test", "--mock-value", "0.5"],
        vec!["test", "--mock-curve", "JawOpen", "--mock-value", "NaN"],
        vec!["test", "--mock-curve", "JawOpen", "--mock-value", "1.1"],
    ] {
        assert!(Config::try_parse_from(args).is_err());
    }
    let config = Config::parse_from(["test", "--backend", "regression", "--mock-curve", "JawOpen"]);
    assert_eq!(
        config.validate().unwrap_err(),
        "mock-curve/mock-value require the mock backend"
    );
}

#[test]
fn diagnostic_default_retains_pulse_and_pcm_timestamps() {
    use audio2face3d_server::{
        animation::{diagnostic_frame, mock_frame},
        config::MockPattern,
    };
    for start in [0, 4000, 8000, 12000, 16000] {
        let original = mock_frame(start, vec![1, 2], MockPattern::JawOpenPulse);
        assert_eq!(
            original,
            diagnostic_frame(
                start,
                vec![1, 2],
                MockPattern::JawOpenPulse,
                Some("JawOpen"),
                None
            )
        );
        let weights = &original.skel_animation.unwrap().blend_shape_weights[0];
        let expected = match start {
            0 | 16000 => 0.0,
            8000 => 1.0,
            _ => 0.5,
        };
        assert_eq!(weights.values[17], expected);
        assert_eq!(original.audio.unwrap().audio_buffer, vec![1, 2]);
        assert_eq!(weights.time_code, start as f64 / 16000.0);
    }
}

#[tokio::test]
async fn diagnostic_jaw_baseline_preserves_selected_curve_and_audio() {
    use proto::controller::animation_data_stream::StreamPart as Output;
    for (name, index) in [("MouthClose", 18), ("TongueOut", 51)] {
        for value in ["0", "0.5", "1"] {
            let mut running = Running::start(&[
                "--mock-curve",
                name,
                "--mock-value",
                value,
                "--mock-jaw-open",
                "0.5",
            ])
            .await;
            let (tx, mut stream) = running.stream().await;
            let pcm = vec![23; 1600];
            tx.send(audio(pcm.clone())).await.unwrap();
            tx.send(end()).await.unwrap();
            let mut returned = Vec::new();
            let mut success = false;
            while let Some(message) = stream.message().await.unwrap() {
                match message.stream_part.unwrap() {
                    Output::AnimationData(frame) => {
                        let audio = frame.audio.unwrap();
                        let weights = frame.skel_animation.unwrap().blend_shape_weights;
                        assert_eq!(weights[0].time_code, audio.time_code);
                        assert_eq!(weights[0].values.len(), 52);
                        for (i, weight) in weights[0].values.iter().enumerate() {
                            let expected = if i == 17 {
                                0.5
                            } else if i == index {
                                value.parse::<f32>().unwrap()
                            } else {
                                0.0
                            };
                            assert_eq!(*weight, expected, "{name}: {i}");
                        }
                        returned.extend(audio.audio_buffer);
                    }
                    Output::Status(status) => {
                        assert_eq!(status.code, 0);
                        success = true;
                    }
                    _ => {}
                }
            }
            assert!(success);
            assert_eq!(returned, pcm);
            drop(tx);
            running.stop().await;
        }
    }
}

#[test]
fn diagnostic_jaw_baseline_rejects_ambiguous_or_invalid_settings() {
    for args in [
        vec!["test", "--mock-jaw-open", "0.5"],
        vec![
            "test",
            "--mock-curve",
            "TongueOut",
            "--mock-jaw-open",
            "NaN",
        ],
        vec![
            "test",
            "--mock-curve",
            "TongueOut",
            "--mock-jaw-open",
            "1.1",
        ],
    ] {
        assert!(Config::try_parse_from(args).is_err());
    }
    assert!(
        Config::parse_from(["test", "--mock-curve", "JawOpen", "--mock-jaw-open", "0.5"])
            .validate()
            .is_err()
    );
    assert!(
        Config::parse_from([
            "test",
            "--backend",
            "regression",
            "--mock-curve",
            "TongueOut",
            "--mock-jaw-open",
            "0.5"
        ])
        .validate()
        .is_err()
    );
}
