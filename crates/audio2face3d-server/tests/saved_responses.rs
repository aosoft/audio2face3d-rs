//! Optional real-model compatibility replay of independently captured wire data.
#![cfg(feature = "runtime")]
use audio2face3d_server::{config::Config, proto, server};
use clap::Parser;
use prost::Message;
use proto::nvidia_ace::{
    emotion_aggregate::v1::EmotionAggregate, emotion_with_timecode::v1::EmotionWithTimeCode,
    services::a2f_controller::v1::a2f_controller_service_client::A2fControllerServiceClient,
};
use std::{fs, path::Path, time::Duration};
use tokio::{net::TcpListener, sync::oneshot};

fn packets<T: Message + Default>(path: &Path) -> Vec<T> {
    let data = fs::read(path).unwrap();
    let mut rest = data.as_slice();
    let mut values = vec![];
    while !rest.is_empty() {
        assert!(rest.len() >= 4);
        let len = u32::from_le_bytes(rest[..4].try_into().unwrap()) as usize;
        rest = &rest[4..];
        assert!(len <= rest.len());
        values.push(T::decode(&rest[..len]).unwrap());
        rest = &rest[len..];
    }
    values
}
fn near(a: f32, b: f32) {
    assert!(
        a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-5,
        "{a} != {b}"
    );
}
fn compare_emotions(a: &[EmotionWithTimeCode], b: &[EmotionWithTimeCode]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.time_code, b.time_code);
        assert_eq!(a.emotion.len(), b.emotion.len());
        for (name, value) in &a.emotion {
            near(*value, b.emotion[name]);
        }
    }
}
fn compare(
    mut a: proto::controller::AnimationDataStream,
    b: proto::controller::AnimationDataStream,
) {
    use proto::controller::animation_data_stream::StreamPart;
    match (&mut a.stream_part, &b.stream_part) {
        (
            Some(StreamPart::AnimationDataStreamHeader(a)),
            Some(StreamPart::AnimationDataStreamHeader(b)),
        ) => a.start_time_code_since_epoch = b.start_time_code_since_epoch,
        (Some(StreamPart::AnimationData(a)), Some(StreamPart::AnimationData(b))) => {
            let af = &mut a.skel_animation.as_mut().unwrap().blend_shape_weights;
            let bf = &b.skel_animation.as_ref().unwrap().blend_shape_weights;
            assert_eq!(af.len(), bf.len());
            for (a, b) in af.iter_mut().zip(bf) {
                assert_eq!(a.values.len(), b.values.len());
                for (a, b) in a.values.iter_mut().zip(&b.values) {
                    near(*a, *b);
                    *a = *b;
                }
            }
            assert_eq!(a.metadata.len(), b.metadata.len());
            for (key, a) in &mut a.metadata {
                let b = &b.metadata[key];
                assert_eq!(a.type_url, b.type_url);
                assert_eq!(key, "emotion_aggregate");
                let x = EmotionAggregate::decode(a.value.as_slice()).unwrap();
                let y = EmotionAggregate::decode(b.value.as_slice()).unwrap();
                compare_emotions(&x.input_emotions, &y.input_emotions);
                compare_emotions(&x.a2e_output, &y.a2e_output);
                compare_emotions(&x.a2f_smoothed_output, &y.a2f_smoothed_output);
                a.value = b.value.clone();
            }
        }
        _ => {}
    }
    assert_eq!(a, b);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires prepared models and A2F_BASELINE_DIR captured by temp/unified-client-work/capture-baseline.py"]
async fn saved_real_model_responses_match_after_protocol_extraction() {
    let base = std::path::PathBuf::from(
        std::env::var_os("A2F_BASELINE_DIR").expect("set A2F_BASELINE_DIR"),
    );
    assert_eq!(
        proto::DESCRIPTOR,
        fs::read(base.join("ace_descriptor.bin")).unwrap()
    );
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let mut total = 0;
    for group in ["regression", "regression-a2e"] {
        let dir = base.join(group);
        let raw = fs::read_to_string(dir.join("arguments.txt")).unwrap();
        let args = std::iter::once("replay".to_owned()).chain(raw.lines().map(|arg| {
            if arg.starts_with("models/") {
                repo.join(arg).display().to_string()
            } else {
                arg.to_owned()
            }
        }));
        let config = Config::parse_from(args);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel();
        let task = tokio::spawn(server::serve(config, listener, async {
            let _ = stopped.await;
        }));
        let mut client = A2fControllerServiceClient::connect(format!("http://{addr}"))
            .await
            .unwrap();
        let mut cases = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.is_dir())
            .collect::<Vec<_>>();
        cases.sort();
        for case in cases {
            let requests = packets::<proto::controller::AudioStream>(&case.join("request.bin"));
            let expected =
                packets::<proto::controller::AnimationDataStream>(&case.join("response.bin"));
            let mut request = tonic::Request::new(tokio_stream::iter(requests));
            request.set_timeout(Duration::from_secs(90));
            let mut output = client
                .process_audio_stream(request)
                .await
                .unwrap()
                .into_inner();
            for packet in expected {
                let actual = tokio::time::timeout(Duration::from_secs(90), output.message())
                    .await
                    .unwrap()
                    .unwrap()
                    .expect("truncated response");
                compare(actual, packet);
            }
            assert!(output.message().await.unwrap().is_none());
            eprintln!(
                "PASS {group}/{}",
                case.file_name().unwrap().to_string_lossy()
            );
            total += 1;
        }
        stop.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(20), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
    }
    assert_eq!(total, 12);
}
