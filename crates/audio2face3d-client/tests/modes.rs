#![cfg(all(feature = "direct", feature = "server"))]
mod support;
use audio2face3d_client::{types::*, *};
use audio2face3d_server::{config::Config, server};
use clap::Parser;
use std::time::Duration;
use support::*;
fn compare(direct_config: DirectConfig, args: Vec<String>) {
    let direct = wait(Client::direct(direct_config)).unwrap();
    let mut expected = vec![];
    for rate in [16000, 44100, 48000] {
        let mut options = RequestOptions::new(AudioFormat::pcm16(rate, 1).unwrap());
        options.timeout = Some(Duration::from_secs(120));
        expected.push(collect(&direct, options, pcm(31, (rate / 10 + 7) as usize)).unwrap());
    }
    wait(direct.shutdown()).unwrap();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let config = Config::parse_from(args);
    let listener = rt
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let (tx, rx) = tokio::sync::oneshot::channel();
    let serving = rt.spawn(server::serve(config, listener, async {
        let _ = rx.await;
    }));
    // The native server prepares its engine before accepting requests; connect is bounded.
    let mut remote_config = ServerConfig::new(url);
    remote_config.runtime = Some(rt.handle().clone());
    remote_config.connect_timeout = Duration::from_secs(120);
    let remote = wait(Client::server(remote_config)).unwrap();
    for (rate, a) in [16000, 44100, 48000].into_iter().zip(expected) {
        let mut options = RequestOptions::new(AudioFormat::pcm16(rate, 1).unwrap());
        options.timeout = Some(Duration::from_secs(120));
        let bytes = pcm(31, (rate / 10 + 7) as usize);
        let b = collect(&remote, options, bytes).unwrap();
        assert_eq!(returned_pcm(&a), returned_pcm(&b));
        let ac: Vec<_> = a
            .iter()
            .filter_map(|e| {
                if let OutputEvent::Curves(c) = e {
                    Some(c)
                } else {
                    None
                }
            })
            .collect();
        let bc: Vec<_> = b
            .iter()
            .filter_map(|e| {
                if let OutputEvent::Curves(c) = e {
                    Some(c)
                } else {
                    None
                }
            })
            .collect();
        assert!(!ac.is_empty());
        assert_eq!(ac.len(), bc.len());
        for (a, b) in ac.iter().zip(bc) {
            assert_eq!(a.layout(), b.layout());
            assert!(a.time().as_nanos().abs_diff(b.time().as_nanos()) <= 1);
            for (a, b) in a.values().iter().zip(b.values()) {
                assert!((a - b).abs() <= 2e-5, "{a} vs {b}");
            }
        }
        let ae: Vec<_> = a
            .iter()
            .filter_map(|e| {
                if let OutputEvent::Emotion(c) = e {
                    Some(c)
                } else {
                    None
                }
            })
            .collect();
        let be: Vec<_> = b
            .iter()
            .filter_map(|e| {
                if let OutputEvent::Emotion(c) = e {
                    Some(c)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(ae.len(), be.len());
        for (a, b) in ae.into_iter().zip(be) {
            for (a, b) in
                [&a.input, &a.mixed, &a.smoothed]
                    .into_iter()
                    .zip([&b.input, &b.mixed, &b.smoothed])
            {
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert!(a.time().as_nanos().abs_diff(b.time().as_nanos()) <= 1);
                    assert_eq!(a.values().len(), b.values().len());
                    for (k, v) in a.values() {
                        assert!((v - b.values()[k]).abs() <= 2e-5);
                    }
                }
            }
        }
    }
    wait(remote.shutdown()).unwrap();
    tx.send(()).unwrap();
    rt.block_on(serving).unwrap().unwrap();
}
#[test]
fn common_application_matches_real_mock_server() {
    compare(DirectConfig::default(), vec!["test".into()]);
}
#[cfg(feature = "runtime")]
#[test]
#[ignore = "requires A2F_MODEL and server runtime feature"]
fn native_direct_matches_server() {
    let model = std::env::var("A2F_MODEL").expect("A2F_MODEL");
    compare(
        DirectConfig {
            engine: InferenceConfig {
                backend: BackendKind::Regression,
                model: Some(model.clone().into()),
                ..Default::default()
            },
            ..Default::default()
        },
        vec![
            "test".into(),
            "--backend".into(),
            "regression".into(),
            "--model".into(),
            model,
        ],
    );
}
#[cfg(feature = "runtime")]
#[test]
#[ignore = "requires A2F_MODEL/A2E_MODEL and server runtime feature"]
fn native_emotion_matches_server() {
    let model = std::env::var("A2F_MODEL").expect("A2F_MODEL");
    let emotion = std::env::var("A2E_MODEL").expect("A2E_MODEL");
    compare(
        DirectConfig {
            engine: InferenceConfig {
                backend: BackendKind::Regression,
                model: Some(model.clone().into()),
                emotion_model: Some(emotion.clone().into()),
                ..Default::default()
            },
            ..Default::default()
        },
        vec![
            "test".into(),
            "--backend".into(),
            "regression".into(),
            "--model".into(),
            model,
            "--emotion-model".into(),
            emotion,
        ],
    );
}
