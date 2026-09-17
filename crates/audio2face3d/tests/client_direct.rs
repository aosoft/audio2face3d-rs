#![cfg(any(feature = "mock", feature = "native"))]
mod support;
use audio2face3d::client::{types::*, *};
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Waker},
    time::Duration,
};
use support::*;
fn exercises(config: DirectConfig) {
    let slots = config.max_executions;
    let client = wait(Client::direct(config)).unwrap();
    let mut active = vec![];
    for _ in 0..slots {
        let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
        assert!(matches!(
            wait(output.recv()).unwrap(),
            Some(OutputEvent::StreamInfo(_))
        ));
        active.push((input, output, control));
    }
    let (waiting_input, mut waiting_output, waiting_control) = client
        .clone()
        .start(RequestOptions::default())
        .unwrap()
        .split();
    let mut receiving = waiting_output.recv();
    assert!(
        Pin::new(&mut receiving)
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    drop(receiving);
    waiting_control.cancel();
    assert_eq!(
        wait(waiting_output.recv()).unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert!(wait(waiting_control.closed()).is_err());
    drop(waiting_input);
    let (first_input, first_output, first_control) = active.remove(0);
    first_control.cancel();
    drop(first_input);
    drop(first_output);
    assert!(wait(first_control.closed()).is_err());
    // Other execution slots remain occupied while a freed slot runs new utterances.
    for seed in [20, 80] {
        let bytes = pcm(seed, 1067);
        let events = collect(&client, RequestOptions::default(), bytes.clone()).unwrap();
        assert_eq!(returned_pcm(&events), bytes);
    }
    for (input, output, control) in active {
        control.cancel();
        drop(input);
        drop(output);
        assert!(wait(control.closed()).is_err());
    }
    let mut options = RequestOptions::default();
    options.timeout = Some(Duration::from_millis(30));
    let (input, mut output, control) = client.start(options).unwrap().split();
    loop {
        match wait(output.recv()) {
            Ok(Some(_)) => {}
            Err(e) => {
                assert_eq!(e.kind(), ErrorKind::DeadlineExceeded);
                break;
            }
            _ => panic!(),
        }
    }
    assert!(wait(control.closed()).is_err());
    drop(input);
    let events = collect(&client, RequestOptions::default(), pcm(7, 533)).unwrap();
    assert_eq!(returned_pcm(&events), pcm(7, 533));
    wait(client.shutdown()).unwrap();
}
#[cfg(feature = "mock")]
#[test]
fn mock_queues_and_recovers_without_tokio() {
    for n in [1, 2, 4] {
        exercises(DirectConfig {
            engine: InferenceConfig {
                backend: BackendKind::Mock,
                ..Default::default()
            },
            max_executions: n,
            ..Default::default()
        });
    }
}
#[cfg(feature = "mock")]
#[test]
fn direct_backpressure_cancel_and_shutdown_without_tokio() {
    let config = DirectConfig {
        engine: InferenceConfig {
            backend: BackendKind::Mock,
            ..Default::default()
        },
        limits: Limits {
            input_queue_items: 1,
            output_queue_items: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let client = wait(Client::direct(config)).unwrap();
    let bytes = pcm(9, 16001);
    let events = collect(&client, RequestOptions::default(), bytes.clone()).unwrap();
    assert_eq!(returned_pcm(&events), bytes);
    let (mut input, output, control) = client.start(RequestOptions::default()).unwrap().split();
    wait(input.send(InputChunk::new(
        PcmBuffer::from_vec(pcm(4, 1600)).unwrap(),
        vec![],
    )))
    .unwrap();
    drop(output);
    assert_eq!(
        wait(control.closed()).unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    drop(input);
    wait(client.shutdown()).unwrap();
}
#[cfg(feature = "native")]
#[test]
#[ignore = "requires A2F_MODEL and CUDA/TensorRT"]
fn native_slots_one_two_cancel_recover_without_tokio() {
    let model = std::env::var_os("A2F_MODEL").expect("A2F_MODEL");
    for n in [1, 2] {
        exercises(DirectConfig {
            engine: InferenceConfig {
                backend: BackendKind::Regression,
                model: Some(model.clone().into()),
                ..Default::default()
            },
            max_executions: n,
            ..Default::default()
        });
    }
}
#[cfg(feature = "native")]
#[test]
#[ignore = "requires A2F_MODEL and A2E_MODEL"]
fn native_a2e_without_tokio() {
    let client = wait(Client::direct(DirectConfig {
        engine: InferenceConfig {
            backend: BackendKind::Regression,
            model: Some(std::env::var_os("A2F_MODEL").expect("A2F_MODEL").into()),
            emotion_model: Some(std::env::var_os("A2E_MODEL").expect("A2E_MODEL").into()),
            ..Default::default()
        },
        ..Default::default()
    }))
    .unwrap();
    let events = collect(&client, RequestOptions::default(), pcm(3, 1601)).unwrap();
    assert!(events.iter().any(|e| matches!(e, OutputEvent::Emotion(_))));
    wait(client.shutdown()).unwrap();
}

#[cfg(feature = "mock")]
#[test]
fn direct_admission_timeout_and_fifo_do_not_release_occupied_slots() {
    let client = wait(Client::direct(DirectConfig {
        engine: InferenceConfig {
            backend: BackendKind::Mock,
            ..Default::default()
        },
        queue_timeout: Duration::from_millis(40),
        max_queued: 2,
        ..Default::default()
    }))
    .unwrap();
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::StreamInfo(_))
    ));
    let (queued_input, mut queued_output, queued_control) =
        client.start(RequestOptions::default()).unwrap().split();
    assert_eq!(
        wait(queued_output.recv()).unwrap_err().kind(),
        ErrorKind::DeadlineExceeded
    );
    assert!(wait(queued_control.closed()).is_err());
    drop(queued_input);
    control.cancel();
    drop(input);
    drop(output);
    assert!(wait(control.closed()).is_err());
    wait(client.shutdown()).unwrap();
    let client = wait(Client::direct(DirectConfig {
        engine: InferenceConfig {
            backend: BackendKind::Mock,
            ..Default::default()
        },
        max_queued: 2,
        ..Default::default()
    }))
    .unwrap();
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    wait(output.recv()).unwrap();
    let (first_input, mut first_output, first_control) =
        client.start(RequestOptions::default()).unwrap().split();
    let (second_input, mut second_output, second_control) =
        client.start(RequestOptions::default()).unwrap().split();
    control.cancel();
    drop(input);
    drop(output);
    assert!(wait(control.closed()).is_err());
    assert!(matches!(
        wait(first_output.recv()).unwrap(),
        Some(OutputEvent::StreamInfo(_))
    ));
    let mut second = second_output.recv();
    assert!(
        Pin::new(&mut second)
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
    drop(second);
    first_control.cancel();
    drop(first_input);
    drop(first_output);
    assert!(wait(first_control.closed()).is_err());
    assert!(matches!(
        wait(second_output.recv()).unwrap(),
        Some(OutputEvent::StreamInfo(_))
    ));
    second_control.cancel();
    drop(second_input);
    drop(second_output);
    assert!(wait(second_control.closed()).is_err());
    wait(client.shutdown()).unwrap();
}
