use crate::inference::{
    Backend, Cancellation, Config, Factory, admission::Admission, worker::wait,
};
use crate::types::{AudioFormat, ErrorKind, InputChunk, PcmBuffer, RequestOptions};
use std::{sync::Arc, time::Duration};

fn pcm(seed: u8, count: usize) -> Vec<u8> {
    (0..count * 2)
        .map(|i| seed.wrapping_add((i % 71) as u8))
        .collect()
}
fn push(engine: &mut dyn Backend, bytes: Vec<u8>) {
    wait(engine.push(InputChunk::new(PcmBuffer::from_vec(bytes).unwrap(), vec![]))).unwrap();
}
fn drain(engine: &mut dyn Backend, output: &mut Vec<u8>, cancel: &Cancellation) {
    while let Some(batch) = wait(engine.next_frame(cancel)).unwrap() {
        let audio = batch.audio.unwrap();
        assert_eq!(audio.position().0 * 2, output.len() as u64);
        assert_eq!(batch.curves.len(), 1);
        let frame = &batch.curves[0];
        assert_eq!(frame.layout().names().len(), 52);
        assert_eq!(frame.values().len(), 52);
        assert!(frame.values().iter().all(|v| v.is_finite()));
        assert_eq!(
            frame.time().nearest_sample(16000).unwrap(),
            audio.position()
        );
        output.extend_from_slice(audio.pcm().as_bytes());
    }
}
#[test]
fn mock_streams_and_flushes_with_only_standard_future_waiting() {
    let factory = wait(Factory::prepare(Config::default())).unwrap();
    let cancel = Cancellation::new();
    for rate in [16000, 44100, 48000] {
        let mut engine =
            wait(factory.start(RequestOptions::new(AudioFormat::pcm16(rate, 1).unwrap()))).unwrap();
        let bytes = pcm(12, rate as usize + 1);
        let mut output = vec![];
        for chunk in bytes.chunks(1066) {
            push(engine.as_mut(), chunk.to_vec());
            drain(engine.as_mut(), &mut output, &cancel);
        }
        assert!(!output.is_empty());
        wait(engine.finish()).unwrap();
        drain(engine.as_mut(), &mut output, &cancel);
        assert_eq!(
            output.len() / 2,
            ((rate as usize + 1) * 16000).div_ceil(rate as usize)
        );
        if rate == 16000 {
            assert_eq!(output, bytes);
        }
        wait(engine.close()).unwrap();
    }
}
#[test]
fn shared_input_validation_preserves_empty_finish_and_duration_errors() {
    let factory = wait(Factory::prepare(Config {
        max_audio_seconds: 1,
        ..Config::default()
    }))
    .unwrap();
    let mut engine = wait(factory.start(RequestOptions::default())).unwrap();
    assert_eq!(
        wait(engine.finish()).unwrap_err().kind(),
        ErrorKind::InvalidInput
    );
    wait(engine.close()).unwrap();
    let mut engine = wait(factory.start(RequestOptions::default())).unwrap();
    assert_eq!(
        wait(engine.push(InputChunk::new(
            PcmBuffer::from_vec(vec![0; 32002]).unwrap(),
            vec![]
        )))
        .unwrap_err()
        .kind(),
        ErrorKind::LimitExceeded
    );
    wait(engine.close()).unwrap();
    assert!(
        wait(factory.start(RequestOptions::new(AudioFormat::pcm16(22050, 1).unwrap()))).is_err()
    );
    assert!(
        wait(factory.start(RequestOptions::new(AudioFormat::pcm16(16000, 2).unwrap()))).is_err()
    );
}
#[test]
fn mock_admissions_and_sessions_share_any_available_slot() {
    for n in [1, 2, 4] {
        queue_sessions(Config::default(), n);
    }
}
fn queue_sessions(config: Config, n: usize) {
    let factory = wait(Factory::prepare(config)).unwrap();
    let queue = Admission::new(n, 3, Duration::ZERO).unwrap();
    let cancel = Cancellation::new();
    let mut active = vec![];
    for i in 0..n {
        let permit = Arc::new(wait(queue.acquire(&cancel)).unwrap());
        let mut engine = wait(factory.start(RequestOptions::default())).unwrap();
        let bytes = pcm(i as u8 + 10, 1601);
        push(engine.as_mut(), bytes.clone());
        active.push((engine, permit, bytes));
    }
    let first = queue.acquire(&cancel);
    let second = queue.acquire(&cancel);
    assert_eq!(queue.waiting_count(), 2);
    assert_eq!(queue.available_permits(), 0);
    let (mut engine, worker_permit, expected) = active.pop().unwrap();
    let response_permit = worker_permit.clone();
    let mut output = vec![];
    wait(engine.finish()).unwrap();
    drain(engine.as_mut(), &mut output, &cancel);
    wait(engine.close()).unwrap();
    assert_eq!(output, expected);
    drop(engine);
    drop(worker_permit);
    assert_eq!(queue.waiting_count(), 2); // Output owner still holds execution admission.
    drop(response_permit);
    assert_eq!(queue.waiting_count(), 1);
    for (request, seed) in [(first, 90), (second, 120)] {
        let permit = wait(request).unwrap();
        let mut engine = wait(factory.start(RequestOptions::default())).unwrap();
        let expected = pcm(seed, 1067);
        push(engine.as_mut(), expected.clone());
        let mut output = vec![];
        wait(engine.finish()).unwrap();
        drain(engine.as_mut(), &mut output, &cancel);
        wait(engine.close()).unwrap();
        assert_eq!(output, expected);
        drop(engine);
        drop(permit);
    }
    // These engines have remained occupied while the freed slot ran both queued requests.
    for (mut engine, permit, expected) in active {
        let mut output = vec![];
        wait(engine.finish()).unwrap();
        drain(engine.as_mut(), &mut output, &cancel);
        wait(engine.close()).unwrap();
        assert_eq!(output, expected);
        drop(engine);
        drop(permit);
    }
    wait(factory.release_prepared()).unwrap();
    assert_eq!(queue.available_permits(), n);
    println!("PASS standard-Future sessions, concurrency={n}");
}
#[cfg(feature = "native")]
#[test]
#[ignore = "requires A2F_MODEL, CUDA and TensorRT; uses no async runtime"]
fn regression_load_infer_queue_cancel_and_destroy_without_async_runtime() {
    use crate::inference::BackendKind;
    use std::task::{Context, Waker};
    let model = std::env::var_os("A2F_MODEL").expect("set A2F_MODEL");
    let config = Config {
        backend: BackendKind::Regression,
        model: Some(model.into()),
        ..Config::default()
    };
    for n in [1, 2] {
        queue_sessions(config.clone(), n);
    }
    let factory = wait(Factory::prepare(config.clone())).unwrap();
    let queue = Admission::new(1, 1, Duration::ZERO).unwrap();
    let permit = wait(queue.acquire(&Cancellation::new())).unwrap();
    let mut engine = wait(factory.start(RequestOptions::default())).unwrap();
    push(engine.as_mut(), pcm(5, 16000));
    let cancel = Cancellation::new();
    let mut pending = engine.next_frame(&cancel);
    let immediate = pending
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()));
    cancel.cancel();
    if immediate.is_pending() {
        let _ = wait(pending);
    } else {
        drop(pending);
    }
    assert_eq!(
        wait(engine.next_frame(&cancel)).unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert_eq!(queue.available_permits(), 0);
    wait(engine.close()).unwrap();
    drop(engine);
    drop(permit);
    assert_eq!(queue.available_permits(), 1);
    let permit = wait(queue.acquire(&Cancellation::new())).unwrap();
    let mut engine = wait(factory.start(RequestOptions::default())).unwrap();
    let expected = pcm(4, 533);
    push(engine.as_mut(), expected.clone());
    wait(engine.finish()).unwrap();
    let mut output = vec![];
    drain(engine.as_mut(), &mut output, &Cancellation::new());
    wait(engine.close()).unwrap();
    assert_eq!(output, expected);
    drop(engine);
    drop(permit);
    wait(factory.release_prepared()).unwrap();
    // Explicitly release a warm model that was never assigned to an utterance.
    let warm = wait(Factory::prepare(config)).unwrap();
    wait(warm.release_prepared()).unwrap();
    println!(
        "PASS standard-Future native cancellation, recovery and unused warm-state destruction"
    );
}

#[cfg(feature = "native")]
#[test]
#[ignore = "requires A2F_MODEL and A2E_MODEL; uses no async runtime"]
fn a2e_metadata_without_async_runtime() {
    let config = Config {
        backend: crate::inference::BackendKind::Regression,
        model: Some(std::env::var_os("A2F_MODEL").expect("set A2F_MODEL").into()),
        emotion_model: Some(std::env::var_os("A2E_MODEL").expect("set A2E_MODEL").into()),
        ..Config::default()
    };
    let factory = wait(Factory::prepare(config)).unwrap();
    let mut engine = wait(factory.start(RequestOptions::default())).unwrap();
    let expected = pcm(42, 1601);
    push(engine.as_mut(), expected.clone());
    wait(engine.finish()).unwrap();
    let mut output = vec![];
    let mut count = 0;
    while let Some(batch) = wait(engine.next_frame(&Cancellation::new())).unwrap() {
        let audio = batch.audio.unwrap();
        output.extend_from_slice(audio.pcm().as_bytes());
        let trace = batch.emotion.unwrap();
        for entries in [trace.input, trace.mixed, trace.smoothed] {
            assert_eq!(entries.len(), 1);
            let entry = &entries[0];
            assert_eq!(
                entry.time().nearest_sample(16000).unwrap(),
                audio.position()
            );
            assert_eq!(entry.values().len(), 10);
            assert!(entry.values().values().all(|v| v.is_finite()));
        }
        count += 1;
    }
    wait(engine.close()).unwrap();
    wait(factory.release_prepared()).unwrap();
    assert_eq!(output, expected);
    assert_eq!(count, 4);
}
