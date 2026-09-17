use crate::client::{driver::*, types::*, *};
use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};
#[derive(Default)]
struct Fake {
    jobs: Mutex<VecDeque<(RequestOptions, WorkerSession)>>,
    shutdown: Mutex<Option<DriverShutdown>>,
    hold_shutdown: bool,
}
impl Driver for Fake {
    fn launch(&self, options: RequestOptions, session: WorkerSession) -> Result<()> {
        self.jobs.lock().unwrap().push_back((options, session));
        Ok(())
    }
    fn shutdown(&self, completion: DriverShutdown) {
        if self.hold_shutdown {
            *self.shutdown.lock().unwrap() = Some(completion);
        } else {
            completion.finish(Ok(()));
        }
    }
}
impl Fake {
    fn take(&self) -> (Reader, Writer, WorkerGuard) {
        self.jobs.lock().unwrap().pop_front().unwrap().1.split()
    }
}
fn limits() -> Limits {
    Limits {
        max_requests: 2,
        max_request_bytes: 4096,
        max_buffered_bytes: 65536,
        input_queue_items: 1,
        input_queue_bytes: 4096,
        max_input_chunk_bytes: 2048,
        output_queue_items: 1,
        output_queue_bytes: 4096,
        max_output_event_bytes: 2048,
    }
}
fn setup() -> (Client, Arc<Fake>) {
    let fake = Arc::new(Fake::default());
    (Client::with_driver(limits(), fake.clone()).unwrap(), fake)
}
struct ThreadWake(std::sync::mpsc::Sender<()>);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        let _ = self.0.send(());
    }
}
fn wait<F: Future>(future: F) -> F::Output {
    let (tx, rx) = std::sync::mpsc::channel();
    let waker = Waker::from(Arc::new(ThreadWake(tx)));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut cx) {
            return result;
        }
        assert!(Instant::now() < until, "Future was not completed/woken");
        rx.recv_timeout(until.saturating_duration_since(Instant::now()))
            .expect("Future failed to wake");
    }
}
fn pending<F: Future + Unpin>(future: &mut F) {
    assert!(
        Pin::new(future)
            .poll(&mut Context::from_waker(Waker::noop()))
            .is_pending()
    );
}
fn chunk(bytes: Vec<u8>) -> InputChunk {
    InputChunk::new(PcmBuffer::from_vec(bytes).unwrap(), vec![])
}
fn audio(bytes: Vec<u8>) -> OutputEvent {
    OutputEvent::Audio(
        AudioBlock::new(
            AudioFormat::MONO_16KHZ,
            SamplePosition(0),
            PcmBuffer::from_vec(bytes).unwrap(),
        )
        .unwrap(),
    )
}
fn eof(input: Input, reader: &mut Reader) {
    wait(input.finish()).unwrap();
    assert!(wait(reader.recv()).unwrap().is_none());
}
fn until(mut condition: impl FnMut() -> bool) {
    let end = Instant::now() + Duration::from_secs(5);
    while !condition() {
        assert!(Instant::now() < end);
        thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn local_start_does_not_wait_for_worker_or_response() {
    let (client, fake) = setup();
    let session = client.start(RequestOptions::default()).unwrap();
    let (mut input, output, control) = session.split();
    wait(input.send(chunk(vec![1, 2]))).unwrap();
    assert_eq!(fake.jobs.lock().unwrap().len(), 1);
    let (mut reader, _, guard) = fake.take();
    assert_eq!(
        wait(reader.recv())
            .unwrap()
            .unwrap()
            .chunk()
            .pcm()
            .as_bytes(),
        &[1, 2]
    );
    eof(input, &mut reader);
    guard.finish(Ok(()));
    wait(control.closed()).unwrap();
    drop(output);
    wait(client.shutdown()).unwrap();
}
#[test]
fn successful_output_drains_before_single_completed_and_releases_request_limit() {
    let fake = Arc::new(Fake::default());
    let client = Client::with_driver(
        Limits {
            max_requests: 1,
            ..limits()
        },
        fake.clone(),
    )
    .unwrap();
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, mut writer, guard) = fake.take();
    eof(input, &mut reader);
    wait(writer.emit(audio(vec![1, 2, 3, 4]))).unwrap();
    guard.finish(Ok(()));
    wait(control.closed()).unwrap();
    assert!(client.clone().start(RequestOptions::default()).is_err());
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::Audio(_))
    ));
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::Completed(Summary {
            progress: Progress {
                received_audio_bytes: 4,
                delivered_audio_bytes: 4,
                ..
            }
        }))
    ));
    assert!(wait(output.recv()).unwrap().is_none());
    control.cancel();
    wait(control.closed()).unwrap();
    let session = client.start(RequestOptions::default()).unwrap();
    drop(session);
    drop(fake.take());
    wait(client.shutdown()).unwrap();
}
#[test]
fn full_input_queue_waits_without_copy_and_abandoned_send_is_never_enqueued() {
    let (client, fake) = setup();
    let (mut input, output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, _, guard) = fake.take();
    wait(input.send(chunk(vec![1, 2]))).unwrap();
    let bytes = vec![3, 4];
    let pointer = bytes.as_ptr();
    let mut second = input.send(chunk(bytes));
    pending(&mut second);
    drop(wait(reader.recv()).unwrap());
    wait(second).unwrap();
    let packet = wait(reader.recv()).unwrap().unwrap();
    assert_eq!(packet.chunk().pcm().as_bytes().as_ptr(), pointer);
    drop(packet);
    wait(input.send(chunk(vec![5, 6]))).unwrap();
    let mut abandoned = input.send(chunk(vec![7, 8]));
    pending(&mut abandoned);
    drop(abandoned);
    assert_eq!(
        wait(reader.recv())
            .unwrap()
            .unwrap()
            .chunk()
            .pcm()
            .as_bytes(),
        &[5, 6]
    );
    eof(input, &mut reader);
    guard.finish(Ok(()));
    drop(output);
    wait(control.closed()).unwrap();
    wait(client.shutdown()).unwrap();
}
#[test]
fn full_output_queue_waits_and_preserves_pcm_ownership() {
    let (client, fake) = setup();
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, mut writer, guard) = fake.take();
    eof(input, &mut reader);
    wait(writer.emit(OutputEvent::ProcessingFinished)).unwrap();
    let bytes = vec![9, 10];
    let pointer = bytes.as_ptr();
    let mut next = writer.emit(audio(bytes));
    pending(&mut next);
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::ProcessingFinished)
    ));
    wait(next).unwrap();
    guard.finish(Ok(()));
    let Some(OutputEvent::Audio(block)) = wait(output.recv()).unwrap() else {
        panic!()
    };
    assert_eq!(block.pcm().as_bytes().as_ptr(), pointer);
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::Completed(_))
    ));
    wait(control.closed()).unwrap();
    wait(client.shutdown()).unwrap();
}
#[test]
fn error_bypasses_full_queues_but_closed_waits_for_cleanup() {
    let (client, fake) = setup();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, mut writer, guard) = fake.take();
    wait(input.send(chunk(vec![1, 2]))).unwrap();
    let mut sending = input.send(chunk(vec![3, 4]));
    pending(&mut sending);
    wait(writer.emit(audio(vec![5, 6]))).unwrap();
    let mut emitting = writer.emit(audio(vec![7, 8]));
    pending(&mut emitting);
    guard.fail(Error::new(ErrorKind::Transport, "injected reset"));
    let error = wait(output.recv()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Transport);
    assert_eq!(error.request_id(), Some(control.id()));
    assert_eq!(error.progress().received_audio_bytes, 4);
    assert_eq!(error.progress().delivered_audio_bytes, 0);
    assert!(wait(output.recv()).unwrap().is_none());
    assert_eq!(wait(sending).unwrap_err().kind(), ErrorKind::Transport);
    assert_eq!(wait(emitting).unwrap_err().kind(), ErrorKind::Transport);
    let mut closed = control.closed();
    pending(&mut closed);
    guard.finish(Err(Error::new(ErrorKind::Inference, "cleanup after reset")));
    assert_eq!(wait(closed).unwrap_err().kind(), ErrorKind::Transport);
    assert_eq!(control.request.core.state.lock().unwrap().bytes, 0);
    drop(input);
    wait(client.shutdown()).unwrap();
}
#[test]
fn cancellation_without_poll_reclaims_pending_owned_buffers() {
    let (client, fake) = setup();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    let baseline = control.request.core.state.lock().unwrap().bytes;
    let sending = input.send(chunk(vec![0; 1000]));
    assert!(control.request.core.state.lock().unwrap().bytes > baseline);
    control.cancel();
    control.cancel();
    assert_eq!(control.request.core.state.lock().unwrap().bytes, baseline);
    assert_eq!(wait(sending).unwrap_err().kind(), ErrorKind::Cancelled);
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    let mut closed = control.closed();
    pending(&mut closed);
    guard.finish(Ok(()));
    assert_eq!(wait(closed).unwrap_err().kind(), ErrorKind::Cancelled);
    drop(input);
    wait(client.shutdown()).unwrap();
}
#[test]
fn deadline_runs_without_polling_and_includes_waiting_for_output_consumption() {
    let (client, fake) = setup();
    let options = RequestOptions {
        timeout: Some(Duration::from_millis(20)),
        ..Default::default()
    };
    let (mut input, mut output, control) = client.start(options.clone()).unwrap().split();
    let (_, _, guard) = fake.take();
    let sending = input.send(chunk(vec![0; 1000]));
    until(|| control.request.has_failed());
    assert_eq!(
        wait(sending).unwrap_err().kind(),
        ErrorKind::DeadlineExceeded
    );
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::DeadlineExceeded
    );
    guard.finish(Ok(()));
    drop(input);
    let (input, mut output, control) = client.start(options).unwrap().split();
    let (mut reader, mut writer, guard) = fake.take();
    eof(input, &mut reader);
    wait(writer.emit(audio(vec![1, 2]))).unwrap();
    guard.finish(Ok(()));
    wait(control.closed()).unwrap();
    until(|| control.request.has_failed());
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::DeadlineExceeded
    );
    // closed describes cleanup; cancellation after cleanup does not rewrite that result.
    wait(control.closed()).unwrap();
    wait(client.shutdown()).unwrap();
}
#[test]
fn drops_have_distinct_semantics_and_finish_is_an_ordered_barrier() {
    let (client, fake) = setup();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, _, guard) = fake.take();
    drop(control.clone());
    wait(input.send(chunk(vec![1, 2]))).unwrap();
    wait(input.finish()).unwrap();
    assert!(wait(reader.recv()).unwrap().is_some());
    assert!(wait(reader.recv()).unwrap().is_none());
    guard.finish(Ok(()));
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::Completed(_))
    ));
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    drop(input.finish());
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    drop(guard);
    assert_eq!(
        wait(control.closed()).unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    let session = client.start(RequestOptions::default()).unwrap();
    let (_, _, guard) = fake.take();
    let mut cancelled = guard.cancelled();
    pending(&mut cancelled);
    drop(session);
    assert_eq!(wait(cancelled).kind(), ErrorKind::Cancelled);
    drop(guard);
    wait(client.shutdown()).unwrap();
}
#[test]
fn premature_worker_stop_and_duplicate_terminal_do_not_succeed() {
    let (client, fake) = setup();
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    drop(guard);
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::Inference
    );
    assert!(wait(control.closed()).is_err());
    drop(input);
    let (input, mut output, _) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, mut writer, guard) = fake.take();
    eof(input, &mut reader);
    assert_eq!(
        wait(writer.emit(OutputEvent::Completed(Summary::default())))
            .unwrap_err()
            .kind(),
        ErrorKind::Protocol
    );
    guard.finish(Ok(()));
    assert_eq!(wait(output.recv()).unwrap_err().kind(), ErrorKind::Protocol);
    let (input, mut output, _) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    guard.finish(Ok(()));
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::IncompleteResponse
    );
    drop(input);
    wait(client.shutdown()).unwrap();
}
#[test]
fn byte_limits_and_try_send_return_original_owned_chunk() {
    let (client, fake) = setup();
    let (mut input, output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, _, guard) = fake.take();
    let mut bytes = Vec::with_capacity(8192);
    bytes.extend([1, 2]);
    assert_eq!(
        wait(input.send(chunk(bytes))).unwrap_err().kind(),
        ErrorKind::LimitExceeded
    );
    input.try_send(chunk(vec![1, 2])).unwrap();
    let bytes = vec![3, 4];
    let pointer = bytes.as_ptr();
    let error = input.try_send(chunk(bytes)).unwrap_err();
    assert_eq!(error.error.kind(), ErrorKind::QueueFull);
    assert_eq!(error.chunk.pcm().as_bytes().as_ptr(), pointer);
    drop(wait(reader.recv()).unwrap());
    input.try_send(error.chunk).unwrap();
    drop(wait(reader.recv()).unwrap());
    eof(input, &mut reader);
    guard.finish(Ok(()));
    drop(output);
    wait(control.closed()).unwrap();
    wait(client.shutdown()).unwrap();
}
#[test]
fn global_budget_covers_all_sessions_and_backend_input_handoff() {
    let fake = Arc::new(Fake::default());
    let client = Client::with_driver(
        Limits {
            max_buffered_bytes: 4096,
            max_input_chunk_bytes: 2048,
            max_output_event_bytes: 2048,
            ..limits()
        },
        fake.clone(),
    )
    .unwrap();
    let (mut a, oa, ca) = client.start(RequestOptions::default()).unwrap().split();
    let (mut ra, _, ga) = fake.take();
    let (mut b, ob, cb) = client.start(RequestOptions::default()).unwrap().split();
    let (mut rb, _, gb) = fake.take();
    wait(a.send(chunk(vec![0; 1800]))).unwrap();
    let held = wait(ra.recv()).unwrap().unwrap();
    let result = wait(b.send(chunk(vec![0; 1800])));
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().kind(), ErrorKind::LimitExceeded);
    let (owned, lease) = held.into_parts();
    drop(owned);
    drop(lease);
    wait(b.send(chunk(vec![0; 1800]))).unwrap();
    drop(wait(rb.recv()).unwrap());
    eof(a, &mut ra);
    eof(b, &mut rb);
    ga.finish(Ok(()));
    gb.finish(Ok(()));
    drop(oa);
    drop(ob);
    wait(ca.closed()).unwrap();
    wait(cb.closed()).unwrap();
    wait(client.shutdown()).unwrap();
}
#[test]
fn shutdown_waits_for_worker_and_driver_cleanup_even_when_its_future_is_dropped() {
    let fake = Arc::new(Fake {
        hold_shutdown: true,
        ..Fake::default()
    });
    let client = Client::with_driver(limits(), fake.clone()).unwrap();
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    let mut shutdown = client.shutdown();
    pending(&mut shutdown);
    assert!(client.start(RequestOptions::default()).is_err());
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::ShuttingDown
    );
    drop(shutdown);
    guard.finish(Ok(()));
    let mut shutdown = client.shutdown();
    pending(&mut shutdown);
    fake.shutdown.lock().unwrap().take().unwrap().finish(Ok(()));
    wait(shutdown).unwrap();
    assert_eq!(
        wait(control.closed()).unwrap_err().kind(),
        ErrorKind::ShuttingDown
    );
    drop(input);
}
#[test]
fn only_last_client_drop_cancels_and_receiver_drop_cancels_sender() {
    let (client, fake) = setup();
    let clone = client.clone();
    let (mut input, output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    drop(clone);
    wait(input.send(chunk(vec![1, 2]))).unwrap();
    let mut sending = input.send(chunk(vec![3, 4]));
    pending(&mut sending);
    drop(output);
    assert_eq!(wait(sending).unwrap_err().kind(), ErrorKind::Cancelled);
    drop(guard);
    drop(input);
    drop(control);
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    drop(client);
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::ShuttingDown
    );
    drop(guard);
    assert!(wait(control.closed()).is_err());
    drop(input);
}
#[test]
fn multiple_closed_waiters_are_woken_and_wake_callbacks_can_reenter() {
    let (client, fake) = setup();
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    struct Reenter(Control);
    impl Wake for Reenter {
        fn wake(self: Arc<Self>) {
            let _ = self.0.progress();
        }
    }
    let waker = Waker::from(Arc::new(Reenter(control.clone())));
    let mut receive = output.recv();
    assert!(
        Pin::new(&mut receive)
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    let mut a = control.closed();
    let mut b = control.closed();
    pending(&mut a);
    pending(&mut b);
    control.cancel();
    assert!(wait(receive).is_err());
    drop(guard);
    assert!(wait(a).is_err());
    assert!(wait(b).is_err());
    drop(input);
    wait(client.shutdown()).unwrap();
}
#[test]
fn cancellation_and_backend_completion_race_has_one_terminal() {
    for _ in 0..50 {
        let (client, fake) = setup();
        let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
        let (mut reader, _, guard) = fake.take();
        eof(input, &mut reader);
        thread::scope(|scope| {
            scope.spawn(|| control.cancel());
            scope.spawn(|| guard.finish(Ok(())));
        });
        assert_eq!(
            wait(output.recv()).unwrap_err().kind(),
            ErrorKind::Cancelled
        );
        assert!(wait(output.recv()).unwrap().is_none());
        let _ = wait(control.closed());
        wait(client.shutdown()).unwrap();
    }
}
#[test]
fn common_handles_and_futures_are_send_and_client_is_shared() {
    fn shared<T: Send + Sync + Clone>() {}
    fn send<T: Send>(_: T) {}
    shared::<Client>();
    shared::<Control>();
    let (client, fake) = setup();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    send(input.send(chunk(vec![1, 2])));
    send(output.recv());
    send(control.closed());
    send(input.finish());
    drop(guard);
    drop(output);
    wait(client.shutdown()).unwrap();
}

#[test]
fn byte_capacity_applies_independently_of_item_capacity() {
    let fake = Arc::new(Fake::default());
    let client = Client::with_driver(
        Limits {
            input_queue_items: 8,
            output_queue_items: 8,
            input_queue_bytes: 2048,
            output_queue_bytes: 2048,
            ..limits()
        },
        fake.clone(),
    )
    .unwrap();
    let (mut input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, mut writer, guard) = fake.take();
    wait(input.send(chunk(vec![0; 1500]))).unwrap();
    let mut sending = input.send(chunk(vec![1; 1500]));
    pending(&mut sending);
    drop(wait(reader.recv()).unwrap());
    wait(sending).unwrap();
    drop(wait(reader.recv()).unwrap());
    eof(input, &mut reader);
    wait(writer.emit(audio(vec![0; 1500]))).unwrap();
    let mut emitting = writer.emit(audio(vec![1; 1500]));
    pending(&mut emitting);
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::Audio(_))
    ));
    wait(emitting).unwrap();
    guard.finish(Ok(()));
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::Audio(_))
    ));
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::Completed(_))
    ));
    wait(control.closed()).unwrap();
    wait(client.shutdown()).unwrap();
}

#[test]
fn oversized_output_and_abandoned_delivery_cannot_complete_successfully() {
    for oversized in [false, true] {
        let (client, fake) = setup();
        let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
        let (mut reader, mut writer, guard) = fake.take();
        eof(input, &mut reader);
        let expected = if oversized {
            let mut bytes = Vec::with_capacity(8192);
            bytes.extend([1, 2]);
            assert_eq!(
                wait(writer.emit(audio(bytes))).unwrap_err().kind(),
                ErrorKind::LimitExceeded
            );
            ErrorKind::LimitExceeded
        } else {
            wait(writer.emit(audio(vec![1, 2]))).unwrap();
            let mut emitting = writer.emit(audio(vec![3, 4]));
            pending(&mut emitting);
            drop(emitting);
            ErrorKind::IncompleteResponse
        };
        guard.finish(Ok(()));
        assert_eq!(wait(output.recv()).unwrap_err().kind(), expected);
        assert!(wait(output.recv()).unwrap().is_none());
        assert_eq!(wait(control.closed()).unwrap_err().kind(), expected);
        assert_eq!(control.request.core.state.lock().unwrap().bytes, 0);
        wait(client.shutdown()).unwrap();
    }
}

#[test]
fn duplicate_processing_finished_and_large_errors_are_bounded() {
    let (client, fake) = setup();
    let (input, mut output, _) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, mut writer, guard) = fake.take();
    eof(input, &mut reader);
    wait(writer.emit(OutputEvent::ProcessingFinished)).unwrap();
    assert_eq!(
        wait(writer.emit(OutputEvent::ProcessingFinished))
            .unwrap_err()
            .kind(),
        ErrorKind::Protocol
    );
    guard.finish(Ok(()));
    assert_eq!(wait(output.recv()).unwrap_err().kind(), ErrorKind::Protocol);
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    guard.fail(Error::new(ErrorKind::Transport, "あ".repeat(10000)));
    let error = wait(output.recv()).unwrap_err();
    assert_eq!(error.message().len(), 4095);
    assert_eq!(error.request_id(), Some(control.id()));
    guard.finish(Ok(()));
    assert_eq!(wait(control.closed()).unwrap_err(), error);
    drop(input);
    wait(client.shutdown()).unwrap();
}

#[test]
fn deadline_wakes_a_suspended_receiver() {
    let (client, fake) = setup();
    let options = RequestOptions {
        timeout: Some(Duration::from_millis(30)),
        ..Default::default()
    };
    let (input, mut output, _) = client.start(options).unwrap().split();
    let (_, _, guard) = fake.take();
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::DeadlineExceeded
    );
    drop(guard);
    drop(input);
    wait(client.shutdown()).unwrap();
}

#[test]
fn worker_panic_waits_for_resource_destruction_without_blocking_closed_poll() {
    let (client, fake) = setup();
    let (input, mut output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    struct Resource(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>);
    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.send(()).unwrap();
            self.1.recv_timeout(Duration::from_secs(5)).unwrap();
        }
    }
    let worker = thread::spawn(move || {
        let _guard = guard;
        let _resource = Resource(entered_tx, release_rx);
        panic!("injected backend panic");
    });
    entered_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    control.cancel();
    assert_eq!(
        wait(output.recv()).unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    let mut closed = control.closed();
    pending(&mut closed);
    let mut shutdown = client.shutdown();
    pending(&mut shutdown);
    release_tx.send(()).unwrap();
    assert_eq!(wait(closed).unwrap_err().kind(), ErrorKind::Cancelled);
    assert!(worker.join().is_err());
    wait(shutdown).unwrap();
    drop(input);
}

#[test]
fn dispatch_errors_and_panics_release_request_registration() {
    struct Broken(bool);
    impl Driver for Broken {
        fn launch(&self, _: RequestOptions, _: WorkerSession) -> Result<()> {
            if self.0 {
                panic!("dispatch panic");
            }
            Err(Error::new(ErrorKind::Transport, "dispatch rejected"))
        }
        fn shutdown(&self, _: DriverShutdown) {
            panic!("shutdown panic");
        }
    }
    for panics in [false, true] {
        let client = Client::with_driver(limits(), Arc::new(Broken(panics))).unwrap();
        for _ in 0..4 {
            let error = client.start(RequestOptions::default()).err().unwrap();
            assert_eq!(
                error.kind(),
                if panics {
                    ErrorKind::Inference
                } else {
                    ErrorKind::Transport
                }
            );
        }
        assert_eq!(
            wait(client.shutdown()).unwrap_err().kind(),
            ErrorKind::Inference
        );
    }
}

#[test]
fn shared_curve_layout_capacity_is_charged_and_curve_values_are_moved() {
    let mut names = Vec::with_capacity(1000);
    names.push("jawOpen".to_owned());
    let oversized = Arc::new(CurveLayout::new(LayoutId(1), names).unwrap());
    assert!(oversized.storage_bytes() >= 1000 * std::mem::size_of::<String>());
    let (client, fake) = setup();
    let (input, mut output, _) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, mut writer, guard) = fake.take();
    eof(input, &mut reader);
    let event =
        OutputEvent::Curves(CurveFrame::new(oversized, MediaTime::ZERO, vec![0.5]).unwrap());
    assert_eq!(
        wait(writer.emit(event)).unwrap_err().kind(),
        ErrorKind::LimitExceeded
    );
    guard.finish(Ok(()));
    assert!(wait(output.recv()).is_err());
    let (input, mut output, _) = client.start(RequestOptions::default()).unwrap().split();
    let (mut reader, mut writer, guard) = fake.take();
    eof(input, &mut reader);
    let layout = Arc::new(CurveLayout::new(LayoutId(2), vec!["jawOpen".into()]).unwrap());
    let values = vec![0.75];
    let pointer = values.as_ptr();
    wait(writer.emit(OutputEvent::Curves(
        CurveFrame::new(layout.clone(), MediaTime::ZERO, values).unwrap(),
    )))
    .unwrap();
    guard.finish(Ok(()));
    let Some(OutputEvent::Curves(frame)) = wait(output.recv()).unwrap() else {
        panic!()
    };
    assert!(Arc::ptr_eq(frame.layout(), &layout));
    assert_eq!(frame.values().as_ptr(), pointer);
    assert!(matches!(
        wait(output.recv()).unwrap(),
        Some(OutputEvent::Completed(_))
    ));
    wait(client.shutdown()).unwrap();
}

#[test]
fn all_registered_closed_waiters_receive_notifications() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Count(AtomicUsize);
    impl Wake for Count {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let (client, fake) = setup();
    let (input, output, control) = client.start(RequestOptions::default()).unwrap().split();
    let (_, _, guard) = fake.take();
    let counters: Vec<_> = (0..3)
        .map(|_| Arc::new(Count(AtomicUsize::new(0))))
        .collect();
    let mut futures: Vec<_> = (0..3).map(|_| control.closed()).collect();
    for (future, counter) in futures.iter_mut().zip(&counters) {
        let waker = Waker::from(counter.clone());
        assert!(
            Pin::new(future)
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
    }
    drop(guard);
    for (future, counter) in futures.into_iter().zip(counters) {
        assert!(counter.0.load(Ordering::SeqCst) > 0);
        assert_eq!(wait(future).unwrap_err().kind(), ErrorKind::Inference);
    }
    drop(input);
    drop(output);
    wait(client.shutdown()).unwrap();
}
