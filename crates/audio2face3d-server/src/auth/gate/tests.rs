use super::*;
use std::{
    pin::Pin,
    sync::atomic::AtomicUsize,
    task::{Context, Poll},
};
struct Counts {
    entered: AtomicUsize,
    dropped: AtomicUsize,
}
struct Borrowing {
    counts: Arc<Counts>,
}
struct Borrowed<'a> {
    owner: &'a Borrowing,
    request: AuthRequest<'a>,
}
impl Future for Borrowed<'_> {
    type Output = super::super::AuthResult;
    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        if self.request.api_key.expose() == "poll-panic" {
            panic!("test panic without credentials");
        }
        Poll::Pending
    }
}
impl Drop for Borrowed<'_> {
    fn drop(&mut self) {
        self.owner.counts.dropped.fetch_add(1, Ordering::SeqCst);
    }
}
impl Authenticator for Borrowing {
    type Future<'a> = Borrowed<'a>;
    fn authenticate<'a>(&'a self, request: AuthRequest<'a>) -> Borrowed<'a> {
        self.counts.entered.fetch_add(1, Ordering::SeqCst);
        if request.api_key.expose() == "call-panic" {
            panic!("test call panic without credentials");
        }
        Borrowed {
            owner: self,
            request,
        }
    }
}
fn request(key: &str) -> Request<()> {
    let mut r = Request::new(());
    r.metadata_mut()
        .insert("authorization", format!("Bearer {key}").parse().unwrap());
    r
}
#[tokio::test]
async fn capacity_timeout_stop_and_panics_return_every_permit() {
    let counts = Arc::new(Counts {
        entered: AtomicUsize::new(0),
        dropped: AtomicUsize::new(0),
    });
    let mut gate = AuthGate::new(Some(Arc::new(Borrowing {
        counts: counts.clone(),
    })));
    gate.timeout = Duration::from_millis(25);
    let stop = CancellationToken::new();
    for key in ["call-panic", "poll-panic", "pending"] {
        let r = request(key);
        let result = gate
            .authorize(&r, gate.next_id(), RpcMethod::ProcessAudioStream, &stop)
            .await;
        assert_eq!(
            result.unwrap_err().code(),
            if key == "pending" {
                tonic::Code::Unavailable
            } else {
                tonic::Code::Internal
            }
        );
        assert_eq!(gate.slots.available_permits(), 64);
    }
    let gate = Arc::new(gate);

    // Poll all 64 borrowed futures once without a timeout race.
    let r = request("pending");
    let mut waiting = vec![];
    for _ in 0..64 {
        let mut f =
            Box::pin(gate.authorize(&r, gate.next_id(), RpcMethod::ProcessAudioStream, &stop));
        std::future::poll_fn(|cx| {
            assert!(f.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
        waiting.push(f);
    }
    assert_eq!(gate.slots.available_permits(), 0);
    assert_eq!(
        gate.authorize(&r, gate.next_id(), RpcMethod::HealthCheck, &stop)
            .await
            .unwrap_err()
            .code(),
        tonic::Code::ResourceExhausted
    );
    drop(waiting);
    assert_eq!(gate.slots.available_permits(), 64);
    let mut f = Box::pin(gate.authorize(&r, gate.next_id(), RpcMethod::ProcessAudioStream, &stop));
    std::future::poll_fn(|cx| {
        assert!(f.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    })
    .await;
    stop.cancel();
    assert_eq!(f.await.unwrap_err().code(), tonic::Code::Unavailable);
    assert_eq!(gate.slots.available_permits(), 64);
    assert_eq!(
        counts.entered.load(Ordering::SeqCst),
        counts.dropped.load(Ordering::SeqCst) + 1
    ); // generation panic creates no future
}

#[tokio::test]
async fn core_logging_and_status_never_include_credentials() {
    use audio2face3d::{
        Audio2Face3DContext,
        logging::{LogLevel, Logger, integration::LogScope},
    };
    #[derive(Default)]
    struct Sink(std::sync::Mutex<Vec<String>>);
    impl Logger for Sink {
        fn log_level(&self) -> LogLevel {
            LogLevel::Trace
        }
        fn write_log(&self, _: LogLevel, message: audio2face3d::logging::LogRecord) {
            let message = format!("{} {:?}", message.message, message.fields);
            self.0.lock().unwrap().push(message);
        }
    }
    let sink = Arc::new(Sink::default());
    let context = Audio2Face3DContext::builder().logger(sink.clone()).build();
    let gate = AuthGate::new(Some(Arc::new(|r: AuthRequest<'_>| {
        assert!(!format!("{r:?}").contains("SentinelSecret"));
        Err(AuthError::InvalidCredential)
    })));
    let r = request("SentinelSecret");
    let stop = CancellationToken::new();
    let error = LogScope::new(context)
        .wrap_future(gate.authorize(&r, gate.next_id(), RpcMethod::ProcessAudioStream, &stop))
        .await
        .unwrap_err();
    assert!(!format!("{error:?}").contains("SentinelSecret"));
    let logs = sink.0.lock().unwrap();
    assert!(!logs.is_empty());
    assert!(logs.iter().all(|line| !line.contains("SentinelSecret")));
}
