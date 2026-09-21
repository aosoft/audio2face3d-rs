use crate::ServerError;
use audio2face3d::logging::integration::LogScope;
use std::{
    future::Future,
    pin::Pin,
    sync::atomic::{AtomicU64, Ordering},
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

/// Server observations, not client receipt or playback acknowledgements.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShutdownReport {
    pub inference_requests: u64,
    pub health_requests: u64,
    pub authentication_rejections: u64,
    pub inference_workers_started: u64,
    pub inference_workers_finished: u64,
}
#[derive(Default)]
pub(crate) struct Metrics {
    pub inference_requests: AtomicU64,
    pub health_requests: AtomicU64,
    pub authentication_rejections: AtomicU64,
    pub inference_workers_started: AtomicU64,
    pub inference_workers_finished: AtomicU64,
}
impl Metrics {
    pub fn snapshot(&self) -> ShutdownReport {
        ShutdownReport {
            inference_requests: self.inference_requests.load(Ordering::Relaxed),
            health_requests: self.health_requests.load(Ordering::Relaxed),
            authentication_rejections: self.authentication_rejections.load(Ordering::Relaxed),
            inference_workers_started: self.inference_workers_started.load(Ordering::Relaxed),
            inference_workers_finished: self.inference_workers_finished.load(Ordering::Relaxed),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CleanupStage {
    Prepare,
    Transport,
    Workers,
    PreparedResources,
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Overdue {
    pub stage: CleanupStage,
    pub unfinished: usize,
}

/// Wait for the supervisor; dropping this handle does not cancel cleanup.
/// Keep the caller's Tokio runtime alive until this completes.
pub struct CleanupCompletion {
    receiver: oneshot::Receiver<Result<ShutdownReport, ServerError>>,
}
impl std::fmt::Debug for CleanupCompletion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CleanupCompletion { .. }")
    }
}
impl Future for CleanupCompletion {
    type Output = Result<ShutdownReport, ServerError>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.receiver)
            .poll(cx)
            .map(|r| r.unwrap_or(Err(ServerError::SupervisorStopped)))
    }
}
struct StopOnDrop(CancellationToken);
impl Drop for StopOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

/// The spawned task owns all cleanup; the caller only owns notification and waiting.
pub(crate) async fn supervise<F, Fut>(
    stop: impl Future<Output = ()> + Send,
    run: F,
) -> Result<ShutdownReport, ServerError>
where
    F: FnOnce(CancellationToken, oneshot::Sender<Overdue>) -> Fut,
    Fut: Future<Output = Result<ShutdownReport, ServerError>> + Send + 'static,
{
    let shutdown = CancellationToken::new();
    let _guard = StopOnDrop(shutdown.clone());
    let (complete_tx, receiver) = oneshot::channel();
    let (overdue_tx, mut overdue_rx) = oneshot::channel();
    let task = run(shutdown.clone(), overdue_tx);
    let scope = LogScope::capture();
    tokio::spawn(scope.wrap_future(async move {
        let _ = complete_tx.send(task.await);
    }));
    let mut completion = CleanupCompletion { receiver };
    tokio::pin!(stop);
    let mut stopping = false;
    let mut notified = false;
    loop {
        tokio::select! {
            biased;
            result = &mut completion => return result,
            notice = &mut overdue_rx, if !notified => {
                notified = true;
                if let Ok(notice) = notice {
                    return Err(ServerError::ShutdownTimeout { stage: notice.stage, unfinished: notice.unfinished, completion });
                }
            }
            _ = &mut stop, if !stopping => { stopping = true; shutdown.cancel(); }
        }
    }
}
pub(crate) fn notify(
    sender: &mut Option<oneshot::Sender<Overdue>>,
    stage: CleanupStage,
    unfinished: usize,
) {
    audio2face3d::logging::integration::log(audio2face3d::logging::LogLevel::Warn, || {
        audio2face3d::logging::LogRecord::new("server cleanup stage timed out")
            .field("source", module_path!())
            .field("stage", format!("{stage:?}"))
            .field("unfinished", unfinished as u64)
    });
    if let Some(sender) = sender.take() {
        let _ = sender.send(Overdue { stage, unfinished });
    }
}
/// A timeout only notifies the waiter. The same pinned operation is then awaited.
pub(crate) async fn finish_stage<T>(
    future: impl Future<Output = T>,
    timeout: Duration,
    sender: &mut Option<oneshot::Sender<Overdue>>,
    stage: CleanupStage,
    unfinished: usize,
) -> T {
    tokio::pin!(future);
    match tokio::time::timeout(timeout, &mut future).await {
        Ok(result) => result,
        Err(_) => {
            notify(sender, stage, unfinished);
            future.await
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    #[tokio::test]
    async fn timeout_handle_and_dropped_waiter_preserve_exactly_once_cleanup() {
        for drop_handle in [false, true] {
            let complete = Arc::new(AtomicUsize::new(0));
            let counter = complete.clone();
            let result = supervise(std::future::ready(()), move |stop, sender| async move {
                stop.cancelled().await;
                let mut sender = Some(sender);
                finish_stage(
                    async {
                        tokio::time::sleep(Duration::from_millis(40)).await;
                        counter.fetch_add(1, Ordering::SeqCst);
                    },
                    Duration::from_millis(5),
                    &mut sender,
                    CleanupStage::Workers,
                    1,
                )
                .await;
                Ok(ShutdownReport::default())
            })
            .await;
            let Err(ServerError::ShutdownTimeout {
                completion,
                stage: CleanupStage::Workers,
                unfinished: 1,
            }) = result
            else {
                panic!("missing completion handle")
            };
            assert_eq!(complete.load(Ordering::SeqCst), 0);
            if drop_handle {
                drop(completion);
                tokio::time::sleep(Duration::from_millis(60)).await;
            } else {
                completion.await.unwrap();
            }
            assert_eq!(complete.load(Ordering::SeqCst), 1);
        }
    }
    #[tokio::test]
    async fn dropping_serve_notifies_supervisor_without_blocking_drop() {
        let complete = Arc::new(AtomicUsize::new(0));
        let counter = complete.clone();
        let task = tokio::spawn(supervise(
            std::future::pending(),
            move |stop, _| async move {
                stop.cancelled().await;
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(ShutdownReport::default())
            },
        ));
        tokio::task::yield_now().await;
        task.abort();
        let _ = task.await;
        tokio::time::timeout(Duration::from_secs(1), async {
            while complete.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
    }
}

#[cfg(test)]
mod logging_tests {
    use super::*;
    use audio2face3d::{
        Audio2Face3DContext,
        logging::{LogLevel, LogRecord, Logger},
    };
    use std::sync::{Arc, Mutex};
    #[derive(Default)]
    struct Sink(Mutex<Vec<(LogLevel, LogRecord)>>);
    impl Logger for Sink {
        fn log_level(&self) -> LogLevel {
            LogLevel::Trace
        }
        fn write_log(&self, level: LogLevel, record: LogRecord) {
            self.0.lock().unwrap().push((level, record));
        }
    }
    #[tokio::test]
    async fn overdue_stage_logs_but_still_waits_for_cleanup() {
        let sink = Arc::new(Sink::default());
        let scope = LogScope::new(Audio2Face3DContext::builder().logger(sink.clone()).build());
        let (tx, rx) = oneshot::channel();
        let mut tx = Some(tx);
        let (release, complete) = oneshot::channel();
        let waiting = scope.wrap_future(finish_stage(
            async { complete.await.unwrap() },
            Duration::from_millis(1),
            &mut tx,
            CleanupStage::Workers,
            3,
        ));
        let (value, notice) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(waiting, async {
                let notice = rx.await.unwrap();
                release.send(42).unwrap();
                notice
            })
        })
        .await
        .unwrap();
        assert_eq!(value, 42);
        assert_eq!(notice.unfinished, 3);
        let logs = sink.0.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].0, LogLevel::Warn);
        assert!(
            logs[0]
                .1
                .fields
                .contains(&("stage".into(), "Workers".into()))
        );
        assert!(
            logs[0]
                .1
                .fields
                .contains(&("unfinished".into(), 3_u64.into()))
        );
    }
}
