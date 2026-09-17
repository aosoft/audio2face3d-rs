//! Standard-Future completions and a native-owner control worker.
use crate::logging::integration::LogScope;
use crate::types::{Error, ErrorKind, Result};
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, mpsc},
    task::{Context, Poll, Wake, Waker},
    thread,
};

struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
/// Only used on independent control workers (and runtime-free tests), never in Future::poll.
pub(crate) fn wait<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(value) => return value,
            Poll::Pending => thread::park(),
        }
    }
}
struct State<T> {
    result: Option<Result<T>>,
    waker: Option<Waker>,
}
pub(crate) struct Completion<T> {
    state: Arc<Mutex<State<T>>>,
}
struct Sender<T> {
    state: Arc<Mutex<State<T>>>,
    sent: bool,
}
fn channel<T>() -> (Sender<T>, Completion<T>) {
    let state = Arc::new(Mutex::new(State {
        result: None,
        waker: None,
    }));
    (
        Sender {
            state: state.clone(),
            sent: false,
        },
        Completion { state },
    )
}
impl<T> Sender<T> {
    fn finish(mut self, result: Result<T>) {
        self.sent = true;
        let waker = {
            let mut state = self.state.lock().unwrap();
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}
impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        if self.sent {
            return;
        }
        let waker = {
            let mut state = self.state.lock().unwrap();
            if state.result.is_none() {
                state.result = Some(Err(Error::new(
                    ErrorKind::Inference,
                    "inference control worker stopped unexpectedly",
                )));
            }
            state.waker.take()
        };
        if let Some(waker) = waker {
            waker.wake();
        }
    }
}
impl<T> Future for Completion<T> {
    type Output = Result<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock().unwrap();
        if let Some(result) = state.result.take() {
            Poll::Ready(result)
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
struct JobResult {
    publish: Box<dyn FnOnce() + Send>,
    failed: bool,
}
type Job<T> = Box<dyn FnOnce(&mut T) -> JobResult + Send>;
/// State is constructed, used and destroyed on the same independent standard thread.
/// Dropping a completion never cancels its job; close is ordered after all prior jobs.
pub(crate) struct Worker<T> {
    tx: mpsc::Sender<Job<T>>,
}
impl<T: Send + 'static> Worker<T> {
    pub(crate) async fn start(init: impl FnOnce() -> Result<T> + Send + 'static) -> Result<Self> {
        let (tx, rx) = mpsc::channel::<Job<T>>();
        let (ready_tx, ready) = channel();
        let scope = LogScope::capture();
        thread::Builder::new()
            .name("a2f-control".into())
            .spawn(move || {
                let _scope = scope.enter();
                let mut state = match init() {
                    Ok(state) => {
                        ready_tx.finish(Ok(()));
                        state
                    }
                    Err(error) => {
                        ready_tx.finish(Err(error));
                        return;
                    }
                };
                for job in rx {
                    let result = job(&mut state);
                    if result.failed {
                        // Finish native destruction BEFORE exposing worker failure to its owner.
                        drop(state);
                        (result.publish)();
                        return;
                    }
                    (result.publish)();
                }
                // Native destructors drain failed/in-flight execution before this thread exits.
                drop(state);
            })
            .map_err(|e| Error::new(ErrorKind::Inference, e.to_string()))?;
        let worker = Self { tx };
        ready.await?;
        Ok(worker)
    }
    pub(crate) fn call<R: Send + 'static>(
        &mut self,
        op: impl FnOnce(&mut T) -> Result<R> + Send + 'static,
    ) -> Completion<R> {
        let (tx, result) = channel();
        // A disconnected worker drops tx and completes result with an error.
        let scope = LogScope::capture();
        let _ = self.tx.send(Box::new(move |state| {
            let _scope = scope.enter();
            let (result, failed) =
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op(state))) {
                    Ok(result) => (result, false),
                    Err(_) => (
                        Err(Error::new(
                            ErrorKind::Inference,
                            "inference control worker panicked",
                        )),
                        true,
                    ),
                };
            JobResult {
                publish: Box::new(move || tx.finish(result)),
                failed,
            }
        }));
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    #[test]
    fn dropped_completion_does_not_skip_work_or_release_before_native_drop() {
        let count = Arc::new(AtomicUsize::new(0));
        struct Native {
            owner: thread::ThreadId,
            count: Arc<AtomicUsize>,
        }
        impl Drop for Native {
            fn drop(&mut self) {
                assert_eq!(self.owner, thread::current().id());
                self.count.fetch_add(1, Ordering::SeqCst);
            }
        }
        let observed = count.clone();
        let mut worker = wait(Worker::start(move || {
            Ok(Some(Native {
                owner: thread::current().id(),
                count: observed,
            }))
        }))
        .unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let abandoned = worker.call(move |_| {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        });
        started_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        drop(abandoned);
        let mut closing = Box::pin(worker.call(|state| {
            drop(state.take());
            Ok(())
        }));
        assert!(
            closing
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        assert_eq!(count.load(Ordering::SeqCst), 0);
        release_tx.send(()).unwrap();
        wait(closing).unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn abandoned_initialization_still_destroys_native_state_on_owner() {
        let (start_tx, start_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (drop_tx, drop_rx) = mpsc::channel();
        struct Native(mpsc::Sender<thread::ThreadId>);
        impl Drop for Native {
            fn drop(&mut self) {
                self.0.send(thread::current().id()).unwrap();
            }
        }
        let mut loading = Box::pin(Worker::start(move || {
            start_tx.send(thread::current().id()).unwrap();
            release_rx.recv().unwrap();
            Ok(Native(drop_tx))
        }));
        assert!(
            loading
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        let owner = start_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        drop(loading);
        release_tx.send(()).unwrap();
        assert_eq!(
            drop_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap(),
            owner
        );
        assert_ne!(owner, thread::current().id());
    }
    #[test]
    fn worker_panic_is_published_only_after_native_destruction() {
        let (dropping_tx, dropping_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        struct Native(mpsc::Sender<()>, mpsc::Receiver<()>);
        impl Drop for Native {
            fn drop(&mut self) {
                self.0.send(()).unwrap();
                self.1.recv().unwrap();
            }
        }
        let mut worker = wait(Worker::start(move || Ok(Native(dropping_tx, release_rx)))).unwrap();
        let mut result = Box::pin(worker.call::<()>(|_| panic!("simulated worker failure")));
        dropping_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(
            result
                .as_mut()
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
        release_tx.send(()).unwrap();
        assert_eq!(wait(result).unwrap_err().kind(), ErrorKind::Inference);
        assert_eq!(
            wait(worker.call(|_| Ok(()))).unwrap_err().kind(),
            ErrorKind::Inference
        );
    }
}
