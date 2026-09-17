//! A bounded-by-session-count task set on one standard control thread.
use crate::types::*;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, mpsc},
    task::{Context, Poll, Wake, Waker},
    thread,
};
type Task = Pin<Box<dyn Future<Output = ()> + Send>>;
enum Message {
    Run(Task),
    Stop(Task),
}
pub(crate) struct Executor {
    tx: mpsc::Sender<Message>,
    thread: thread::Thread,
    stopping: std::sync::atomic::AtomicBool,
}
struct ThreadWake(thread::Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
impl Executor {
    pub fn new() -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let handle = thread::Builder::new()
            .name("a2f-direct-control".into())
            .spawn(move || {
                let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
                let mut cx = Context::from_waker(&waker);
                let mut tasks: Vec<Task> = vec![];
                let mut stopping = None;
                loop {
                    while let Ok(message) = rx.try_recv() {
                        match message {
                            Message::Run(task) => tasks.push(task),
                            Message::Stop(task) => stopping = Some(task),
                        }
                    }
                    let mut i = 0;
                    while i < tasks.len() {
                        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            tasks[i].as_mut().poll(&mut cx)
                        }));
                        if !matches!(result, Ok(Poll::Pending)) {
                            drop(tasks.swap_remove(i));
                        } else {
                            i += 1;
                        }
                    }
                    if tasks.is_empty()
                        && let Some(task) = stopping.as_mut()
                        && task.as_mut().poll(&mut cx).is_ready()
                    {
                        break;
                    }
                    thread::park();
                }
            })
            .map_err(|e| Error::new(ErrorKind::RuntimeUnavailable, e.to_string()))?;
        Ok(Self {
            tx,
            thread: handle.thread().clone(),
            stopping: std::sync::atomic::AtomicBool::new(false),
        })
    }
    pub fn spawn(&self, task: impl Future<Output = ()> + Send + 'static) -> Result<()> {
        self.tx.send(Message::Run(Box::pin(task))).map_err(|_| {
            Error::new(
                ErrorKind::RuntimeUnavailable,
                "direct control worker stopped",
            )
        })?;
        self.thread.unpark();
        Ok(())
    }
    pub fn stop(&self, task: impl Future<Output = ()> + Send + 'static) {
        if self
            .stopping
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return;
        }
        let _ = self.tx.send(Message::Stop(Box::pin(task)));
        self.thread.unpark();
    }
}
struct State<T> {
    result: Option<Result<T>>,
    waker: Option<Waker>,
}
pub(crate) struct Sender<T>(Arc<Mutex<State<T>>>);
pub(crate) struct Receiver<T>(Arc<Mutex<State<T>>>);
pub(crate) fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let state = Arc::new(Mutex::new(State {
        result: None,
        waker: None,
    }));
    (Sender(state.clone()), Receiver(state))
}
impl<T> Sender<T> {
    pub fn send(self, result: Result<T>) {
        let wake = {
            let mut s = self.0.lock().unwrap();
            s.result = Some(result);
            s.waker.take()
        };
        if let Some(w) = wake {
            w.wake();
        }
    }
}
impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let wake = {
            let mut s = self.0.lock().unwrap();
            if s.result.is_none() {
                s.result = Some(Err(Error::new(
                    ErrorKind::RuntimeUnavailable,
                    "initialization worker stopped",
                )));
            }
            s.waker.take()
        };
        if let Some(w) = wake {
            w.wake();
        }
    }
}
impl<T> Future for Receiver<T> {
    type Output = Result<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let w = cx.waker().clone();
        let mut s = self.0.lock().unwrap();
        if let Some(r) = s.result.take() {
            Poll::Ready(r)
        } else {
            let old = s.waker.replace(w);
            drop(s);
            drop(old);
            Poll::Pending
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.stop(async {});
    }
}
