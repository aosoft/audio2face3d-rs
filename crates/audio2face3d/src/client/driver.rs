//! Private adapter boundary. A driver dispatches without blocking and owns cleanup.
//! Keep WorkerGuard until native/RPC resources and accepted input leases are released.
use crate::client::{
    core::{Buffered, Core, Lease, Outcome, Request},
    memory,
    notify::Subscription,
    types::*,
};
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

pub(crate) trait Driver: Send + Sync {
    fn launch(&self, options: RequestOptions, session: WorkerSession) -> Result<()>;
    fn shutdown(&self, completion: DriverShutdown);
}
pub(crate) struct DriverShutdown {
    core: Arc<Core>,
    done: bool,
}
impl DriverShutdown {
    pub fn new(core: Arc<Core>) -> Self {
        Self { core, done: false }
    }
    pub fn finish(mut self, result: Result<()>) {
        self.done = true;
        self.core.driver_closed(result);
    }
}
impl Drop for DriverShutdown {
    fn drop(&mut self) {
        if !self.done {
            self.core.driver_closed(Err(Error::new(
                ErrorKind::Inference,
                "backend shutdown stopped unexpectedly",
            )));
        }
    }
}
pub(crate) struct WorkerSession {
    request: Arc<Request>,
    guard: WorkerGuard,
}
impl WorkerSession {
    pub fn new(request: Arc<Request>, lease: Lease) -> Self {
        Self {
            request: request.clone(),
            guard: WorkerGuard {
                request,
                settings: Some(lease),
                done: false,
                unexpected: ErrorKind::Inference,
            },
        }
    }
    pub fn split(self) -> (Reader, Writer, WorkerGuard) {
        (
            Reader {
                request: self.request.clone(),
            },
            Writer {
                request: self.request,
            },
            self.guard,
        )
    }
}
pub(crate) struct WorkerGuard {
    request: Arc<Request>,
    settings: Option<Lease>,
    unexpected: ErrorKind,
    done: bool,
}
impl WorkerGuard {
    #[cfg(feature = "client-grpc")]
    pub fn runtime_owned(&mut self) {
        self.unexpected = ErrorKind::RuntimeUnavailable;
    }
    pub fn fail(&self, error: Error) {
        self.request.fail(error);
    }
    pub fn cancelled(&self) -> Cancelled {
        Cancelled {
            request: self.request.clone(),
            subscription: self.request.notify.subscribe(),
        }
    }
    pub fn finish(mut self, result: Result<()>) {
        self.done = true;
        drop(self.settings.take());
        self.request.closed(result);
    }
}
impl Drop for WorkerGuard {
    fn drop(&mut self) {
        if !self.done {
            drop(self.settings.take());
            self.request.closed(Err(Error::new(
                self.unexpected,
                "backend worker stopped unexpectedly",
            )));
        }
    }
}
pub(crate) struct Cancelled {
    request: Arc<Request>,
    subscription: Subscription,
}
impl Future for Cancelled {
    type Output = Error;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Error> {
        self.subscription.register(cx.waker());
        self.request
            .state
            .lock()
            .unwrap()
            .error()
            .map_or(Poll::Pending, Poll::Ready)
    }
}
pub(crate) struct Reader {
    request: Arc<Request>,
}
pub(crate) struct InputPacket(Buffered<InputChunk>);
impl InputPacket {
    pub fn chunk(&self) -> &InputChunk {
        &self.0.value
    }
    pub fn into_parts(self) -> (InputChunk, Lease) {
        (self.0.value, self.0.lease)
    }
}
impl Reader {
    pub fn recv(&mut self) -> ReadInput<'_> {
        let subscription = self.request.notify.subscribe();
        ReadInput {
            reader: self,
            subscription,
        }
    }
}
pub(crate) struct ReadInput<'a> {
    reader: &'a mut Reader,
    subscription: Subscription,
}
impl Future for ReadInput<'_> {
    type Output = Result<Option<InputPacket>>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.subscription.register(cx.waker());
        self.reader.poll_recv(cx)
    }
}
impl Reader {
    #[cfg_attr(not(feature = "client-grpc"), allow(dead_code))]
    pub fn subscription(&self) -> Subscription {
        self.request.notify.subscribe()
    }
    pub fn poll_recv(&self, _cx: &mut Context<'_>) -> Poll<Result<Option<InputPacket>>> {
        let request = &self.request;
        let mut state = request.state.lock().unwrap();
        if let Some(error) = state.error() {
            return Poll::Ready(Err(error));
        }
        if let Some(item) = state.input.pop_front() {
            state.input_bytes -= item.bytes();
            drop(state);
            request.changed();
            return Poll::Ready(Ok(Some(InputPacket(item))));
        }
        if state.input_finished {
            state.input_eof = true;
            drop(state);
            request.changed();
            return Poll::Ready(Ok(None));
        }
        Poll::Pending
    }
}
pub(crate) struct Writer {
    request: Arc<Request>,
}
impl Writer {
    pub fn emit(&mut self, event: OutputEvent) -> Emit<'_> {
        let result = self.stage(event);
        if let Err(error) = &result {
            self.request.fail(error.clone());
        }
        let subscription = self.request.notify.subscribe();
        Emit {
            writer: self,
            staged: result.is_ok(),
            error: result.err(),
            subscription,
        }
    }
    fn stage(&self, event: OutputEvent) -> Result<()> {
        let (event, bytes) = memory::output(event)?;
        if bytes > self.request.core.limits.max_output_event_bytes {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "output event too large",
            ));
        }
        let lease = self.request.core.reserve(bytes)?;
        let item = Buffered {
            value: event,
            lease,
        };
        let mut state = self.request.state.lock().unwrap();
        if let Some(error) = state.error() {
            return Err(error);
        }
        if !matches!(state.outcome, Outcome::Running) {
            return Err(Error::new(
                ErrorKind::Protocol,
                "output after backend completion",
            ));
        }
        if matches!(item.value, OutputEvent::ProcessingFinished) {
            if state.processing_finished {
                return Err(Error::new(
                    ErrorKind::Protocol,
                    "duplicate processing-finished event",
                ));
            }
            state.processing_finished = true;
        }
        match &item.value {
            OutputEvent::Audio(a) => {
                state.progress.received_audio_bytes += a.pcm().as_bytes().len() as u64
            }
            OutputEvent::Curves(_) => state.progress.received_curve_frames += 1,
            _ => {}
        }
        assert!(state.pending_output.is_none());
        state.pending_output = Some(item);
        Ok(())
    }
}
pub(crate) struct Emit<'a> {
    writer: &'a mut Writer,
    staged: bool,
    error: Option<Error>,
    subscription: Subscription,
}
impl Future for Emit<'_> {
    type Output = Result<()>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.subscription.register(cx.waker());
        if let Some(error) = self.error.take() {
            return Poll::Ready(Err(error));
        }
        let request = self.writer.request.clone();
        let mut state = request.state.lock().unwrap();
        if let Some(error) = state.error() {
            self.staged = false;
            return Poll::Ready(Err(error));
        }
        let bytes = state
            .pending_output
            .as_ref()
            .expect("emit polled after completion")
            .bytes();
        let limits = &request.core.limits;
        if state.output.len() >= limits.output_queue_items
            || bytes > limits.output_queue_bytes - state.output_bytes
        {
            return Poll::Pending;
        }
        let item = state.pending_output.take().unwrap();
        state.output_bytes += bytes;
        state.output.push_back(item);
        drop(state);
        self.staged = false;
        request.changed();
        Poll::Ready(Ok(()))
    }
}
impl Drop for Emit<'_> {
    fn drop(&mut self) {
        if self.staged {
            self.writer.request.fail(Error::new(
                ErrorKind::IncompleteResponse,
                "pending output delivery abandoned",
            ));
        }
    }
}
