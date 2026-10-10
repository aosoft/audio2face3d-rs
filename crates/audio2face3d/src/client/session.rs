use crate::client::{
    core::{Buffered, Request},
    memory,
    notify::Subscription,
    state::Received,
    types::*,
};
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

/// One admitted request. Split into independent sending, receiving and control handles.
pub struct Session {
    input: Input,
    output: Output,
    control: Control,
}
impl Session {
    pub(crate) fn new(request: Arc<Request>) -> Self {
        Self {
            input: Input {
                request: request.clone(),
                finished: false,
            },
            output: Output {
                request: request.clone(),
                ended: false,
            },
            control: Control { request },
        }
    }
    pub fn id(&self) -> RequestId {
        self.control.id()
    }
    pub fn split(self) -> (Input, Output, Control) {
        (self.input, self.output, self.control)
    }
}
/// Owned sending endpoint. Dropping without awaiting finish cancels the request.
pub struct Input {
    pub(crate) request: Arc<Request>,
    finished: bool,
}
#[derive(Debug)]
pub struct TrySendError {
    pub error: Error,
    pub chunk: InputChunk,
}
impl Input {
    pub fn send(&mut self, chunk: InputChunk) -> SendChunk<'_> {
        let result = self.stage(chunk);
        let subscription = self.request.notify.subscribe();
        SendChunk {
            input: self,
            staged: result.is_ok(),
            error: result.err(),
            subscription,
        }
    }
    fn stage(&self, chunk: InputChunk) -> Result<()> {
        let (chunk, bytes) = memory::input(chunk)?;
        if bytes > self.request.core.limits.max_input_chunk_bytes {
            return Err(self.request.context(Error::new(
                ErrorKind::LimitExceeded,
                "input chunk too large",
            )));
        }
        chunk
            .validate(self.request.format)
            .map_err(|e| self.request.context(e))?;
        let lease = self
            .request
            .core
            .reserve(bytes)
            .map_err(|e| self.request.context(e))?;
        let mut item = Some(Buffered {
            value: chunk,
            lease,
        });
        {
            let mut state = self.request.state.lock().unwrap();
            state.stage_input(&mut item)?;
        }
        Ok(())
    }
    /// Nonblocking ownership-preserving alternative. QueueFull can be retried after draining.
    #[allow(
        clippy::result_large_err,
        reason = "Returns rejected input without allocating on a full queue"
    )]
    pub fn try_send(&mut self, chunk: InputChunk) -> std::result::Result<(), TrySendError> {
        let (chunk, bytes) = memory::input(chunk).expect("validated PCM storage");
        let mut chunk = Some(chunk);
        let result = (|| {
            if bytes > self.request.core.limits.max_input_chunk_bytes {
                return Err(Error::new(
                    ErrorKind::LimitExceeded,
                    "input chunk too large",
                ));
            }
            chunk.as_ref().unwrap().validate(self.request.format)?;
            let mut state = self.request.state.lock().unwrap();
            let limits = &self.request.core.limits;
            state.check_input(bytes, limits)?;
            let lease = self.request.core.reserve(bytes)?;
            state.enqueue_input(Buffered {
                value: chunk.take().unwrap(),
                lease,
            });
            Ok(())
        })();
        match result {
            Ok(()) => {
                self.request.changed();
                Ok(())
            }
            Err(error) => Err(TrySendError {
                error: self.request.context(error),
                chunk: chunk.unwrap(),
            }),
        }
    }
    /// Consumes the input handle. Dropping this Future before its first poll cancels input.
    pub fn finish(self) -> Finish {
        Finish { input: Some(self) }
    }
}
impl Drop for Input {
    fn drop(&mut self) {
        if !self.finished {
            self.request.fail(Error::new(
                ErrorKind::Cancelled,
                "input dropped before finish",
            ));
        }
    }
}
pub struct SendChunk<'a> {
    input: &'a mut Input,
    staged: bool,
    error: Option<Error>,
    subscription: Subscription,
}
impl Future for SendChunk<'_> {
    type Output = Result<()>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.subscription.register(cx.waker());
        if let Some(error) = self.error.take() {
            return Poll::Ready(Err(error));
        }
        let request = self.input.request.clone();
        let result = request
            .state
            .lock()
            .unwrap()
            .accept_input(&request.core.limits);
        if result.is_pending() {
            return result;
        }
        self.staged = false;
        request.changed();
        result
    }
}
impl Drop for SendChunk<'_> {
    fn drop(&mut self) {
        if self.staged {
            let item = self.input.request.state.lock().unwrap().abandon_input();
            drop(item);
            self.input.request.changed();
        }
    }
}
/// Ordered end-of-input Future. Dropping before it is polled cancels the request.
pub struct Finish {
    input: Option<Input>,
}
impl Future for Finish {
    type Output = Result<()>;
    fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut input = self.input.take().expect("finish polled after completion");
        {
            let mut state = input.request.state.lock().unwrap();
            if let Err(error) = state.finish_input() {
                return Poll::Ready(Err(error));
            }
        }
        input.finished = true;
        input.request.changed();
        Poll::Ready(Ok(()))
    }
}
/// Receiving endpoint. Dropping before terminal consumption cancels the request.
pub struct Output {
    pub(crate) request: Arc<Request>,
    ended: bool,
}
impl Output {
    /// Receives one owned event. Failure is reported once, then None.
    /// Successful output ends with Completed, then None. A pending Future can be
    /// dropped and recreated without consuming an event; its Waker subscription
    /// belongs to that Future and must remain alive when awaiting a notification.
    pub fn recv(&mut self) -> Receive<'_> {
        let subscription = self.request.notify.subscribe();
        Receive {
            output: self,
            subscription,
        }
    }
}
impl Drop for Output {
    fn drop(&mut self) {
        if !self.ended {
            self.request
                .fail(Error::new(ErrorKind::Cancelled, "output dropped"));
        }
    }
}
pub struct Receive<'a> {
    output: &'a mut Output,
    subscription: Subscription,
}
impl Future for Receive<'_> {
    type Output = Result<Option<OutputEvent>>;
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.output.ended {
            return Poll::Ready(Ok(None));
        }
        self.subscription.register(cx.waker());
        let request = self.output.request.clone();
        let result = request.state.lock().unwrap().receive();
        let value = match result {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Err(error)) => {
                self.output.ended = true;
                Err(error)
            }
            Poll::Ready(Ok(Received::Complete(summary))) => {
                self.output.ended = true;
                Ok(Some(OutputEvent::Completed(summary)))
            }
            Poll::Ready(Ok(Received::Event(Buffered { value, lease }))) => {
                drop(lease);
                Ok(Some(value))
            }
        };
        request.changed();
        Poll::Ready(value)
    }
}
#[derive(Clone)]
/// Cloneable cancellation and cleanup handle; dropping it alone does not cancel.
pub struct Control {
    pub(crate) request: Arc<Request>,
}
impl Control {
    pub fn id(&self) -> RequestId {
        self.request.id
    }
    pub fn cancel(&self) {
        self.request
            .fail(Error::new(ErrorKind::Cancelled, "request cancelled"));
    }
    pub fn progress(&self) -> Progress {
        self.request.state.lock().unwrap().progress()
    }
    /// Cleanup completion, independently of whether successful output has been consumed.
    pub fn closed(&self) -> Closed {
        Closed {
            request: self.request.clone(),
            subscription: self.request.notify.subscribe(),
        }
    }
}
/// Owned Future acknowledging backend cleanup, independently of output consumption.
pub struct Closed {
    request: Arc<Request>,
    subscription: Subscription,
}
impl Future for Closed {
    type Output = Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.subscription.register(cx.waker());
        self.request.state.lock().unwrap().closed()
    }
}
