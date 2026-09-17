use crate::client::{
    core::{Core, Request},
    driver::{Driver, WorkerSession},
    memory,
    notify::Subscription,
    session::Session,
    types::*,
};
use crate::logging::integration::LogScope;
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Instant,
};

/// Queue byte limits use retained storage charges. Each endpoint can additionally
/// stage one pending operation; max_buffered_bytes covers queues, pending operations,
/// request settings and backend input handoffs. This is not an allocator/RSS limit.
#[derive(Clone, Debug)]
pub struct Limits {
    pub max_requests: usize,
    pub max_request_bytes: usize,
    pub max_buffered_bytes: usize,
    pub input_queue_items: usize,
    pub input_queue_bytes: usize,
    pub max_input_chunk_bytes: usize,
    pub output_queue_items: usize,
    pub output_queue_bytes: usize,
    pub max_output_event_bytes: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            max_requests: 64,
            max_request_bytes: 256 * 1024,
            max_buffered_bytes: 64 * 1024 * 1024,
            input_queue_items: 16,
            input_queue_bytes: 2 * 1024 * 1024,
            max_input_chunk_bytes: 1024 * 1024,
            output_queue_items: 16,
            output_queue_bytes: 2 * 1024 * 1024,
            max_output_event_bytes: 1024 * 1024,
        }
    }
}
impl Limits {
    pub fn validate(&self) -> Result<()> {
        if [
            self.max_requests,
            self.max_request_bytes,
            self.max_buffered_bytes,
            self.input_queue_items,
            self.input_queue_bytes,
            self.max_input_chunk_bytes,
            self.output_queue_items,
            self.output_queue_bytes,
            self.max_output_event_bytes,
        ]
        .contains(&0)
            || self.max_input_chunk_bytes > self.input_queue_bytes
            || self.max_output_event_bytes > self.output_queue_bytes
            || self.max_request_bytes > self.max_buffered_bytes
            || self.max_input_chunk_bytes > self.max_buffered_bytes
            || self.max_output_event_bytes > self.max_buffered_bytes
        {
            return Err(Error::invalid("invalid client queue/storage limits"));
        }
        Ok(())
    }
}
/// Clones share request capacity, memory budget and backend resources.
#[derive(Clone)]
pub struct Client {
    owner: Arc<Owner>,
}
struct Owner {
    scope: LogScope,
    core: Arc<Core>,
    driver: Arc<dyn Driver>,
}
impl Owner {
    fn shutdown(&self) {
        let _scope = self.scope.enter();
        if self.core.begin_shutdown() {
            let completion = crate::client::driver::DriverShutdown::new(self.core.clone());
            let driver = self.driver.clone();
            // Driver callbacks run outside all core locks. Their completion owns cleanup.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                driver.shutdown(completion)
            }));
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        self.shutdown();
    }
}
impl Client {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn with_driver(limits: Limits, driver: Arc<dyn Driver>) -> Result<Self> {
        limits.validate()?;
        Ok(Self {
            owner: Arc::new(Owner {
                scope: LogScope::capture(),
                core: Core::new(limits)?,
                driver,
            }),
        })
    }
    /// Local registration only. Does not wait for a response, model load or execution slot.
    pub fn start(&self, options: RequestOptions) -> Result<Session> {
        let _scope = self.owner.scope.enter();
        options.validate()?;
        if options.input_format.channels() != 1
            || ![16000, 44100, 48000].contains(&options.input_format.sample_rate())
        {
            return Err(Error::invalid(
                "expected mono PCM16 at 16000/44100/48000 Hz",
            ));
        }
        let deadline = options
            .timeout
            .map(|d| {
                Instant::now()
                    .checked_add(d)
                    .ok_or_else(|| Error::invalid("request deadline exceeds clock range"))
            })
            .transpose()?;
        let bytes = memory::request(&options);
        let (request, lease) = Request::register(
            self.owner.core.clone(),
            options.input_format,
            deadline,
            bytes,
        )?;
        #[cfg(feature = "tracing")]
        let span = tracing::error_span!("client_request", id = request.id.0);
        #[cfg(feature = "tracing")]
        let _span = span.enter();
        let session = Session::new(request.clone());
        let worker = WorkerSession::new(request.clone(), lease);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.owner.driver.launch(options, worker)
        })) {
            Ok(Ok(())) => Ok(session),
            result => {
                let error = match result {
                    Ok(Err(error)) => error,
                    Err(_) => Error::new(ErrorKind::Inference, "backend dispatch panicked"),
                    Ok(Ok(())) => unreachable!(),
                };
                request.fail(error.clone());
                drop(session);
                Err(error.with_request(request.id))
            }
        }
    }
    /// Starts shutdown immediately, even if the returned Future is never polled.
    pub fn shutdown(&self) -> Shutdown {
        self.owner.shutdown();
        Shutdown {
            core: self.owner.core.clone(),
            subscription: self.owner.core.notify.subscribe(),
        }
    }
}
pub struct Shutdown {
    core: Arc<Core>,
    subscription: Subscription,
}
impl Future for Shutdown {
    type Output = Result<()>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.subscription.register(cx.waker());
        let state = self.core.state.lock().unwrap();
        if state.requests.is_empty()
            && let Some(result) = &state.driver_closed
        {
            return Poll::Ready(result.clone());
        }
        Poll::Pending
    }
}
