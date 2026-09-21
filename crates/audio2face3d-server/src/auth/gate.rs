use super::{AuthError, AuthRequest, Authenticator, Principal, RequestId, RpcMethod, bearer};
use std::{
    future::Future,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;
use tonic::{Request, Status};

pub(crate) struct AuthGate<A> {
    authenticator: Option<Arc<A>>,
    slots: Semaphore,
    timeout: Duration,
    next_id: AtomicU64,
}
impl<A: Authenticator> AuthGate<A> {
    pub(crate) fn new(authenticator: Option<Arc<A>>) -> Self {
        Self {
            authenticator,
            slots: Semaphore::new(64),
            timeout: Duration::from_secs(5),
            next_id: AtomicU64::new(1),
        }
    }
    pub(crate) fn next_id(&self) -> RequestId {
        RequestId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }
    pub(crate) async fn authorize<T>(
        &self,
        request: &Request<T>,
        id: RequestId,
        method: RpcMethod,
        shutdown: &CancellationToken,
    ) -> Result<Option<Principal>, Status> {
        let started = std::time::Instant::now();
        let result=async {
        if shutdown.is_cancelled() {
            return Err(Status::unavailable("server shutting down"));
        }
        let Some(authenticator) = &self.authenticator else {
            return Ok(None);
        };
        let key = bearer::parse(request.metadata())?;
        let _permit = self
            .slots
            .try_acquire()
            .map_err(|_| Status::resource_exhausted("authentication capacity exhausted"))?;
        let request = AuthRequest {
            api_key: key,
            request_id: id,
            method,
            peer_addr: request.remote_addr(),
        };
        let future = catch_unwind(AssertUnwindSafe(|| authenticator.authenticate(request)))
            .map_err(|_| Status::internal("authentication failed internally"))?;
        tokio::pin!(future);
        let protected = std::future::poll_fn(|cx| {
            match catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(cx))) {
                Ok(poll) => poll.map(|r| r.map_err(status)),
                Err(_) => std::task::Poll::Ready(Err(Status::internal(
                    "authentication failed internally",
                ))),
            }
        });
        let result = tokio::select! {
            biased;
            _ = shutdown.cancelled() => Err(Status::unavailable("server shutting down")),
            result = tokio::time::timeout(self.timeout, protected) => result.unwrap_or_else(|_| Err(Status::unavailable("authentication timed out"))),
        };
        result.map(Some)
        }.await;
        // Request outcome is logged at the RPC boundary; this is detailed auth diagnostics.
        let code = result.as_ref().err().map_or(tonic::Code::Ok, Status::code);
        let level = if matches!(method, RpcMethod::ProcessAudioStream) || result.is_ok() {
            audio2face3d::logging::LogLevel::Debug
        } else {
            crate::diagnostics::classification(code).0
        };
        audio2face3d::logging::integration::log(level, || {
            audio2face3d::logging::LogRecord::new("authentication finished")
                .field("source", module_path!())
                .field("rpc_id", id.0)
                .field("method", format!("{method:?}"))
                .field("code", format!("{code:?}"))
                .field("accepted", result.is_ok())
                .field("elapsed_us", started.elapsed().as_micros() as u64)
        });
        result
    }
}
fn status(error: AuthError) -> Status {
    match error {
        AuthError::InvalidCredential => Status::unauthenticated("invalid credential"),
        AuthError::Forbidden => Status::permission_denied("request forbidden"),
        AuthError::Unavailable => Status::unavailable("authentication unavailable"),
    }
}

#[cfg(test)]
mod tests;
