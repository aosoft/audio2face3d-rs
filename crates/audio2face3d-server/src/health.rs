use crate::{
    auth::{Authenticator, RpcMethod, gate::AuthGate},
    lifecycle::Metrics,
    request::RequestContext,
};
use audio2face3d::logging::integration::LogScope;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, atomic::Ordering},
};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tonic::{Request, Response, Status};
use tonic_health::{
    pb::{HealthCheckRequest, HealthCheckResponse, health_server::Health},
    server::{HealthReporter, HealthService as Inner},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HealthAuth {
    #[default]
    Public,
    SameAsInference,
}
pub(crate) struct HealthService<A> {
    inner: Inner,
    gate: Arc<AuthGate<A>>,
    policy: HealthAuth,
    shutdown: CancellationToken,
    workers: TaskTracker,
    metrics: Arc<Metrics>,
    scope: LogScope,
}
impl<A: Authenticator> HealthService<A> {
    pub(crate) fn new(
        reporter: HealthReporter,
        gate: Arc<AuthGate<A>>,
        policy: HealthAuth,
        shutdown: CancellationToken,
        workers: TaskTracker,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            inner: Inner::from_health_reporter(reporter),
            gate,
            policy,
            shutdown,
            workers,
            metrics,
            scope: LogScope::capture(),
        }
    }
    async fn authorize(
        &self,
        request: &Request<HealthCheckRequest>,
        context: RequestContext,
        method: RpcMethod,
    ) -> Result<(), Status> {
        if self.shutdown.is_cancelled() {
            return Err(Status::unavailable("server shutting down"));
        }
        if self.policy == HealthAuth::SameAsInference
            && let Err(error) = self
                .gate
                .authorize(request, context.id, method, &self.shutdown)
                .await
        {
            self.metrics
                .authentication_rejections
                .fetch_add(1, Ordering::Relaxed);
            return Err(error);
        }
        Ok(())
    }
}
impl<A: Authenticator> Health for HealthService<A> {
    fn check<'a, 'f>(
        &'a self,
        request: Request<HealthCheckRequest>,
    ) -> Pin<Box<dyn Future<Output = Result<Response<HealthCheckResponse>, Status>> + Send + 'f>>
    where
        'a: 'f,
        Self: 'f,
    {
        self.metrics.health_requests.fetch_add(1, Ordering::Relaxed);
        let context = RequestContext::new(self.gate.next_id(), request.metadata());
        Box::pin(self.scope.wrap_future(async move {
            let context = context?;
            context
                .run(async {
                    self.authorize(&request, context, RpcMethod::HealthCheck)
                        .await?;
                    self.inner.check(Request::new(request.into_inner())).await
                })
                .await
        }))
    }
    type WatchStream = ReceiverStream<Result<HealthCheckResponse, Status>>;
    fn watch<'a, 'f>(
        &'a self,
        request: Request<HealthCheckRequest>,
    ) -> Pin<Box<dyn Future<Output = Result<Response<Self::WatchStream>, Status>> + Send + 'f>>
    where
        'a: 'f,
        Self: 'f,
    {
        self.metrics.health_requests.fetch_add(1, Ordering::Relaxed);
        let context = RequestContext::new(self.gate.next_id(), request.metadata());
        Box::pin(self.scope.wrap_future(async move {
            let context = context?;
            context.run(async {
                self.authorize(&request, context, RpcMethod::HealthWatch).await?;
                let mut input = self.inner.watch(Request::new(request.into_inner())).await?.into_inner();
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                let shutdown = self.shutdown.clone();
                self.workers.spawn(LogScope::capture().wrap_future(async move {
                    loop {
                        tokio::select! {
                            biased;
                            _ = shutdown.cancelled() => break,
                            _ = tx.closed() => break,
                            _ = context.expired() => { let _ = tx.try_send(Err(Status::deadline_exceeded("RPC deadline exceeded"))); break; }
                            value = input.next() => {
                                let Some(value) = value else { break; };
                                tokio::select! {
                                    _ = shutdown.cancelled() => break,
                                    _ = context.expired() => break,
                                    sent = tx.send(value) => if sent.is_err() { break; }
                                }
                            }
                        }
                    }
                }));
                Ok(Response::new(ReceiverStream::new(rx)))
            }).await
        }))
    }
}
