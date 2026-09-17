use crate::auth::{Authenticator, RpcMethod, gate::AuthGate};
use crate::{
    admission::Admission,
    audio::validate_header,
    backend::Factory,
    config::Config,
    proto::{
        A2fControllerService,
        controller::{AudioStream, audio_stream::StreamPart},
    },
    session::{self, ResponseStream},
};
use audio2face3d::logging::integration::LogScope;
use std::sync::Arc;
use std::{future::Future, pin::Pin};
use tokio::sync::{mpsc, oneshot};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tonic::{Request, Response, Status, Streaming};
use tracing::Instrument;

pub struct Service<A> {
    scope: LogScope,
    config: Config,
    admission: Admission,
    shutdown: CancellationToken,
    workers: TaskTracker,
    gate: Arc<AuthGate<A>>,
    factory: Arc<Factory>,
}
impl<A: Authenticator> Service<A> {
    pub fn new(
        config: Config,
        shutdown: CancellationToken,
        workers: TaskTracker,
        factory: Arc<Factory>,
        gate: Arc<AuthGate<A>>,
    ) -> Result<Self, Status> {
        Ok(Self {
            scope: LogScope::capture(),
            admission: Admission::new(
                config.max_streams,
                config.request_queue_capacity,
                std::time::Duration::from_millis(config.request_queue_timeout_ms),
            )?,
            config,
            shutdown,
            workers,
            gate,
            factory,
        })
    }
}
impl<A: Authenticator> A2fControllerService for Service<A> {
    type ProcessAudioStreamStream = ResponseStream;
    fn process_audio_stream<'borrow, 'future>(
        &'borrow self,
        request: Request<Streaming<AudioStream>>,
    ) -> Pin<Box<dyn Future<Output = Result<Response<ResponseStream>, Status>> + Send + 'future>>
    where
        'borrow: 'future,
        Self: 'future,
    {
        let scope = self.scope.clone();
        let request_context =
            crate::request::RequestContext::new(self.gate.next_id(), request.metadata());
        let id = request_context.as_ref().map(|c| c.id.0).unwrap_or(0);
        let span = scope.in_scope(|| tracing::error_span!("rpc", id));
        let handler_span = span.clone();
        Box::pin(
            scope.wrap_future(
                async move {
                    let request_context = request_context?;
                    request_context
                        .run(async move {
                            if self.shutdown.is_cancelled() {
                                return Err(Status::unavailable("server shutting down"));
                            }
                            let _principal = self
                                .gate
                                .authorize(
                                    &request,
                                    request_context.id,
                                    RpcMethod::ProcessAudioStream,
                                    &self.shutdown,
                                )
                                .await?;
                            let mut input = request.into_inner();
                            let permit = self.admission.acquire(&self.shutdown).await?;
                            let permit = Arc::new(permit);
                            let stream_permit = permit.clone();
                            let first =
                                session::read_input(&mut input, &self.config, &self.shutdown)
                                    .await?;
                            let Some(StreamPart::AudioStreamHeader(header)) = first.stream_part
                            else {
                                return Err(Status::invalid_argument(
                                    "first message must be AudioStreamHeader",
                                ));
                            };
                            validate_header(header.audio_header.as_ref())?;
                            let factory = self.factory.clone();
                            let (tx, rx) = mpsc::channel(self.config.output_queue_capacity);
                            let (terminal_tx, terminal_rx) = oneshot::channel();
                            let config = self.config.clone();
                            let stream_cancel = self.shutdown.child_token();
                            let shutdown = stream_cancel.clone();
                            let worker_scope = LogScope::capture();
                            self.workers.spawn(worker_scope.wrap_future(
                async move {
                    let _permit = permit;
                    tracing::info!("started");
                    let result = async {
                        let mut backend = factory.start(&config, &header).await?;
                        let result =
                            session::run(&mut input, &mut backend, &tx, &config, &shutdown).await;
                        let cleanup = backend.close().await;
                        result.and(cleanup)
                    }
                    .await;
                    if let Err(error) = &result {
                        tracing::warn!(code = ?error.code(), message = error.message(), "failed");
                    } else {
                        tracing::info!("completed");
                    }
                    let _ = terminal_tx.send(result);
                }
                .instrument(span.clone()),
            ));
                            Ok(Response::new(span.in_scope(|| {
                                ResponseStream::new(rx, terminal_rx, stream_permit, stream_cancel)
                            })))
                        })
                        .await
                }
                .instrument(handler_span),
            ),
        )
    }
}
