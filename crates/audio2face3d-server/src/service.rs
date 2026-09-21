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

pub struct Service<A> {
    scope: LogScope,
    metrics: Arc<crate::lifecycle::Metrics>,
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
        metrics: Arc<crate::lifecycle::Metrics>,
    ) -> Result<Self, Status> {
        Ok(Self {
            scope: LogScope::capture(),
            metrics,
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
        self.metrics
            .inference_requests
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let scope = self.scope.clone();
        let id = self.gate.next_id();
        let request_context = crate::request::RequestContext::new(id, request.metadata());
        let scope = scope.field("rpc_id", id.0);
        Box::pin(scope.wrap_future(async move {
            audio2face3d::logging::integration::log(audio2face3d::logging::LogLevel::Info, || {
                audio2face3d::logging::LogRecord::new("received").field("source", module_path!())
            });
            let mut observation = Some(crate::diagnostics::RequestLog::new(self.shutdown.clone()));
            let result = async {
                let request_context = request_context?;
                request_context
                    .run(async {
                        if self.shutdown.is_cancelled() {
                            return Err(Status::unavailable("server shutting down"));
                        }
                        observation.as_mut().unwrap().stage = "authentication";
                        let _principal = self
                            .gate
                            .authorize(
                                &request,
                                request_context.id,
                                RpcMethod::ProcessAudioStream,
                                &self.shutdown,
                            )
                            .await
                            .inspect_err(|_| {
                                self.metrics
                                    .authentication_rejections
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            })?;
                        let mut input = request.into_inner();
                        observation.as_mut().unwrap().stage = "admission";
                        let waiting = std::time::Instant::now();
                        LogScope::capture().log(audio2face3d::logging::LogLevel::Debug, || {
                            audio2face3d::logging::LogRecord::new("waiting for execution slot")
                                .field("source", module_path!())
                        });
                        let permit = self.admission.acquire(&self.shutdown).await?;
                        LogScope::capture().log(audio2face3d::logging::LogLevel::Debug, || {
                            audio2face3d::logging::LogRecord::new("execution slot acquired")
                                .field("source", module_path!())
                                .field("wait_us", waiting.elapsed().as_micros() as u64)
                        });
                        let permit = Arc::new(permit);
                        let stream_permit = permit.clone();
                        observation.as_mut().unwrap().stage = "input_header";
                        let first =
                            session::read_input(&mut input, &self.config, &self.shutdown).await?;
                        let Some(StreamPart::AudioStreamHeader(header)) = first.stream_part else {
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
                        let metrics = self.metrics.clone();
                        metrics
                            .inference_workers_started
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let mut observation = observation.take().unwrap();
                        self.workers.spawn(worker_scope.wrap_future(async move {
                            let _permit = permit;
                            audio2face3d::logging::integration::log(
                                audio2face3d::logging::LogLevel::Info,
                                || {
                                    audio2face3d::logging::LogRecord::new("started")
                                        .field("source", module_path!())
                                },
                            );
                            let result = async {
                                observation.stage = "backend_start";
                                let mut backend = factory.start(&config, &header).await?;
                                observation.stage = "streaming";
                                let result = request_context
                                    .run(session::run(
                                        &mut input,
                                        &mut backend,
                                        &tx,
                                        &config,
                                        &shutdown,
                                        &mut observation,
                                    ))
                                    .await;
                                let failure_stage = observation.stage;
                                observation.stage = "cleanup";
                                let cleanup = backend.close().await;
                                observation.cleanup_failed |= cleanup.is_err();
                                if result.is_err() && cleanup.is_ok() {
                                    observation.stage = failure_stage;
                                }
                                result.and(cleanup)
                            }
                            .await;
                            observation.finish(&result);
                            metrics
                                .inference_workers_finished
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let _ = terminal_tx.send(result);
                        }));
                        Ok(Response::new(LogScope::capture().in_scope(|| {
                            ResponseStream::new_with_deadline(
                                rx,
                                terminal_rx,
                                stream_permit,
                                stream_cancel,
                                request_context.deadline,
                            )
                        })))
                    })
                    .await
            }
            .await;
            if let Some(mut observation) = observation {
                observation.finish(&result);
            }
            result
        }))
    }
}
