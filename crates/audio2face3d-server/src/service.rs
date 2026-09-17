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
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::{future::Future, pin::Pin};
use tokio::sync::{mpsc, oneshot};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tonic::{Request, Response, Status, Streaming};
use tracing::Instrument;

pub struct Service {
    config: Config,
    admission: Admission,
    shutdown: CancellationToken,
    workers: TaskTracker,
    next_id: AtomicU64,
    factory: Arc<Factory>,
}
impl Service {
    pub fn new(
        config: Config,
        shutdown: CancellationToken,
        workers: TaskTracker,
        factory: Arc<Factory>,
    ) -> Result<Self, Status> {
        Ok(Self {
            admission: Admission::new(
                config.max_streams,
                config.request_queue_capacity,
                std::time::Duration::from_millis(config.request_queue_timeout_ms),
            )?,
            config,
            shutdown,
            workers,
            next_id: AtomicU64::new(1),
            factory,
        })
    }
}
impl A2fControllerService for Service {
    type ProcessAudioStreamStream = ResponseStream;
    fn process_audio_stream<'borrow, 'future>(
        &'borrow self,
        request: Request<Streaming<AudioStream>>,
    ) -> Pin<Box<dyn Future<Output = Result<Response<ResponseStream>, Status>> + Send + 'future>>
    where
        'borrow: 'future,
        Self: 'future,
    {
        Box::pin(async move {
            if self.shutdown.is_cancelled() {
                return Err(Status::unavailable("server shutting down"));
            }
            let permit = self.admission.acquire(&self.shutdown).await?;
            let permit = Arc::new(permit);
            let stream_permit = permit.clone();
            let mut input = request.into_inner();
            let first = session::read_input(&mut input, &self.config, &self.shutdown).await?;
            let Some(StreamPart::AudioStreamHeader(header)) = first.stream_part else {
                return Err(Status::invalid_argument(
                    "first message must be AudioStreamHeader",
                ));
            };
            validate_header(header.audio_header.as_ref())?;
            let id = self.next_id.fetch_add(1, Ordering::Relaxed);
            let span = tracing::info_span!("rpc", id);
            let factory = self.factory.clone();
            let (tx, rx) = mpsc::channel(self.config.output_queue_capacity);
            let (terminal_tx, terminal_rx) = oneshot::channel();
            let config = self.config.clone();
            let stream_cancel = self.shutdown.child_token();
            let shutdown = stream_cancel.clone();
            self.workers.spawn(
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
                .instrument(span),
            );
            Ok(Response::new(ResponseStream::new(
                rx,
                terminal_rx,
                stream_permit,
                stream_cancel,
            )))
        })
    }
}
