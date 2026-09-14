use crate::{
    audio::validate_header,
    backend::{Backend, mock::MockBackend},
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
use tokio::sync::{Semaphore, mpsc, oneshot};
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tonic::{Request, Response, Status, Streaming};
use tracing::Instrument;

pub struct Service {
    config: Config,
    slots: Arc<Semaphore>,
    shutdown: CancellationToken,
    workers: TaskTracker,
    next_id: AtomicU64,
}
impl Service {
    pub fn new(config: Config, shutdown: CancellationToken, workers: TaskTracker) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(config.max_streams)),
            config,
            shutdown,
            workers,
            next_id: AtomicU64::new(1),
        }
    }
}
#[tonic::async_trait]
impl A2fControllerService for Service {
    type ProcessAudioStreamStream = ResponseStream;
    async fn process_audio_stream(
        &self,
        request: Request<Streaming<AudioStream>>,
    ) -> Result<Response<ResponseStream>, Status> {
        if self.shutdown.is_cancelled() {
            return Err(Status::unavailable("server shutting down"));
        }
        let permit = self
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("concurrent stream limit reached"))?;
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
        let mut backend = MockBackend::start(&self.config, &header);
        let (tx, rx) = mpsc::channel(self.config.output_queue_capacity);
        let (terminal_tx, terminal_rx) = oneshot::channel();
        let config = self.config.clone();
        let shutdown = self.shutdown.clone();
        self.workers.spawn(
            async move {
                let _permit = permit;
                tracing::info!("started");
                let result = session::run(&mut input, &mut backend, &tx, &config, &shutdown).await;
                if let Err(error) = &result {
                    backend.cancel();
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
        )))
    }
}
