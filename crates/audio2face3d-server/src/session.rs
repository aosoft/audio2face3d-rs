use crate::{
    animation,
    backend::Backend,
    config::Config,
    proto::{
        controller::{
            self, animation_data_stream::StreamPart as Output, audio_stream::StreamPart as Input,
        },
        status,
    },
};
use std::sync::Arc;
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, mpsc, oneshot};
use tokio_stream::Stream;
use tokio_util::sync::CancellationToken;
use tonic::{Status, Streaming};

/// A separate terminal channel makes errors observable even when data is backed up.
pub struct ResponseStream {
    data: mpsc::Receiver<controller::AnimationDataStream>,
    terminal: Option<oneshot::Receiver<Result<(), Status>>>,
    ended: bool,
    permit: Option<Arc<OwnedSemaphorePermit>>,
}
impl ResponseStream {
    pub fn new(
        data: mpsc::Receiver<controller::AnimationDataStream>,
        terminal: oneshot::Receiver<Result<(), Status>>,
        permit: Arc<OwnedSemaphorePermit>,
    ) -> Self {
        Self {
            data,
            terminal: Some(terminal),
            ended: false,
            permit: Some(permit),
        }
    }
}
impl Stream for ResponseStream {
    type Item = Result<controller::AnimationDataStream, Status>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.ended {
            return Poll::Ready(None);
        }
        if let Some(terminal) = &mut self.terminal
            && let Poll::Ready(result) = Pin::new(terminal).poll(cx)
        {
            self.terminal = None;
            let result = result
                .unwrap_or_else(|_| Err(Status::internal("session worker stopped unexpectedly")));
            if let Err(error) = result {
                self.ended = true;
                self.data.close();
                self.permit.take();
                return Poll::Ready(Some(Err(error)));
            }
        }
        match self.data.poll_recv(cx) {
            Poll::Ready(None) if self.terminal.is_some() => Poll::Pending,
            Poll::Ready(Some(message)) => Poll::Ready(Some(Ok(message))),
            Poll::Ready(None) => {
                self.ended = true;
                self.permit.take();
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

pub async fn read_input(
    input: &mut Streaming<controller::AudioStream>,
    config: &Config,
    shutdown: &CancellationToken,
) -> Result<controller::AudioStream, Status> {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => Err(Status::unavailable("server shutting down")),
        result = tokio::time::timeout(config.input_timeout(), input.message()) => {
            result.map_err(|_| Status::deadline_exceeded("input idle timeout"))??
                .ok_or_else(|| Status::invalid_argument("input closed before EndOfAudio"))
        }
    }
}

async fn send(
    tx: &mpsc::Sender<controller::AnimationDataStream>,
    part: Output,
    config: &Config,
    shutdown: &CancellationToken,
) -> Result<(), Status> {
    tokio::select! {
        biased;
        _ = shutdown.cancelled() => Err(Status::unavailable("server shutting down")),
        result = tokio::time::timeout(config.output_timeout(), tx.send(controller::AnimationDataStream { stream_part: Some(part) })) => {
            result.map_err(|_| Status::deadline_exceeded("output backpressure timeout"))?
                .map_err(|_| Status::cancelled("response reader closed"))
        }
    }
}

pub async fn run(
    input: &mut Streaming<controller::AudioStream>,
    backend: &mut dyn Backend,
    tx: &mpsc::Sender<controller::AnimationDataStream>,
    config: &Config,
    shutdown: &CancellationToken,
) -> Result<(), Status> {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    send(
        tx,
        animation::header(epoch).stream_part.unwrap(),
        config,
        shutdown,
    )
    .await?;
    loop {
        let message = tokio::select! {
            _ = tx.closed() => return Err(Status::cancelled("response reader closed")),
            message = read_input(input, config, shutdown) => message?,
        };
        let finished = match message.stream_part {
            Some(Input::AudioWithEmotion(audio)) => {
                backend.push(audio)?;
                false
            }
            Some(Input::EndOfAudio(_)) => {
                backend.finish()?;
                true
            }
            Some(Input::AudioStreamHeader(_)) => {
                return Err(Status::invalid_argument("duplicate audio header"));
            }
            None => return Err(Status::invalid_argument("missing or unknown stream_part")),
        };
        while let Some(frame) = backend.next_frame() {
            send(tx, Output::AnimationData(frame), config, shutdown).await?;
        }
        if finished {
            send(
                tx,
                Output::Event(controller::Event {
                    event_type: 0,
                    metadata: None,
                }),
                config,
                shutdown,
            )
            .await?;
            send(
                tx,
                Output::Status(status::Status {
                    code: 0,
                    message: "Mock audio processing completed successfully (no inference).".into(),
                }),
                config,
                shutdown,
            )
            .await?;
            return Ok(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use tokio::sync::Semaphore;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn full_queue_times_out_and_error_bypasses_queued_data() {
        let config = Config::parse_from(["test", "--output-timeout-ms", "20"]);
        let shutdown = CancellationToken::new();
        let slots = Arc::new(Semaphore::new(1));
        let permit = Arc::new(slots.clone().acquire_owned().await.unwrap());
        let (tx, rx) = mpsc::channel(1);
        let (terminal_tx, terminal_rx) = oneshot::channel();
        let mut stream = ResponseStream::new(rx, terminal_rx, permit);
        tx.send(animation::header(0.0)).await.unwrap();
        let error = send(
            &tx,
            Output::Event(controller::Event::default()),
            &config,
            &shutdown,
        )
        .await
        .unwrap_err();
        assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
        terminal_tx.send(Err(error)).unwrap();
        assert_eq!(
            stream.next().await.unwrap().unwrap_err().code(),
            tonic::Code::DeadlineExceeded
        );
        assert!(stream.next().await.is_none());
        assert_eq!(slots.available_permits(), 1);
    }

    #[tokio::test]
    async fn success_drains_queued_output_before_releasing_slot() {
        let slots = Arc::new(Semaphore::new(1));
        let permit = Arc::new(slots.clone().acquire_owned().await.unwrap());
        let (tx, rx) = mpsc::channel(1);
        let (terminal_tx, terminal_rx) = oneshot::channel();
        let mut stream = ResponseStream::new(rx, terminal_rx, permit);
        tx.send(animation::header(0.0)).await.unwrap();
        terminal_tx.send(Ok(())).unwrap();
        drop(tx);
        assert_eq!(slots.available_permits(), 0);
        assert!(stream.next().await.unwrap().is_ok());
        assert_eq!(slots.available_permits(), 0);
        assert!(stream.next().await.is_none());
        assert_eq!(slots.available_permits(), 1);
    }
}
