//! Cleanup belongs to the request task and must survive RPC cancellation.
use crate::{diagnostics::RequestLog, request::RequestContext};
use std::future::Future;
use tonic::Status;

pub(crate) async fn finish(
    context: RequestContext,
    streaming: Result<(), Status>,
    cleanup: impl Future<Output = Result<(), Status>>,
    success: impl Future<Output = Result<(), Status>>,
    observation: &mut RequestLog,
) -> Result<(), Status> {
    let failure_stage = observation.stage;
    observation.stage = "cleanup";
    // This future is intentionally outside RequestContext::run. In particular,
    // a native close may have already moved its worker into this future.
    let cleanup = cleanup.await;
    observation.cleanup_failed |= cleanup.is_err();
    if streaming.is_err() && cleanup.is_ok() {
        observation.stage = failure_stage;
    }
    streaming.and(cleanup)?;
    observation.stage = "output";
    context.run(success).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        auth::RequestId,
        config::Config,
        session::{self, ResponseStream},
    };
    use audio2face3d::inference::{Cancellation, admission::Admission};
    use std::sync::Arc;
    use tokio::sync::{mpsc, oneshot};
    use tokio_stream::StreamExt;
    use tokio_util::sync::CancellationToken;

    #[tokio::test]
    async fn deadline_cancellation_and_response_drop_keep_cleanup_and_permit_alive() {
        for mode in 0..3 {
            let config = Config::builder(crate::config::BackendKind::Mock)
                .build_settings()
                .unwrap();
            let admission = Admission::new(1, 1, std::time::Duration::ZERO).unwrap();
            let permit = Arc::new(admission.acquire(&Cancellation::new()).await.unwrap());
            let cancel = CancellationToken::new();
            let deadline = (mode == 0)
                .then(|| tokio::time::Instant::now() + std::time::Duration::from_millis(20));
            let context = RequestContext {
                id: RequestId(1),
                deadline,
            };
            let (tx, rx) = mpsc::channel(2);
            let (terminal, terminal_rx) = oneshot::channel();
            let mut stream = ResponseStream::new_with_deadline(
                rx,
                terminal_rx,
                permit.clone(),
                cancel.clone(),
                deadline,
            );
            let (entered, started) = oneshot::channel();
            let (release, gate) = oneshot::channel();
            let worker_cancel = cancel.clone();
            let worker = tokio::spawn(async move {
                let _permit = permit;
                let mut observation = RequestLog::new(worker_cancel.clone());
                let result = finish(
                    context,
                    Ok(()),
                    async {
                        entered.send(()).unwrap();
                        gate.await.unwrap();
                        Ok(())
                    },
                    session::send_success(&tx, "ok", &config, &worker_cancel),
                    &mut observation,
                )
                .await;
                let code = result.as_ref().err().map(Status::code);
                let _ = terminal.send(result);
                code
            });
            started.await.unwrap();
            match mode {
                0 => {
                    let error = stream.next().await.unwrap().unwrap_err();
                    assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
                }
                1 => cancel.cancel(),
                _ => {}
            }
            drop(stream);
            assert!(!worker.is_finished());
            assert_eq!(admission.available_permits(), 0);
            release.send(()).unwrap();
            let code = worker.await.unwrap().unwrap();
            assert_eq!(
                code,
                if mode == 0 {
                    tonic::Code::DeadlineExceeded
                } else {
                    tonic::Code::Unavailable
                }
            );
            assert_eq!(admission.available_permits(), 1);
        }
    }

    #[tokio::test]
    async fn cleanup_failure_suppresses_success_and_preserves_streaming_error_priority() {
        for streaming_failed in [false, true] {
            let config = Config::builder(crate::config::BackendKind::Mock)
                .build_settings()
                .unwrap();
            let cancel = CancellationToken::new();
            let mut observation = RequestLog::new(cancel.clone());
            observation.stage = "input";
            let (tx, mut rx) = mpsc::channel(2);
            let streaming = if streaming_failed {
                Err(Status::invalid_argument("input failed"))
            } else {
                Ok(())
            };
            let result = finish(
                RequestContext {
                    id: RequestId(1),
                    deadline: None,
                },
                streaming,
                async { Err(Status::internal("cleanup failed")) },
                session::send_success(&tx, "ok", &config, &cancel),
                &mut observation,
            )
            .await;
            assert_eq!(
                result.unwrap_err().code(),
                if streaming_failed {
                    tonic::Code::InvalidArgument
                } else {
                    tonic::Code::Internal
                }
            );
            assert!(observation.cleanup_failed);
            assert_eq!(observation.stage, "cleanup");
            assert!(rx.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn successful_cleanup_preserves_wire_event_then_success() {
        let config = Config::builder(crate::config::BackendKind::Mock)
            .build_settings()
            .unwrap();
        let cancel = CancellationToken::new();
        let mut observation = RequestLog::new(cancel.clone());
        let (tx, mut rx) = mpsc::channel(2);
        finish(
            RequestContext {
                id: RequestId(1),
                deadline: None,
            },
            Ok(()),
            async { Ok(()) },
            session::send_success(&tx, "ok", &config, &cancel),
            &mut observation,
        )
        .await
        .unwrap();
        assert!(matches!(
            rx.recv().await.unwrap().stream_part,
            Some(crate::proto::controller::animation_data_stream::StreamPart::Event(_))
        ));
        assert!(matches!(
            rx.recv().await.unwrap().stream_part,
            Some(
                crate::proto::controller::animation_data_stream::StreamPart::Status(
                    crate::proto::status::Status { code: 0, .. }
                )
            )
        ));
    }
}
