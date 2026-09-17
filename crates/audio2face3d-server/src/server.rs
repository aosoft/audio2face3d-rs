use crate::{
    config::Config,
    proto::{A2fControllerServiceServer, SERVICE_NAME},
    service::Service,
};
use std::{error::Error, future::Future, sync::Arc, time::Duration};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tonic::transport::Server;
use tonic_health::ServingStatus;

/// Serve a pre-bound listener; tests can bind port zero and inject shutdown.
pub(crate) async fn serve_typed(
    config: Config,
    listener: TcpListener,
    stop: impl Future<Output = ()> + Send,
) -> Result<(), ServerError> {
    config
        .validate()
        .map_err(|e| ServerError::Config(crate::ConfigError(e)))?;
    let factory = Arc::new(
        crate::backend::Factory::prepare(&config)
            .await
            .map_err(ServerError::Prepare)?,
    );
    let shutdown = CancellationToken::new();
    let workers = TaskTracker::new();
    let (health, health_service) = tonic_health::server::health_reporter();
    health.set_service_status("", ServingStatus::Serving).await;
    health
        .set_service_status(SERVICE_NAME, ServingStatus::Serving)
        .await;
    let service = A2fControllerServiceServer::new(
        Service::new(
            config.clone(),
            shutdown.clone(),
            workers.clone(),
            factory.clone(),
        )
        .map_err(ServerError::Prepare)?,
    )
    .max_decoding_message_size(config.max_message_bytes)
    .max_encoding_message_size(config.max_message_bytes);
    let cancel = shutdown.clone();
    let stopping = async move {
        stop.await;
        health
            .set_service_status("", ServingStatus::NotServing)
            .await;
        health
            .set_service_status(SERVICE_NAME, ServingStatus::NotServing)
            .await;
        tracing::info!("stopping");
        cancel.cancel();
    };
    tracing::info!(address = %listener.local_addr()?, backend = ?config.backend, "serving");
    let result = Server::builder()
        .add_service(health_service)
        .add_service(service)
        .serve_with_incoming_shutdown(TcpListenerStream::new(listener), stopping);
    tokio::pin!(result);
    let outcome = tokio::select! {
        outcome = &mut result => outcome.map_err(ServerError::Transport),
        _ = shutdown.cancelled() => {
            tokio::time::timeout(Duration::from_millis(config.shutdown_timeout_ms), &mut result).await
                .map_err(|_| ServerError::ShutdownTimeout)
                .and_then(|r| r.map_err(ServerError::Transport))
        }
    };
    shutdown.cancel();
    workers.close();
    tokio::time::timeout(
        Duration::from_millis(config.shutdown_timeout_ms),
        workers.wait(),
    )
    .await
    .map_err(|_| ServerError::ShutdownTimeout)?;
    factory
        .release_prepared()
        .await
        .map_err(ServerError::Cleanup)?;
    outcome
}

#[derive(Debug)]
pub enum ServerError {
    Config(crate::ConfigError),
    Prepare(tonic::Status),
    Transport(tonic::transport::Error),
    Io(std::io::Error),
    ShutdownTimeout,
    Cleanup(tonic::Status),
}
impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl Error for ServerError {}
impl From<std::io::Error> for ServerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Compatibility helper. New embedding applications should use the Server builder.
pub async fn serve(
    config: Config,
    listener: TcpListener,
    stop: impl Future<Output = ()> + Send,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    serve_typed(config, listener, stop)
        .await
        .map_err(|e| Box::new(e) as _)
}
