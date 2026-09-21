use crate::{
    HealthAuth,
    auth::{Authenticator, gate::AuthGate},
    config::Config,
    lifecycle::{self, CleanupCompletion, CleanupStage, Metrics, ShutdownReport},
    proto::{A2fControllerServiceServer, SERVICE_NAME},
    service::Service,
};
use audio2face3d::logging::integration::LogScope;
use std::{error::Error, future::Future, sync::Arc, time::Duration};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tokio_util::{sync::CancellationToken, task::TaskTracker};
use tonic::transport::Server;
use tonic_health::ServingStatus;

pub(crate) async fn serve_typed<A: Authenticator>(
    config: Config,
    authenticator: Option<Arc<A>>,
    health_auth: HealthAuth,
    listener: TcpListener,
    stop: impl Future<Output = ()> + Send,
) -> Result<ShutdownReport, ServerError> {
    config
        .validate()
        .map_err(|e| ServerError::Config(crate::ConfigError(e)))?;
    lifecycle::supervise(stop, move |shutdown, sender| async move {
        run(
            config,
            authenticator,
            health_auth,
            listener,
            shutdown,
            sender,
        )
        .await
    })
    .await
}
async fn run<A: Authenticator>(
    config: Config,
    authenticator: Option<Arc<A>>,
    health_auth: HealthAuth,
    listener: TcpListener,
    shutdown: CancellationToken,
    sender: tokio::sync::oneshot::Sender<lifecycle::Overdue>,
) -> Result<ShutdownReport, ServerError> {
    let started = std::time::Instant::now();
    let result = run_inner(
        config,
        authenticator,
        health_auth,
        listener,
        shutdown,
        sender,
    )
    .await;
    if let Err(error) = &result {
        audio2face3d::logging::integration::log(audio2face3d::logging::LogLevel::Error, || {
            let stage = match error {
                ServerError::Config(_) => "configuration",
                ServerError::Prepare(_) => "prepare",
                ServerError::Transport(_) => "transport",
                ServerError::Io(_) => "io",
                ServerError::Cleanup(_) => "cleanup",
                ServerError::ShutdownTimeout { .. } => "shutdown",
                ServerError::SupervisorStopped => "supervisor",
            };
            audio2face3d::logging::LogRecord::new("server stopped with error")
                .field("source", module_path!())
                .field("stage", stage)
                .field("elapsed_us", started.elapsed().as_micros() as u64)
        });
    }
    result
}
async fn run_inner<A: Authenticator>(
    config: Config,
    authenticator: Option<Arc<A>>,
    health_auth: HealthAuth,
    listener: TcpListener,
    shutdown: CancellationToken,
    sender: tokio::sync::oneshot::Sender<lifecycle::Overdue>,
) -> Result<ShutdownReport, ServerError> {
    let mut sender = Some(sender);
    let timeout = Duration::from_millis(config.shutdown_timeout_ms);
    let prepared = crate::backend::Factory::prepare(&config);
    tokio::pin!(prepared);
    let prepared = tokio::select! {
        biased;
        result = &mut prepared => result,
        _ = shutdown.cancelled() => lifecycle::finish_stage(&mut prepared, timeout, &mut sender, CleanupStage::Prepare, 1).await,
    };
    let factory = Arc::new(prepared.map_err(ServerError::Prepare)?);
    let metrics = Arc::new(Metrics::default());
    let workers = TaskTracker::new();
    let (health, _) = tonic_health::server::health_reporter();
    let gate = Arc::new(AuthGate::new(authenticator));
    let mut outcome = Ok(());
    if !shutdown.is_cancelled() {
        let health_service =
            tonic_health::pb::health_server::HealthServer::new(crate::health::HealthService::new(
                health.clone(),
                gate.clone(),
                health_auth,
                shutdown.clone(),
                workers.clone(),
                metrics.clone(),
            ));
        health.set_service_status("", ServingStatus::Serving).await;
        health
            .set_service_status(SERVICE_NAME, ServingStatus::Serving)
            .await;
        match Service::new(
            config.clone(),
            shutdown.clone(),
            workers.clone(),
            factory.clone(),
            gate,
            metrics.clone(),
        ) {
            Err(error) => outcome = Err(ServerError::Prepare(error)),
            Ok(service) => {
                let service = A2fControllerServiceServer::new(service)
                    .max_decoding_message_size(config.max_message_bytes)
                    .max_encoding_message_size(config.max_message_bytes);
                let cancel = shutdown.clone();
                let stopping = async move {
                    cancel.cancelled().await;
                    health
                        .set_service_status("", ServingStatus::NotServing)
                        .await;
                    health
                        .set_service_status(SERVICE_NAME, ServingStatus::NotServing)
                        .await;
                    audio2face3d::logging::integration::log(
                        audio2face3d::logging::LogLevel::Info,
                        || {
                            audio2face3d::logging::LogRecord::new("stopping")
                                .field("source", module_path!())
                        },
                    );
                };
                audio2face3d::logging::integration::log(
                    audio2face3d::logging::LogLevel::Info,
                    || {
                        audio2face3d::logging::LogRecord::new("serving")
                            .field("source", module_path!())
                            .field("address", format!("{:?}", listener.local_addr()))
                            .field("backend", format!("{:?}", config.backend))
                    },
                );
                // This scope drops a timed-out transport before waiting for workers.
                let transport = Server::builder()
                    .http2_max_header_list_size(16 * 1024)
                    .add_service(health_service)
                    .add_service(service)
                    .serve_with_incoming_shutdown(TcpListenerStream::new(listener), stopping);
                tokio::pin!(transport);
                outcome = tokio::select! {
                    result = &mut transport => result.map_err(ServerError::Transport),
                    _ = shutdown.cancelled() => {
                        match tokio::time::timeout(timeout, &mut transport).await {
                            Ok(result) => result.map_err(ServerError::Transport),
                            Err(_) => { lifecycle::notify(&mut sender, CleanupStage::Transport, 1); Ok(()) }
                        }
                    }
                };
            }
        }
    }
    shutdown.cancel();
    workers.close();
    lifecycle::finish_stage(
        workers.wait(),
        timeout,
        &mut sender,
        CleanupStage::Workers,
        workers.len(),
    )
    .await;
    let cleanup = lifecycle::finish_stage(
        factory.release_prepared(),
        timeout,
        &mut sender,
        CleanupStage::PreparedResources,
        1,
    )
    .await;
    cleanup.map_err(ServerError::Cleanup)?;
    outcome?;
    let report = metrics.snapshot();
    audio2face3d::logging::integration::log(audio2face3d::logging::LogLevel::Info, || {
        audio2face3d::logging::LogRecord::new("server cleanup complete")
            .field("source", module_path!())
            .field("started", report.inference_workers_started)
            .field("finished", report.inference_workers_finished)
    });
    Ok(report)
}

#[derive(Debug)]
pub enum ServerError {
    Config(crate::ConfigError),
    Prepare(tonic::Status),
    Transport(tonic::transport::Error),
    Io(std::io::Error),
    ShutdownTimeout {
        stage: CleanupStage,
        unfinished: usize,
        completion: CleanupCompletion,
    },
    Cleanup(tonic::Status),
    SupervisorStopped,
}
impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShutdownTimeout {
                stage, unfinished, ..
            } => write!(
                f,
                "shutdown timed out during {stage:?} ({unfinished} unfinished)"
            ),
            other => write!(f, "{other:?}"),
        }
    }
}
impl Error for ServerError {}
impl From<std::io::Error> for ServerError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Serve without authentication, preserving the currently selected logging scope.
/// The typed error keeps ownership of a possible cleanup completion handle.
pub async fn serve(
    config: Config,
    listener: TcpListener,
    stop: impl Future<Output = ()> + Send,
) -> Result<ShutdownReport, ServerError> {
    LogScope::capture()
        .wrap_future(serve_typed::<crate::auth::NoAuth>(
            config,
            None,
            HealthAuth::Public,
            listener,
            stop,
        ))
        .await
}
