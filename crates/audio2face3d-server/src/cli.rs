mod args;
mod auth;
use audio2face3d::cli_logging as logging;
use audio2face3d_server::{Server, ServerError, proto::DESCRIPTOR};
use clap::{Parser, error::ErrorKind};
use std::error::Error;

#[tokio::main]
pub async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut args = match args::Args::try_parse() {
        Ok(args) => args,
        Err(error)
            if matches!(
                error.kind(),
                ErrorKind::DisplayHelp | ErrorKind::DisplayVersion
            ) =>
        {
            error.print()?;
            return Ok(());
        }
        // Clap can echo argument values. Never print a parse error carrying secrets.
        Err(_) => return Err("invalid command line arguments; use --help".into()),
    };
    if let Some(path) = &args.export_descriptor {
        std::fs::write(path, DESCRIPTOR)?;
        return Ok(());
    }
    let authenticator = auth::resolve(args.api_key.take())?;
    let logging = logging::Logging::start(&args.logging, env!("CARGO_PKG_NAME"))?;
    let result = async {
    let context = audio2face3d::Audio2Face3DContext::builder()
        .logger(logging.logger.clone())
        .native_runtime(args.runtime.resolve()?)
        .build();
    let server = Server::builder(args.config()?)
        .context(context)
        .authentication(authenticator)
        .health_auth(args.health_auth)
        .build()?;
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    let outcome = server
        .serve(listener, async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error, "signal handler failed");
            }
        })
        .await;
    match outcome {
        Ok(_) => Ok(()),
        Err(ServerError::ShutdownTimeout {
            stage,
            unfinished,
            completion,
        }) => {
            eprintln!(
                "shutdown deadline exceeded during {stage:?} ({unfinished} unfinished); awaiting cleanup"
            );
            completion.await?;
            // The synchronous stderr writer is done before the runtime is destroyed.
            Err("shutdown exceeded its stage deadline; cleanup completed".into())
        }
        Err(error) => Err(Box::new(error) as Box<dyn Error + Send + Sync>),
    }
    }.await;
    logging.finish()?;
    result
}
