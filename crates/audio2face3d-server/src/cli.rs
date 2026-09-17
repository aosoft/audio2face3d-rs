mod args;
mod logging;
use audio2face3d_server::{Server, proto::DESCRIPTOR};
use clap::Parser;
use std::error::Error;

#[tokio::main]
pub async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = args::Args::parse();
    if let Some(path) = &args.export_descriptor {
        std::fs::write(path, DESCRIPTOR)?;
        return Ok(());
    }
    let logger = logging::StderrLogger::from_env()?;
    let context = audio2face3d::Audio2Face3DContext::builder()
        .logger(std::sync::Arc::new(logger))
        .build();
    let config = args.config();
    config.validate()?;
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    Server::builder(config)
        .context(context)
        .build()?
        .serve(listener, async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error, "signal handler failed");
            }
        })
        .await
        .map_err(|e| Box::new(e) as _)
}
