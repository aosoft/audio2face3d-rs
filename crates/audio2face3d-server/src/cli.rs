mod args;
use audio2face3d_server::{Server, proto::DESCRIPTOR};
use clap::Parser;
use std::error::Error;

#[tokio::main]
pub async fn run() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = args::Args::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_ansi(false)
        .init();
    if let Some(path) = &args.export_descriptor {
        std::fs::write(path, DESCRIPTOR)?;
        return Ok(());
    }
    let config = args.config();
    config.validate()?;
    let listener = tokio::net::TcpListener::bind(args.listen).await?;
    Server::builder(config)
        .build()?
        .serve(listener, async {
            if let Err(error) = tokio::signal::ctrl_c().await {
                tracing::error!(%error, "signal handler failed");
            }
        })
        .await
        .map_err(|e| Box::new(e) as _)
}
