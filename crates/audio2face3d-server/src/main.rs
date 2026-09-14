use audio2face3d_server::{config::Config, proto::DESCRIPTOR, server};
use clap::Parser;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let config = Config::parse();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_ansi(false)
        .init();
    if let Some(path) = &config.export_descriptor {
        std::fs::write(path, DESCRIPTOR)?;
        return Ok(());
    }
    config.validate()?;
    let listener = tokio::net::TcpListener::bind(config.listen).await?;
    server::serve(config, listener, async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            tracing::error!(%error, "signal handler failed");
        }
    })
    .await
}
