//! Embeddable server configuration and lifecycle.
use crate::config::Config;
use std::{fmt, future::Future};
#[derive(Debug)]
pub struct ConfigError(pub(crate) String);
impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ConfigError {}
pub struct ServerBuilder {
    config: Config,
}
pub struct Server {
    config: Config,
}
impl Server {
    pub fn builder(config: Config) -> ServerBuilder {
        ServerBuilder { config }
    }
    pub async fn serve(
        self,
        listener: tokio::net::TcpListener,
        stop: impl Future<Output = ()> + Send,
    ) -> Result<(), crate::server::ServerError> {
        crate::server::serve_typed(self.config, listener, stop).await
    }
}
impl ServerBuilder {
    pub fn build(self) -> Result<Server, ConfigError> {
        self.config.validate().map_err(ConfigError)?;
        Ok(Server {
            config: self.config,
        })
    }
}
