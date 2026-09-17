//! Embeddable server configuration and lifecycle.
use crate::config::Config;
use audio2face3d::{Audio2Face3DContext, logging::integration::LogScope};
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
    context: Audio2Face3DContext,
}
pub struct Server {
    config: Config,
    context: Audio2Face3DContext,
}
impl Server {
    pub fn builder(config: Config) -> ServerBuilder {
        ServerBuilder {
            config,
            context: Audio2Face3DContext::default(),
        }
    }
    pub async fn serve(
        self,
        listener: tokio::net::TcpListener,
        stop: impl Future<Output = ()> + Send,
    ) -> Result<(), crate::server::ServerError> {
        LogScope::new(self.context)
            .wrap_future(crate::server::serve_typed(self.config, listener, stop))
            .await
    }
}
impl ServerBuilder {
    pub fn context(mut self, context: Audio2Face3DContext) -> Self {
        self.context = context;
        self
    }

    pub fn build(self) -> Result<Server, ConfigError> {
        self.config.validate().map_err(ConfigError)?;
        Ok(Server {
            config: self.config,
            context: self.context,
        })
    }
}
