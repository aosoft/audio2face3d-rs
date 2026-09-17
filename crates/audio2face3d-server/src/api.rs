//! Embeddable server configuration and lifecycle.
use crate::{
    auth::{Authenticator, NoAuth},
    config::Config,
};
use audio2face3d::{Audio2Face3DContext, logging::integration::LogScope};
use std::{fmt, future::Future, sync::Arc};
#[derive(Debug)]
pub struct ConfigError(pub(crate) String);
impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ConfigError {}
pub struct ServerBuilder<A = NoAuth> {
    config: Config,
    context: Audio2Face3DContext,
    authenticator: Option<Arc<A>>,
}
pub struct Server<A = NoAuth> {
    config: Config,
    context: Audio2Face3DContext,
    authenticator: Option<Arc<A>>,
}
impl Server<NoAuth> {
    pub fn builder(config: Config) -> ServerBuilder<NoAuth> {
        ServerBuilder {
            config,
            context: Audio2Face3DContext::default(),
            authenticator: None,
        }
    }
}
impl<A: Authenticator> Server<A> {
    pub async fn serve(
        self,
        listener: tokio::net::TcpListener,
        stop: impl Future<Output = ()> + Send,
    ) -> Result<(), crate::server::ServerError> {
        LogScope::new(self.context)
            .wrap_future(crate::server::serve_typed(
                self.config,
                self.authenticator,
                listener,
                stop,
            ))
            .await
    }
}
impl<A: Authenticator> ServerBuilder<A> {
    pub fn context(mut self, context: Audio2Face3DContext) -> Self {
        self.context = context;
        self
    }
    pub fn authentication<B: Authenticator>(self, authenticator: Option<B>) -> ServerBuilder<B> {
        ServerBuilder {
            config: self.config,
            context: self.context,
            authenticator: authenticator.map(Arc::new),
        }
    }
    pub fn without_authentication(self) -> ServerBuilder<NoAuth> {
        ServerBuilder {
            config: self.config,
            context: self.context,
            authenticator: None,
        }
    }
    pub fn build(self) -> Result<Server<A>, ConfigError> {
        self.config.validate().map_err(ConfigError)?;
        Ok(Server {
            config: self.config,
            context: self.context,
            authenticator: self.authenticator,
        })
    }
}
