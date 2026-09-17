//! Immutable shared application resources.
use crate::logging::{Logger, NoopLogger};
use std::sync::Arc;
#[derive(Clone)]
pub struct Audio2Face3DContext {
    inner: Arc<ContextInner>,
}
struct ContextInner {
    logger: Arc<dyn Logger>,
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    legacy: bool,
}
pub struct Audio2Face3DContextBuilder {
    logger: Arc<dyn Logger>,
}
impl Audio2Face3DContext {
    pub fn builder() -> Audio2Face3DContextBuilder {
        Audio2Face3DContextBuilder::default()
    }
    pub fn logger(&self) -> &Arc<dyn Logger> {
        &self.inner.logger
    }
    pub(crate) fn legacy() -> Self {
        Self {
            inner: Arc::new(ContextInner {
                logger: Arc::new(NoopLogger),
                legacy: true,
            }),
        }
    }
    pub(crate) fn shares_resources(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
    #[cfg(feature = "tracing")]
    pub(crate) fn is_legacy(&self) -> bool {
        self.inner.legacy
    }
}
impl Default for Audio2Face3DContext {
    fn default() -> Self {
        Self::builder().build()
    }
}
impl Default for Audio2Face3DContextBuilder {
    fn default() -> Self {
        Self {
            logger: Arc::new(NoopLogger),
        }
    }
}
impl Audio2Face3DContextBuilder {
    pub fn logger(mut self, logger: Arc<dyn Logger>) -> Self {
        self.logger = logger;
        self
    }
    pub fn build(self) -> Audio2Face3DContext {
        Audio2Face3DContext {
            inner: Arc::new(ContextInner {
                logger: self.logger,
                legacy: false,
            }),
        }
    }
}
impl std::fmt::Debug for Audio2Face3DContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Audio2Face3DContext")
            .finish_non_exhaustive()
    }
}
