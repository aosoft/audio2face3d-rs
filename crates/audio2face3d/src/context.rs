//! Immutable shared application resources.
use crate::logging::{Logger, NoopLogger};
use crate::runtime::NativeRuntimeConfig;
use std::sync::Arc;
#[derive(Clone)]
pub struct Audio2Face3DContext {
    inner: Arc<ContextInner>,
}
#[cfg(feature = "cuda")]
#[derive(Default)]
struct NativeResources {
    cuda: std::sync::OnceLock<Arc<crate::cuda::api::CudaApi>>,
    #[cfg(feature = "tensorrt")]
    tensorrt: std::sync::OnceLock<Arc<crate::tensorrt::api::NativeApi>>,
    device_ready: std::sync::atomic::AtomicBool,
}
struct ContextInner {
    #[cfg(feature = "cuda")]
    resources: NativeResources,
    logger: Arc<dyn Logger>,
    native_runtime: NativeRuntimeConfig,
    #[cfg_attr(not(feature = "tracing"), allow(dead_code))]
    legacy: bool,
}
pub struct Audio2Face3DContextBuilder {
    logger: Arc<dyn Logger>,
    native_runtime: NativeRuntimeConfig,
}
impl Audio2Face3DContext {
    pub fn builder() -> Audio2Face3DContextBuilder {
        Audio2Face3DContextBuilder::default()
    }
    pub fn logger(&self) -> &Arc<dyn Logger> {
        &self.inner.logger
    }
    pub fn native_runtime(&self) -> &NativeRuntimeConfig {
        &self.inner.native_runtime
    }
    pub(crate) fn legacy() -> Self {
        Self {
            inner: Arc::new(ContextInner {
                #[cfg(feature = "cuda")]
                resources: NativeResources::default(),
                logger: Arc::new(NoopLogger),
                native_runtime: NativeRuntimeConfig::default(),
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
            native_runtime: NativeRuntimeConfig::default(),
        }
    }
}
impl Audio2Face3DContextBuilder {
    pub fn native_runtime(mut self, config: NativeRuntimeConfig) -> Self {
        self.native_runtime = config;
        self
    }
    pub fn logger(mut self, logger: Arc<dyn Logger>) -> Self {
        self.logger = logger;
        self
    }
    pub fn build(self) -> Audio2Face3DContext {
        Audio2Face3DContext {
            inner: Arc::new(ContextInner {
                #[cfg(feature = "cuda")]
                resources: NativeResources::default(),
                logger: self.logger,
                native_runtime: self.native_runtime,
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

impl Audio2Face3DContext {
    /// Returns observations already made by this Context, without loading libraries.
    pub fn native_runtime_info(&self) -> Option<crate::runtime::NativeRuntimeInfo> {
        #[cfg(feature = "cuda")]
        {
            #[cfg(feature = "tensorrt")]
            let info = self
                .inner
                .resources
                .tensorrt
                .get()
                .map(|api| api.info.clone())
                .or_else(|| self.inner.resources.cuda.get().map(|api| api.info.clone()));
            #[cfg(not(feature = "tensorrt"))]
            let info = self.inner.resources.cuda.get().map(|api| api.info.clone());
            info.map(|mut info| {
                if self
                    .inner
                    .resources
                    .device_ready
                    .load(std::sync::atomic::Ordering::Acquire)
                {
                    info.state = crate::runtime::NativeRuntimeState::DeviceReady;
                }
                info
            })
        }
        #[cfg(not(feature = "cuda"))]
        None
    }
    /// Initializes enabled native components synchronously. No async runtime is required.
    /// Call from a caller-owned worker when blocking native initialization is undesirable.
    #[cfg(feature = "cuda")]
    pub fn initialize_native(
        &self,
    ) -> Result<crate::runtime::NativeRuntimeInfo, crate::runtime::NativeRuntimeError> {
        #[cfg(feature = "tensorrt")]
        crate::tensorrt::api::NativeApi::initialize(self)?;
        #[cfg(not(feature = "tensorrt"))]
        crate::cuda::api::CudaApi::initialize(self)?;
        Ok(self
            .native_runtime_info()
            .expect("initialized native resources"))
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn cached_cuda(&self) -> Option<Arc<crate::cuda::api::CudaApi>> {
        self.inner.resources.cuda.get().cloned()
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn retain_cuda(&self, api: Arc<crate::cuda::api::CudaApi>) {
        let _ = self.inner.resources.cuda.set(api);
    }
    #[cfg(feature = "tensorrt")]
    pub(crate) fn cached_tensorrt(&self) -> Option<Arc<crate::tensorrt::api::NativeApi>> {
        self.inner.resources.tensorrt.get().cloned()
    }
    #[cfg(feature = "tensorrt")]
    pub(crate) fn retain_tensorrt(&self, api: Arc<crate::tensorrt::api::NativeApi>) {
        let _ = self.inner.resources.tensorrt.set(api);
    }
    #[cfg(feature = "cuda")]
    pub(crate) fn native_device_ready(&self) {
        self.inner
            .resources
            .device_ready
            .store(true, std::sync::atomic::Ordering::Release);
    }
}
