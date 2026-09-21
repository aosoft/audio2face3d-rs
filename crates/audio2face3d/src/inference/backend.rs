#[cfg(feature = "mock")]
use super::mock::MockBackend;
use crate::inference::{BackendKind, Cancellation, Config};
#[cfg(not(feature = "native"))]
use crate::types::ErrorKind;
use crate::types::{Error, InputChunk, OutputBatch, RequestOptions, Result};
use crate::{Audio2Face3DContext, logging::integration::LogScope};
#[cfg(any(feature = "mock", feature = "native"))]
use crate::{
    inference::{config::validate_format, resample::Resampler},
    types::PcmBuffer,
};
#[cfg(feature = "native")]
use std::sync::Mutex;
use std::{future::Future, pin::Pin};

pub type EngineFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;
/// A single utterance. Await each operation before submitting the next operation.
/// `next_frame` drains currently available output; None before finish means more input is needed.
/// Always await close before releasing execution admission, even after errors/cancellation.
pub trait Backend: Send {
    fn push(&mut self, input: InputChunk) -> EngineFuture<'_, ()>;
    fn next_frame<'a>(
        &'a mut self,
        cancel: &'a Cancellation,
    ) -> EngineFuture<'a, Option<OutputBatch>>;
    fn finish(&mut self) -> EngineFuture<'_, ()>;
    fn close(&mut self) -> EngineFuture<'_, ()>;
    fn success_message(&self) -> &'static str;
}
/// Performs one warm load; each subsequent utterance owns a fresh native runtime.
pub struct Factory {
    scope: LogScope,
    #[cfg_attr(not(any(feature = "mock", feature = "native")), allow(dead_code))]
    config: Config,
    #[cfg(feature = "native")]
    prepared: Mutex<Option<crate::inference::regression::RegressionBackend>>,
}
impl Factory {
    async fn prepare_inner(config: Config, scope: LogScope) -> Result<Self> {
        tracing::info!("preparing inference");
        config.validate()?;
        #[cfg(not(feature = "native"))]
        if config.backend == BackendKind::Regression {
            return Err(Error::new(
                ErrorKind::RuntimeUnavailable,
                "regression requires --features native",
            ));
        }
        #[cfg(feature = "native")]
        let prepared = if config.backend == BackendKind::Regression {
            Some(
                crate::inference::regression::RegressionBackend::load(
                    config.clone(),
                    RequestOptions::default(),
                )
                .await?,
            )
        } else {
            None
        };
        tracing::info!("inference prepared");
        Ok(Self {
            scope,
            config,
            #[cfg(feature = "native")]
            prepared: Mutex::new(prepared),
        })
    }
    /// Release unused warm state on its owner worker. Active backends remain independently owned.
    async fn release_prepared_inner(&self) -> Result<()> {
        use crate::logging::Logger;
        self.scope
            .context()
            .logger()
            .log(crate::logging::LogLevel::Debug, || {
                "audio2face3d::inference release prepared inference resources".into()
            });
        #[cfg(feature = "native")]
        {
            let prepared = self.prepared.lock().unwrap().take();
            if let Some(mut backend) = prepared {
                backend.close().await?;
            }
        }
        Ok(())
    }
    #[cfg(any(feature = "mock", feature = "native"))]
    async fn start_inner(&self, options: RequestOptions) -> Result<Box<dyn Backend>> {
        validate_format(options.input_format)?;
        options.validate()?;
        let rate = options.input_format.sample_rate();
        let inner: Box<dyn Backend> = match self.config.backend {
            BackendKind::Mock => {
                #[cfg(feature = "mock")]
                {
                    Box::new(MockBackend::start(&self.config))
                }
                #[cfg(not(feature = "mock"))]
                {
                    return Err(Error::new(
                        crate::types::ErrorKind::RuntimeUnavailable,
                        "mock backend is not compiled",
                    ));
                }
            }
            BackendKind::Regression => {
                #[cfg(feature = "native")]
                {
                    let custom = options.face.is_some()
                        || options.blendshapes.is_some()
                        || options.emotion.is_some()
                        || options.emotion_post_processing.is_some();
                    let mut prepared = self.prepared.lock().unwrap().take();
                    if custom && let Some(mut backend) = prepared.take() {
                        backend.close().await?;
                    }
                    Box::new(match prepared {
                        Some(backend) => backend,
                        None => {
                            crate::inference::regression::RegressionBackend::load(
                                self.config.clone(),
                                options,
                            )
                            .await?
                        }
                    })
                }
                #[cfg(not(feature = "native"))]
                {
                    return Err(Error::new(
                        ErrorKind::RuntimeUnavailable,
                        "regression requires --features native",
                    ));
                }
            }
        };
        tracing::debug!("starting inference request");
        Ok(Box::new(ScopedBackend {
            scope: LogScope::capture(),
            inner: Box::new(ResamplingBackend {
                inner,
                format: crate::types::AudioFormat::pcm16(rate, 1)?,
                resampler: Resampler::new(rate, self.config.max_audio_seconds),
                finished: false,
                closed: false,
            }),
        }))
    }
    #[cfg(not(any(feature = "mock", feature = "native")))]
    async fn start_inner(&self, _options: RequestOptions) -> Result<Box<dyn Backend>> {
        Err(Error::new(
            ErrorKind::RuntimeUnavailable,
            "no inference backend is compiled",
        ))
    }
}
#[cfg(any(feature = "mock", feature = "native"))]
struct ResamplingBackend {
    inner: Box<dyn Backend>,
    format: crate::types::AudioFormat,
    resampler: Resampler,
    finished: bool,
    closed: bool,
}
#[cfg(any(feature = "mock", feature = "native"))]
impl Backend for ResamplingBackend {
    fn push(&mut self, input: InputChunk) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            if self.closed || self.finished {
                return Err(Error::invalid("audio input is already finished"));
            }
            input.validate(self.format)?;
            let (pcm, emotions) = input.into_parts();
            let pcm = PcmBuffer::from_vec(self.resampler.push_owned(pcm.into_vec())?)?;
            self.inner.push(InputChunk::new(pcm, emotions)).await
        })
    }
    fn next_frame<'a>(
        &'a mut self,
        cancel: &'a Cancellation,
    ) -> EngineFuture<'a, Option<OutputBatch>> {
        self.inner.next_frame(cancel)
    }
    fn finish(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            if self.finished || self.closed {
                return Err(Error::invalid("audio input is already finished"));
            }
            self.finished = true;
            let tail = self.resampler.finish();
            if !tail.is_empty() {
                self.inner
                    .push(InputChunk::new(PcmBuffer::from_vec(tail)?, vec![]))
                    .await?;
            }
            self.inner.finish().await
        })
    }
    fn close(&mut self) -> EngineFuture<'_, ()> {
        self.closed = true;
        self.inner.close()
    }
    fn success_message(&self) -> &'static str {
        self.inner.success_message()
    }
}

impl Factory {
    pub async fn prepare(config: Config) -> Result<Self> {
        let scope = LogScope::capture();
        scope
            .wrap_future(Self::prepare_inner(config, scope.clone()))
            .await
    }
    pub async fn prepare_with_context(
        config: Config,
        context: Audio2Face3DContext,
    ) -> Result<Self> {
        let scope = LogScope::new(context);
        scope
            .wrap_future(Self::prepare_inner(config, scope.clone()))
            .await
    }
    pub async fn release_prepared(&self) -> Result<()> {
        self.scope.wrap_future(self.release_prepared_inner()).await
    }
    pub async fn start(&self, options: RequestOptions) -> Result<Box<dyn Backend>> {
        let scope = LogScope::capture();
        let scope = if scope.context().shares_resources(self.scope.context()) {
            scope
        } else {
            self.scope.clone()
        };
        scope.wrap_future(self.start_inner(options)).await
    }
}

#[cfg(any(feature = "mock", feature = "native"))]
struct ScopedBackend {
    inner: Box<dyn Backend>,
    scope: LogScope,
}
#[cfg(any(feature = "mock", feature = "native"))]
impl Backend for ScopedBackend {
    fn push(&mut self, input: InputChunk) -> EngineFuture<'_, ()> {
        let scope = self.scope.clone();
        Box::pin(scope.wrap_future(self.inner.push(input)))
    }
    fn next_frame<'a>(
        &'a mut self,
        cancel: &'a Cancellation,
    ) -> EngineFuture<'a, Option<OutputBatch>> {
        let scope = self.scope.clone();
        Box::pin(scope.wrap_future(self.inner.next_frame(cancel)))
    }
    fn finish(&mut self) -> EngineFuture<'_, ()> {
        let scope = self.scope.clone();
        Box::pin(scope.wrap_future(self.inner.finish()))
    }
    fn close(&mut self) -> EngineFuture<'_, ()> {
        let scope = self.scope.clone();
        Box::pin(scope.wrap_future(async move {
            tracing::debug!("closing inference request");
            self.inner.close().await
        }))
    }
    fn success_message(&self) -> &'static str {
        self.inner.success_message()
    }
}
