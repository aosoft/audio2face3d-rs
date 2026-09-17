use crate::{
    BackendKind, Cancellation, Config, config::validate_format, mock::MockBackend,
    resample::Resampler,
};
#[cfg(not(feature = "runtime"))]
use audio2face3d_types::ErrorKind;
use audio2face3d_types::{Error, InputChunk, OutputBatch, PcmBuffer, RequestOptions, Result};
#[cfg(feature = "runtime")]
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
    config: Config,
    #[cfg(feature = "runtime")]
    prepared: Mutex<Option<crate::regression::RegressionBackend>>,
}
impl Factory {
    pub async fn prepare(config: Config) -> Result<Self> {
        config.validate()?;
        #[cfg(not(feature = "runtime"))]
        if config.backend == BackendKind::Regression {
            return Err(Error::new(
                ErrorKind::RuntimeUnavailable,
                "regression requires --features runtime",
            ));
        }
        #[cfg(feature = "runtime")]
        let prepared = if config.backend == BackendKind::Regression {
            Some(
                crate::regression::RegressionBackend::load(
                    config.clone(),
                    RequestOptions::default(),
                )
                .await?,
            )
        } else {
            None
        };
        Ok(Self {
            config,
            #[cfg(feature = "runtime")]
            prepared: Mutex::new(prepared),
        })
    }
    /// Release unused warm state on its owner worker. Active backends remain independently owned.
    pub async fn release_prepared(&self) -> Result<()> {
        #[cfg(feature = "runtime")]
        {
            let prepared = self.prepared.lock().unwrap().take();
            if let Some(mut backend) = prepared {
                backend.close().await?;
            }
        }
        Ok(())
    }
    pub async fn start(&self, options: RequestOptions) -> Result<Box<dyn Backend>> {
        validate_format(options.input_format)?;
        options.validate()?;
        let rate = options.input_format.sample_rate();
        let inner: Box<dyn Backend> = match self.config.backend {
            BackendKind::Mock => Box::new(MockBackend::start(&self.config)),
            BackendKind::Regression => {
                #[cfg(feature = "runtime")]
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
                            crate::regression::RegressionBackend::load(self.config.clone(), options)
                                .await?
                        }
                    })
                }
                #[cfg(not(feature = "runtime"))]
                {
                    return Err(Error::new(
                        ErrorKind::RuntimeUnavailable,
                        "regression requires --features runtime",
                    ));
                }
            }
        };
        Ok(Box::new(ResamplingBackend {
            inner,
            format: audio2face3d_types::AudioFormat::pcm16(rate, 1)?,
            resampler: Resampler::new(rate, self.config.max_audio_seconds),
            finished: false,
            closed: false,
        }))
    }
}
struct ResamplingBackend {
    inner: Box<dyn Backend>,
    format: audio2face3d_types::AudioFormat,
    resampler: Resampler,
    finished: bool,
    closed: bool,
}
impl Backend for ResamplingBackend {
    fn push(&mut self, input: InputChunk) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            if self.closed || self.finished {
                return Err(Error::invalid("audio input is already finished"));
            }
            input.validate(self.format)?;
            let (pcm, emotions) = input.into_parts();
            let pcm = PcmBuffer::from_vec(self.resampler.push(pcm.as_bytes())?)?;
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
