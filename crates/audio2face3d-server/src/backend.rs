#[cfg(feature = "runtime")]
pub mod audio2face;
#[cfg(feature = "runtime")]
mod emotion;
pub mod mock;
#[cfg(feature = "runtime")]
mod parameters;
pub(crate) mod resample;

use crate::{
    config::{BackendKind, Config},
    proto::{a2f::AudioWithEmotion, animation::AnimationData, controller::AudioStreamHeader},
};
use tonic::Status;

/// Each RPC owns its state; native calls are awaited through completion.
#[tonic::async_trait]
pub trait Backend: Send {
    fn push(&mut self, input: AudioWithEmotion) -> Result<(), Status>;
    async fn next_frame(
        &mut self,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<AnimationData>, Status>;
    fn finish(&mut self) -> Result<(), Status>;
    async fn close(&mut self) -> Result<(), Status>;
    fn success_message(&self) -> &'static str;
}

pub struct Factory {
    #[cfg(feature = "runtime")]
    prepared: tokio::sync::Mutex<Option<audio2face::Runtime>>,
}

impl Factory {
    pub async fn prepare(config: &Config) -> Result<Self, Status> {
        config.validate().map_err(Status::invalid_argument)?;
        #[cfg(feature = "runtime")]
        {
            let prepared = if config.backend == BackendKind::Regression {
                Some(audio2face::load(config.clone()).await?)
            } else {
                None
            };
            Ok(Self {
                prepared: tokio::sync::Mutex::new(prepared),
            })
        }
        #[cfg(not(feature = "runtime"))]
        Ok(Self {})
    }

    pub async fn start(
        &self,
        config: &Config,
        header: &AudioStreamHeader,
    ) -> Result<Box<dyn Backend>, Status> {
        match config.backend {
            BackendKind::Mock => Ok(Box::new(mock::MockBackend::start(config, header))),
            BackendKind::Regression => {
                #[cfg(feature = "runtime")]
                {
                    let custom = header.face_params.is_some()
                        || header.blendshape_params.is_some()
                        || header.emotion_params.is_some()
                        || header.emotion_post_processing_params.is_some();
                    let mut prepared = self.prepared.lock().await.take();
                    if custom && let Some(runtime) = prepared.take() {
                        tokio::task::spawn_blocking(move || drop(runtime))
                            .await
                            .map_err(|e| Status::internal(e.to_string()))?;
                    }
                    let runtime = match prepared {
                        Some(runtime) => runtime,
                        None => {
                            audio2face::load_with_header(config.clone(), header.clone()).await?
                        }
                    };
                    Ok(Box::new(audio2face::RegressionBackend::new(
                        runtime, config,
                    )))
                }
                #[cfg(not(feature = "runtime"))]
                Err(Status::failed_precondition(
                    "regression requires --features runtime",
                ))
            }
        }
    }
}
