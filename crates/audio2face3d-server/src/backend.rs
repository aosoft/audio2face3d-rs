//! CLI/wire boundary around the runtime-independent inference crate.
use crate::{
    config::{BackendKind, Config},
    proto::{a2f::AudioWithEmotion, animation::AnimationData, controller::AudioStreamHeader},
};
use audio2face3d_inference as inference;
use audio2face3d_protocol::convert;
use audio2face3d_types::{AudioFormat, Error, ErrorKind, RequestOptions};
use tokio_util::sync::CancellationToken;
use tonic::Status;

pub(crate) fn status(error: Error) -> Status {
    match error.kind() {
        ErrorKind::InvalidInput | ErrorKind::Unsupported | ErrorKind::Protocol => {
            Status::invalid_argument(error.message())
        }
        ErrorKind::QueueFull | ErrorKind::LimitExceeded => {
            Status::resource_exhausted(error.message())
        }
        ErrorKind::DeadlineExceeded => Status::deadline_exceeded(error.message()),
        ErrorKind::Cancelled => Status::cancelled(error.message()),
        ErrorKind::ShuttingDown => Status::unavailable(error.message()),
        ErrorKind::RuntimeUnavailable => Status::failed_precondition(error.message()),
        _ => Status::internal(error.message()),
    }
}
fn engine_config(config: &Config) -> inference::Config {
    inference::Config {
        backend: match config.backend {
            BackendKind::Mock => inference::BackendKind::Mock,
            BackendKind::Regression => inference::BackendKind::Regression,
        },
        model: config.model.clone(),
        emotion_model: config.emotion_model.clone(),
        device: config.device,
        max_audio_seconds: config.max_audio_seconds,
        mock_pattern: config.mock_pattern.into(),
        mock_curve: config.mock_curve.clone(),
        mock_value: config.mock_value,
        mock_jaw_open: config.mock_jaw_open,
    }
}
pub struct Factory {
    inner: inference::Factory,
}
impl Factory {
    pub async fn prepare(config: &Config) -> Result<Self, Status> {
        config.validate().map_err(Status::invalid_argument)?;
        Ok(Self {
            inner: inference::Factory::prepare(engine_config(config))
                .await
                .map_err(status)?,
        })
    }
    pub async fn release_prepared(&self) -> Result<(), Status> {
        self.inner.release_prepared().await.map_err(status)
    }
    pub async fn start(
        &self,
        config: &Config,
        header: &AudioStreamHeader,
    ) -> Result<Backend, Status> {
        let options = if config.backend == BackendKind::Mock {
            if header.face_params.is_some()
                || header.blendshape_params.is_some()
                || header.emotion_params.is_some()
                || header.emotion_post_processing_params.is_some()
            {
                tracing::warn!(
                    "mock ignores face, blendshape and emotion settings; output is diagnostic only"
                );
            }
            RequestOptions::new(
                convert::decode_audio_format(
                    header
                        .audio_header
                        .ok_or_else(|| Status::invalid_argument("audio_header is required"))?,
                )
                .map_err(status)?,
            )
        } else {
            convert::decode_request(header.clone()).map_err(status)?
        };
        let format = options.input_format;
        let inner = self.inner.start(options).await.map_err(status)?;
        Ok(Backend {
            inner,
            format,
            kind: config.backend,
            max_seconds: f64::from(config.max_audio_seconds),
        })
    }
}
pub struct Backend {
    inner: Box<dyn inference::Backend>,
    format: AudioFormat,
    kind: BackendKind,
    max_seconds: f64,
}
impl Backend {
    pub async fn push(&mut self, input: AudioWithEmotion) -> Result<(), Status> {
        let input = prepare_input(input, self.format, self.kind, self.max_seconds)?;
        self.inner.push(input).await.map_err(status)
    }
    pub async fn next_frame(
        &mut self,
        shutdown: &CancellationToken,
    ) -> Result<Option<AnimationData>, Status> {
        let cancel = inference::Cancellation::new();
        let pending = self.inner.next_frame(&cancel);
        tokio::pin!(pending);
        let output = tokio::select! {
            biased;
            _ = shutdown.cancelled() => { cancel.cancel(); pending.await },
            result = &mut pending => result,
        }
        .map_err(status)?;
        output
            .map(convert::encode_animation)
            .transpose()
            .map_err(status)
    }
    pub async fn finish(&mut self) -> Result<(), Status> {
        self.inner.finish().await.map_err(status)
    }
    pub async fn close(&mut self) -> Result<(), Status> {
        self.inner.close().await.map_err(status)
    }
    pub fn success_message(&self) -> &'static str {
        self.inner.success_message()
    }
}

fn prepare_input(
    mut input: AudioWithEmotion,
    format: AudioFormat,
    kind: BackendKind,
    max_seconds: f64,
) -> Result<audio2face3d_types::InputChunk, Status> {
    if kind == BackendKind::Mock {
        if !input.emotions.is_empty() {
            tracing::debug!("mock ignores input emotion keyframes");
            input.emotions.clear();
        }
    } else {
        for key in &mut input.emotions {
            if !key.time_code.is_finite() || key.time_code < 0.0 || key.time_code > max_seconds {
                return Err(Status::invalid_argument(
                    "emotion time_code outside clip limit",
                ));
            }
            // Preserve the original ACE -> native sample rounding before conversion
            // to the shared nanosecond representation (including sub-ns near ties).
            key.time_code = (key.time_code * 16000.0).round() / 16000.0;
        }
    }
    convert::decode_input(input, format).map_err(status)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::nvidia_ace::emotion_with_timecode::v1::EmotionWithTimeCode;
    #[test]
    fn wire_emotion_sample_rounding_precedes_nanosecond_conversion() {
        for seconds in [
            0.0,
            0.0000312499,
            0.00003125,
            0.0000312501,
            0.0333437499,
            0.03334375,
            599.9999999999,
            600.0,
        ] {
            let input = AudioWithEmotion {
                audio_buffer: vec![],
                emotions: vec![EmotionWithTimeCode {
                    time_code: seconds,
                    emotion: Default::default(),
                }],
            };
            let prepared = prepare_input(
                input,
                AudioFormat::MONO_16KHZ,
                BackendKind::Regression,
                600.0,
            )
            .unwrap();
            assert_eq!(
                prepared.emotions()[0]
                    .time()
                    .nearest_sample(16000)
                    .unwrap()
                    .0,
                (seconds * 16000.0).round() as u64
            );
        }
        for seconds in [-1e-12, f64::NAN, f64::INFINITY, 600.0000000001] {
            let input = AudioWithEmotion {
                audio_buffer: vec![],
                emotions: vec![EmotionWithTimeCode {
                    time_code: seconds,
                    emotion: Default::default(),
                }],
            };
            assert_eq!(
                prepare_input(
                    input,
                    AudioFormat::MONO_16KHZ,
                    BackendKind::Regression,
                    600.0
                )
                .unwrap_err()
                .code(),
                tonic::Code::InvalidArgument
            );
        }
    }
    #[tokio::test]
    async fn mock_keeps_ignoring_unsupported_settings_and_emotion_keys() {
        use clap::Parser;
        let config = Config::parse_from(["test"]);
        let factory = Factory::prepare(&config).await.unwrap();
        let mut header = AudioStreamHeader {
            audio_header: Some(crate::proto::audio::AudioHeader {
                audio_format: 0,
                channel_count: 1,
                samples_per_second: 16000,
                bits_per_sample: 16,
            }),
            face_params: Some(Default::default()),
            ..Default::default()
        };
        header
            .face_params
            .as_mut()
            .unwrap()
            .float_params
            .insert("unknown".into(), f32::NAN);
        let mut engine = factory.start(&config, &header).await.unwrap();
        let input = AudioWithEmotion {
            audio_buffer: vec![1, 2],
            emotions: vec![EmotionWithTimeCode {
                time_code: f64::NAN,
                emotion: Default::default(),
            }],
        };
        engine.push(input).await.unwrap();
        engine.finish().await.unwrap();
        assert_eq!(
            engine
                .next_frame(&CancellationToken::new())
                .await
                .unwrap()
                .unwrap()
                .audio
                .unwrap()
                .audio_buffer,
            vec![1, 2]
        );
        engine.close().await.unwrap();
    }
}
