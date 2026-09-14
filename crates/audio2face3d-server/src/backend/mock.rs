use super::Backend;
use crate::{
    animation::mock_frame,
    audio::{FrameBuffer, SAMPLE_RATE},
    config::Config,
    proto::{a2f::AudioWithEmotion, animation::AnimationData, controller::AudioStreamHeader},
};
use tonic::Status;

pub struct MockBackend {
    buffer: FrameBuffer,
    finished: bool,
    config: Config,
}
impl MockBackend {
    pub fn start(config: &Config, header: &AudioStreamHeader) -> Self {
        if header.face_params.is_some()
            || header.blendshape_params.is_some()
            || header.emotion_params.is_some()
            || header.emotion_post_processing_params.is_some()
        {
            tracing::warn!(
                "mock ignores face, blendshape and emotion settings; output is diagnostic only"
            );
        }
        Self {
            buffer: FrameBuffer::default(),
            finished: false,
            config: config.clone(),
        }
    }
}
#[tonic::async_trait]
impl Backend for MockBackend {
    fn push(&mut self, input: AudioWithEmotion) -> Result<(), Status> {
        if !input.emotions.is_empty() {
            tracing::debug!("mock ignores input emotion keyframes");
        }
        self.buffer.push(
            &input.audio_buffer,
            u64::from(self.config.max_audio_seconds) * SAMPLE_RATE,
        )
    }
    async fn next_frame(
        &mut self,
        _cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<AnimationData>, Status> {
        Ok(self
            .buffer
            .pop(self.finished)
            .map(|(start, pcm)| mock_frame(start, pcm, self.config.mock_pattern)))
    }
    fn finish(&mut self) -> Result<(), Status> {
        if self.buffer.is_empty() {
            return Err(Status::invalid_argument(
                "audio clip must contain at least one sample",
            ));
        }
        self.finished = true;
        Ok(())
    }
    async fn close(&mut self) -> Result<(), Status> {
        self.buffer = FrameBuffer::default();
        Ok(())
    }
    fn success_message(&self) -> &'static str {
        "Mock audio processing completed successfully (no inference)."
    }
}
