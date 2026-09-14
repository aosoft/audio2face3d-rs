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
    fn next_frame(&mut self) -> Option<AnimationData> {
        self.buffer
            .pop(self.finished)
            .map(|(start, pcm)| mock_frame(start, pcm, self.config.mock_pattern))
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
    fn cancel(&mut self) {
        self.buffer = FrameBuffer::default();
    }
}
