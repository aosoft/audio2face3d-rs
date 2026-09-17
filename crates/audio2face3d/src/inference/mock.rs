use crate::inference::{
    Backend, Cancellation, Config, EngineFuture,
    animation::diagnostic_frame,
    audio::{FrameBuffer, SAMPLE_RATE},
};
use crate::types::{Error, InputChunk, OutputBatch};
pub(crate) struct MockBackend {
    buffer: FrameBuffer,
    finished: bool,
    config: Config,
}
impl MockBackend {
    pub fn start(config: &Config) -> Self {
        Self {
            buffer: FrameBuffer::default(),
            finished: false,
            config: config.clone(),
        }
    }
}
impl Backend for MockBackend {
    fn push(&mut self, input: InputChunk) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            self.buffer.push_owned(
                input.into_parts().0.into_vec(),
                u64::from(self.config.max_audio_seconds) * SAMPLE_RATE,
            )
        })
    }
    fn next_frame<'a>(
        &'a mut self,
        _cancel: &'a Cancellation,
    ) -> EngineFuture<'a, Option<OutputBatch>> {
        Box::pin(async move {
            self.buffer
                .pop(self.finished)
                .map(|(start, pcm)| {
                    diagnostic_frame(
                        start,
                        pcm,
                        self.config.mock_pattern,
                        self.config.mock_curve.as_deref(),
                        self.config.mock_value,
                        self.config.mock_jaw_open,
                    )
                })
                .transpose()
        })
    }
    fn finish(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(async move {
            if self.buffer.is_empty() {
                return Err(Error::invalid(
                    "audio clip must contain at least one sample",
                ));
            }
            self.finished = true;
            Ok(())
        })
    }
    fn close(&mut self) -> EngineFuture<'_, ()> {
        self.buffer = FrameBuffer::default();
        Box::pin(std::future::ready(Ok(())))
    }
    fn success_message(&self) -> &'static str {
        "Mock audio processing completed successfully (no inference)."
    }
}
