pub mod mock;
use crate::proto::{a2f::AudioWithEmotion, animation::AnimationData};
use tonic::Status;

/// Per-RPC owned state. The runtime backend will supply owned output here.
pub trait Backend: Send {
    fn push(&mut self, input: AudioWithEmotion) -> Result<(), Status>;
    fn next_frame(&mut self) -> Option<AnimationData>;
    fn finish(&mut self) -> Result<(), Status>;
    fn cancel(&mut self);
}
