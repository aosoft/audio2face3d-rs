//! Wire audio validation and compatibility re-exports.
use crate::proto::audio::AudioHeader;
pub use audio2face3d::inference::audio::{FPS, FrameBuffer, SAMPLE_RATE};
use tonic::Status;
pub fn validate_header(header: Option<&AudioHeader>) -> Result<(), Status> {
    let header = header.ok_or_else(|| Status::invalid_argument("audio_header is required"))?;
    let format = audio2face3d::protocol::convert::decode_audio_format(*header)
        .map_err(crate::backend::status)?;
    audio2face3d::inference::config::validate_format(format).map_err(crate::backend::status)
}
