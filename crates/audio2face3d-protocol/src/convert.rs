//! Consuming conversions move audio and curve payloads; borrowed views never
//! outlive a native callback. Packet conversion does not determine stream success:
//! the stream driver must validate ordering, final status and transport trailers.
mod animation;
mod audio;
mod emotion;
mod request;

pub use animation::{decode_animation, decode_stream_info, encode_animation, encode_stream_info};
pub use audio::{decode_audio_format, decode_input, encode_audio_format, encode_input};
pub use emotion::{decode_emotion, decode_emotion_trace, encode_emotion, encode_emotion_trace};
pub use request::{EncodedRequest, decode_request, encode_request};

use audio2face3d_types::{Error, ErrorKind};
fn protocol(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Protocol, message)
}
fn unsupported(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::Unsupported, message)
}
fn remote(error: Error) -> Error {
    if error.kind() == ErrorKind::InvalidInput {
        protocol(error.to_string())
    } else {
        error
    }
}
