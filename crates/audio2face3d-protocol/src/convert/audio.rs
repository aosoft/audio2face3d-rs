use super::{decode_emotion, encode_emotion, protocol, remote, unsupported};
use crate::wire::{a2f, audio};
use audio2face3d_types::{AudioFormat, InputChunk, PcmBuffer, Result, SampleFormat};

pub fn decode_audio_format(value: audio::AudioHeader) -> Result<AudioFormat> {
    if value.audio_format != 0 || value.bits_per_sample != 16 {
        return Err(unsupported("only PCM16LE is supported"));
    }
    let channels =
        u16::try_from(value.channel_count).map_err(|_| protocol("channel count overflow"))?;
    AudioFormat::pcm16(value.samples_per_second, channels).map_err(remote)
}
pub fn encode_audio_format(value: AudioFormat) -> Result<audio::AudioHeader> {
    if value.sample_format() != SampleFormat::Pcm16Le {
        return Err(unsupported("only PCM16LE is supported"));
    }
    Ok(audio::AudioHeader {
        audio_format: 0,
        channel_count: u32::from(value.channels()),
        samples_per_second: value.sample_rate(),
        bits_per_sample: 16,
    })
}
pub fn decode_input(value: a2f::AudioWithEmotion, format: AudioFormat) -> Result<InputChunk> {
    let chunk = InputChunk::new(
        PcmBuffer::from_vec(value.audio_buffer).map_err(remote)?,
        value
            .emotions
            .into_iter()
            .map(decode_emotion)
            .collect::<Result<_>>()?,
    );
    chunk.validate(format).map_err(remote)?;
    Ok(chunk)
}
pub fn encode_input(value: InputChunk, format: AudioFormat) -> Result<a2f::AudioWithEmotion> {
    value.validate(format)?;
    let (pcm, emotions) = value.into_parts();
    Ok(a2f::AudioWithEmotion {
        audio_buffer: pcm.into_vec(),
        emotions: emotions.into_iter().map(encode_emotion).collect(),
    })
}
