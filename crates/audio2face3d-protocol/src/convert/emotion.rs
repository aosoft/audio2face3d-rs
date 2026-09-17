use super::{protocol, remote};
use crate::wire::nvidia_ace::{
    emotion_aggregate::v1::EmotionAggregate, emotion_with_timecode::v1::EmotionWithTimeCode,
};
use audio2face3d_types::{EmotionKeyframe, EmotionTrace, MediaTime, Result};
use prost::Message;

const EMOTION_TYPE: &str = "nvidia_ace.emotion_aggregate.v1.EmotionAggregate";

pub fn decode_emotion(value: EmotionWithTimeCode) -> Result<EmotionKeyframe> {
    EmotionKeyframe::new(
        MediaTime::from_seconds(value.time_code).map_err(remote)?,
        value.emotion.into_iter().collect(),
    )
    .map_err(remote)
}
pub fn encode_emotion(value: EmotionKeyframe) -> EmotionWithTimeCode {
    let (time, values) = value.into_parts();
    EmotionWithTimeCode {
        time_code: time.as_seconds(),
        emotion: values.into_iter().collect(),
    }
}
pub fn decode_emotion_trace(value: prost_types::Any) -> Result<EmotionTrace> {
    if value.type_url.rsplit('/').next() != Some(EMOTION_TYPE) || !value.type_url.contains('/') {
        return Err(protocol("emotion aggregate has an unexpected type URL"));
    }
    let value = EmotionAggregate::decode(value.value.as_slice())
        .map_err(|e| protocol(format!("invalid emotion aggregate: {e}")))?;
    Ok(EmotionTrace {
        input: value
            .input_emotions
            .into_iter()
            .map(decode_emotion)
            .collect::<Result<_>>()?,
        mixed: value
            .a2e_output
            .into_iter()
            .map(decode_emotion)
            .collect::<Result<_>>()?,
        smoothed: value
            .a2f_smoothed_output
            .into_iter()
            .map(decode_emotion)
            .collect::<Result<_>>()?,
    })
}
pub fn encode_emotion_trace(value: EmotionTrace) -> prost_types::Any {
    let value = EmotionAggregate {
        input_emotions: value.input.into_iter().map(encode_emotion).collect(),
        a2e_output: value.mixed.into_iter().map(encode_emotion).collect(),
        a2f_smoothed_output: value.smoothed.into_iter().map(encode_emotion).collect(),
    };
    prost_types::Any {
        type_url: format!("type.googleapis.com/{EMOTION_TYPE}"),
        value: value.encode_to_vec(),
    }
}
