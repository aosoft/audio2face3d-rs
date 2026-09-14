use crate::{
    audio::SAMPLE_RATE,
    config::MockPattern,
    proto::{animation, audio, controller},
};

// UE ACE 2.5 CurveNames[0..52], in the same order as Mark's non-neutral skin poses.
pub const CURVE_NAMES: [&str; 52] = [
    "EyeBlinkLeft",
    "EyeLookDownLeft",
    "EyeLookInLeft",
    "EyeLookOutLeft",
    "EyeLookUpLeft",
    "EyeSquintLeft",
    "EyeWideLeft",
    "EyeBlinkRight",
    "EyeLookDownRight",
    "EyeLookInRight",
    "EyeLookOutRight",
    "EyeLookUpRight",
    "EyeSquintRight",
    "EyeWideRight",
    "JawForward",
    "JawLeft",
    "JawRight",
    "JawOpen",
    "MouthClose",
    "MouthFunnel",
    "MouthPucker",
    "MouthLeft",
    "MouthRight",
    "MouthSmileLeft",
    "MouthSmileRight",
    "MouthFrownLeft",
    "MouthFrownRight",
    "MouthDimpleLeft",
    "MouthDimpleRight",
    "MouthStretchLeft",
    "MouthStretchRight",
    "MouthRollLower",
    "MouthRollUpper",
    "MouthShrugLower",
    "MouthShrugUpper",
    "MouthPressLeft",
    "MouthPressRight",
    "MouthLowerDownLeft",
    "MouthLowerDownRight",
    "MouthUpperUpLeft",
    "MouthUpperUpRight",
    "BrowDownLeft",
    "BrowDownRight",
    "BrowInnerUp",
    "BrowOuterUpLeft",
    "BrowOuterUpRight",
    "CheekPuff",
    "CheekSquintLeft",
    "CheekSquintRight",
    "NoseSneerLeft",
    "NoseSneerRight",
    "TongueOut",
];

pub fn header(epoch_seconds: f64) -> controller::AnimationDataStream {
    controller::AnimationDataStream {
        stream_part: Some(
            controller::animation_data_stream::StreamPart::AnimationDataStreamHeader(
                controller::AnimationDataStreamHeader {
                    audio_header: Some(audio::AudioHeader {
                        audio_format: 0,
                        channel_count: 1,
                        samples_per_second: 16_000,
                        bits_per_sample: 16,
                    }),
                    skel_animation_header: Some(animation::SkelAnimationHeader {
                        blend_shapes: CURVE_NAMES.iter().map(|s| (*s).into()).collect(),
                        joints: vec![],
                    }),
                    start_time_code_since_epoch: epoch_seconds,
                },
            ),
        ),
    }
}

pub fn mock_frame(start: u64, pcm: Vec<u8>, pattern: MockPattern) -> animation::AnimationData {
    let mut values = vec![0.0; CURVE_NAMES.len()];
    let index = match pattern {
        MockPattern::JawOpenPulse => 17,
        MockPattern::EyeBlinkLeft => 0,
        MockPattern::EyeBlinkRight => 7,
        MockPattern::MouthSmileLeft => 23,
        MockPattern::MouthSmileRight => 24,
    };
    let phase = (start % SAMPLE_RATE) as f32 / SAMPLE_RATE as f32;
    values[index] = 1.0 - (2.0 * phase - 1.0).abs();
    let time_code = start as f64 / SAMPLE_RATE as f64;
    animation::AnimationData {
        skel_animation: Some(animation::SkelAnimation {
            blend_shape_weights: vec![animation::FloatArrayWithTimeCode { time_code, values }],
            ..Default::default()
        }),
        audio: Some(animation::AudioWithTimeCode {
            time_code,
            audio_buffer: pcm,
        }),
        ..Default::default()
    }
}
