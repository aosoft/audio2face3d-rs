use crate::{
    config::MockPattern,
    proto::{animation, audio, controller},
};

pub use audio2face3d::inference::animation::CURVE_NAMES;

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
    diagnostic_frame(start, pcm, pattern, None, None)
}

pub fn diagnostic_frame(
    start: u64,
    pcm: Vec<u8>,
    pattern: MockPattern,
    curve: Option<&str>,
    weight: Option<f32>,
) -> animation::AnimationData {
    audio2face3d::protocol::convert::encode_animation(
        audio2face3d::inference::animation::diagnostic_frame(
            start,
            pcm,
            pattern.into(),
            curve,
            weight,
            None,
        )
        .expect("valid diagnostic PCM and weights"),
    )
    .expect("valid diagnostic animation")
}
