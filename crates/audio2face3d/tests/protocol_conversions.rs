#![cfg(any(feature = "client-grpc", feature = "grpc-server"))]
use audio2face3d::protocol::{
    convert::*,
    wire::{self, a2f, animation, controller},
};
use audio2face3d::types::*;
use prost::Message;
use std::{collections::HashMap, sync::Arc, time::Duration};

fn header() -> controller::AudioStreamHeader {
    controller::AudioStreamHeader {
        audio_header: Some(encode_audio_format(AudioFormat::MONO_16KHZ).unwrap()),
        ..Default::default()
    }
}
fn info() -> StreamInfo {
    StreamInfo::new(
        Some(AudioFormat::MONO_16KHZ),
        Some(Arc::new(
            CurveLayout::new(LayoutId(1), vec!["A".into(), "B".into()]).unwrap(),
        )),
    )
}
fn frame(time: f64) -> animation::FloatArrayWithTimeCode {
    animation::FloatArrayWithTimeCode {
        time_code: time,
        values: vec![0.1, 0.7],
    }
}

#[test]
fn preserves_absent_and_empty_parameter_containers() {
    for present in [false, true] {
        let mut h = header();
        if present {
            h.face_params = Some(Default::default());
            h.blendshape_params = Some(Default::default());
            h.emotion_params = Some(Default::default());
            h.emotion_post_processing_params = Some(Default::default());
        }
        let result = encode_request(decode_request(h.clone()).unwrap()).unwrap();
        assert_eq!(result.header, h);
        assert_eq!(result.timeout, None);
    }
}

#[test]
fn all_named_parameters_and_optional_zero_false_round_trip() {
    let mut h = header();
    let keys = [
        "upperFaceSmoothing",
        "lowerFaceSmoothing",
        "upperFaceStrength",
        "lowerFaceStrength",
        "faceMaskLevel",
        "faceMaskSoftness",
        "skinStrength",
        "blinkStrength",
        "blinkOffset",
        "eyelidOpenOffset",
        "lipOpenOffset",
        "tongueStrength",
        "tongueHeightOffset",
        "tongueDepthOffset",
    ];
    h.face_params = Some(a2f::FaceParameters {
        float_params: keys
            .into_iter()
            .enumerate()
            .map(|(i, k)| (k.into(), i as f32 / 10.0))
            .collect(),
        ..Default::default()
    });
    h.blendshape_params = Some(a2f::BlendShapeParameters {
        bs_weight_multipliers: HashMap::from([("JawOpen".into(), 0.0)]),
        bs_weight_offsets: HashMap::from([("MouthClose".into(), 0.0)]),
        enable_clamping_bs_weight: Some(false),
    });
    h.emotion_params = Some(a2f::EmotionParameters {
        live_transition_time: Some(0.5),
        beginning_emotion: HashMap::from([("joy".into(), 0.0)]),
    });
    h.emotion_post_processing_params = Some(a2f::EmotionPostProcessingParameters {
        emotion_contrast: Some(0.3),
        live_blend_coef: Some(0.0),
        enable_preferred_emotion: Some(false),
        preferred_emotion_strength: Some(0.0),
        emotion_strength: Some(0.0),
        max_emotions: Some(1),
    });
    let decoded = decode_request(h.clone()).unwrap();
    assert_eq!(
        decoded.face.as_ref().unwrap().upper_face_smoothing,
        Some(0.0)
    );
    assert_eq!(
        decoded
            .emotion_post_processing
            .as_ref()
            .unwrap()
            .use_preferred,
        Some(false)
    );
    assert_eq!(encode_request(decoded).unwrap().header, h);
}

#[test]
fn deadline_is_carried_separately_without_being_lost() {
    let mut request = RequestOptions::default();
    request.timeout = Some(Duration::from_secs(3));
    let encoded = encode_request(request).unwrap();
    assert_eq!(encoded.timeout, Some(Duration::from_secs(3)));
    assert_eq!(decode_request(encoded.header).unwrap().timeout, None);
}

#[test]
fn malformed_or_unsupported_requests_are_rejected() {
    assert!(decode_request(Default::default()).is_err());
    let mut h = header();
    h.audio_header.as_mut().unwrap().samples_per_second = 8000;
    assert_eq!(
        decode_request(h).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    for params in [
        a2f::FaceParameters {
            integer_params: HashMap::from([("x".into(), 0)]),
            ..Default::default()
        },
        a2f::FaceParameters {
            float_params: HashMap::from([("unknown".into(), 0.0)]),
            ..Default::default()
        },
        a2f::FaceParameters {
            float_params: HashMap::from([("upperFaceStrength".into(), f32::NAN)]),
            ..Default::default()
        },
    ] {
        let mut h = header();
        h.face_params = Some(params);
        assert!(decode_request(h).is_err());
    }
    let mut h = header();
    h.emotion_post_processing_params = Some(a2f::EmotionPostProcessingParameters {
        max_emotions: Some(-1),
        ..Default::default()
    });
    assert!(decode_request(h).is_err());
}

#[test]
fn input_conversion_moves_pcm_and_preserves_sparse_emotion() {
    let data = vec![1, 2, 3, 4];
    let ptr = data.as_ptr();
    let input = a2f::AudioWithEmotion {
        audio_buffer: data,
        emotions: vec![
            wire::nvidia_ace::emotion_with_timecode::v1::EmotionWithTimeCode {
                time_code: 0.1,
                emotion: HashMap::from([("joy".into(), 0.0)]),
            },
        ],
    };
    let decoded = decode_input(input, AudioFormat::MONO_16KHZ).unwrap();
    assert_eq!(decoded.pcm().as_bytes().as_ptr(), ptr);
    assert_eq!(decoded.emotions()[0].values().get("joy"), Some(&0.0));
    let encoded = encode_input(decoded, AudioFormat::MONO_16KHZ).unwrap();
    assert_eq!(encoded.audio_buffer.as_ptr(), ptr);
    assert_eq!(encoded.emotions[0].emotion.len(), 1);
}

#[test]
fn rejects_invalid_input_pcm_and_emotion() {
    assert!(
        decode_input(
            a2f::AudioWithEmotion {
                audio_buffer: vec![0],
                emotions: vec![]
            },
            AudioFormat::MONO_16KHZ
        )
        .is_err()
    );
    for (time, value) in [(f64::NAN, 0.1), (-1.0, 0.1), (0.0, f32::NAN), (0.0, 1.1)] {
        let input = a2f::AudioWithEmotion {
            audio_buffer: vec![0; 2],
            emotions: vec![
                wire::nvidia_ace::emotion_with_timecode::v1::EmotionWithTimeCode {
                    time_code: time,
                    emotion: HashMap::from([("joy".into(), value)]),
                },
            ],
        };
        assert!(decode_input(input, AudioFormat::MONO_16KHZ).is_err());
    }
}

#[test]
fn output_header_transfers_layout_and_keeps_epoch_optional() {
    let value = info();
    let encoded = encode_stream_info(value.clone()).unwrap();
    let decoded = decode_stream_info(encoded, LayoutId(1)).unwrap();
    assert_eq!(decoded, value);
    let mut h = encode_stream_info(value).unwrap();
    h.start_time_code_since_epoch = f64::INFINITY;
    assert!(decode_stream_info(h, LayoutId(1)).is_err());
    let h = controller::AnimationDataStreamHeader {
        skel_animation_header: Some(animation::SkelAnimationHeader {
            blend_shapes: vec!["A".into(), "A".into()],
            joints: vec![],
        }),
        ..Default::default()
    };
    assert!(decode_stream_info(h, LayoutId(1)).is_err());
}

#[test]
fn multiple_frames_and_audio_move_without_copying_or_dropping() {
    let context = info();
    let frames = vec![frame(0.0), frame(0.1)];
    let first_ptr = frames[0].values.as_ptr();
    let second_ptr = frames[1].values.as_ptr();
    let pcm = vec![1, 2, 3, 4];
    let pcm_ptr = pcm.as_ptr();
    let value = animation::AnimationData {
        skel_animation: Some(animation::SkelAnimation {
            blend_shape_weights: frames,
            ..Default::default()
        }),
        audio: Some(animation::AudioWithTimeCode {
            time_code: 0.0,
            audio_buffer: pcm,
        }),
        ..Default::default()
    };
    let decoded = decode_animation(value, &context).unwrap();
    assert_eq!(decoded.curves.len(), 2);
    assert_eq!(decoded.curves[0].values().as_ptr(), first_ptr);
    assert_eq!(decoded.curves[1].values().as_ptr(), second_ptr);
    assert!(Arc::ptr_eq(
        decoded.curves[0].layout(),
        context.curves.as_ref().unwrap()
    ));
    assert_eq!(
        decoded.audio.as_ref().unwrap().pcm().as_bytes().as_ptr(),
        pcm_ptr
    );
    let encoded = encode_animation(decoded).unwrap();
    assert_eq!(encoded.audio.unwrap().audio_buffer.as_ptr(), pcm_ptr);
    let frames = encoded.skel_animation.unwrap().blend_shape_weights;
    assert_eq!(frames[0].values.as_ptr(), first_ptr);
    assert_eq!(frames[1].values.as_ptr(), second_ptr);
}

#[test]
fn audio_only_and_curve_only_packets_are_independent() {
    let audio = animation::AnimationData {
        audio: Some(animation::AudioWithTimeCode {
            time_code: 0.1,
            audio_buffer: vec![0; 2],
        }),
        ..Default::default()
    };
    let value = decode_animation(audio, &info()).unwrap();
    assert!(value.curves.is_empty());
    assert_eq!(value.audio.unwrap().position(), SamplePosition(1600));
    let curves = animation::AnimationData {
        skel_animation: Some(animation::SkelAnimation {
            blend_shape_weights: vec![frame(0.0)],
            ..Default::default()
        }),
        ..Default::default()
    };
    let value = decode_animation(curves, &info()).unwrap();
    assert!(value.audio.is_none());
    assert_eq!(value.curves.len(), 1);
}

#[test]
fn corrupt_timing_layout_or_unsupported_payload_is_not_silently_accepted() {
    for frames in [
        vec![frame(0.1), frame(0.0)],
        vec![frame(0.0), frame(0.0)],
        vec![frame(f64::NAN)],
        vec![animation::FloatArrayWithTimeCode {
            time_code: 0.0,
            values: vec![0.0],
        }],
    ] {
        let data = animation::AnimationData {
            skel_animation: Some(animation::SkelAnimation {
                blend_shape_weights: frames,
                ..Default::default()
            }),
            ..Default::default()
        };
        assert!(decode_animation(data, &info()).is_err());
    }
    let camera = animation::AnimationData {
        camera: Some(Default::default()),
        ..Default::default()
    };
    assert_eq!(
        decode_animation(camera, &info()).unwrap_err().kind(),
        ErrorKind::Unsupported
    );
    let joint = animation::AnimationData {
        skel_animation: Some(animation::SkelAnimation {
            translations: vec![Default::default()],
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(decode_animation(joint, &info()).is_err());
    let data = animation::AnimationData {
        audio: Some(animation::AudioWithTimeCode {
            time_code: 0.00003,
            audio_buffer: vec![0; 2],
        }),
        ..Default::default()
    };
    assert!(decode_animation(data, &info()).is_err());
}

#[test]
fn emotion_metadata_preserves_all_three_series_and_empty_lists() {
    let make = |time, value| {
        EmotionKeyframe::new(
            MediaTime::from_seconds(time).unwrap(),
            [("joy".into(), value)].into_iter().collect(),
        )
        .unwrap()
    };
    let trace = EmotionTrace {
        input: vec![make(0.0, 0.0), make(0.1, 0.2)],
        mixed: vec![],
        smoothed: vec![make(0.0, 0.1)],
    };
    let encoded = encode_emotion_trace(trace.clone());
    assert_eq!(decode_emotion_trace(encoded).unwrap(), trace);
    let bad = prost_types::Any {
        type_url: "type.googleapis.com/Other".into(),
        value: vec![],
    };
    assert!(decode_emotion_trace(bad).is_err());
    let bad = prost_types::Any {
        type_url: "type.googleapis.com/nvidia_ace.emotion_aggregate.v1.EmotionAggregate".into(),
        value: vec![0xff],
    };
    assert!(decode_emotion_trace(bad).is_err());
}

#[test]
fn unknown_optional_metadata_is_reported_and_not_reencoded_silently() {
    let data = animation::AnimationData {
        metadata: HashMap::from([(
            "extension".into(),
            prost_types::Any {
                type_url: "Other".into(),
                value: vec![],
            },
        )]),
        ..Default::default()
    };
    let batch = decode_animation(data, &info()).unwrap();
    assert_eq!(batch.diagnostics.len(), 1);
    assert!(encode_animation(batch).is_err());
}

#[test]
fn descriptor_declares_the_original_service_and_duplex_method() {
    let set = prost_types::FileDescriptorSet::decode(wire::DESCRIPTOR).unwrap();
    let file = set
        .file
        .iter()
        .find(|f| f.package.as_deref() == Some("nvidia_ace.services.a2f_controller.v1"))
        .unwrap();
    let service = &file.service[0];
    assert_eq!(service.name.as_deref(), Some("A2FControllerService"));
    assert_eq!(
        wire::SERVICE_NAME,
        "nvidia_ace.services.a2f_controller.v1.A2FControllerService"
    );
    assert_eq!(
        service.method[0].name.as_deref(),
        Some("ProcessAudioStream")
    );
    assert_eq!(service.method[0].client_streaming, Some(true));
    assert_eq!(service.method[0].server_streaming, Some(true));
}
