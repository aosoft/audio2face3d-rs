use super::{decode_audio_format, encode_audio_format, protocol, remote, unsupported};
use crate::wire::{a2f, controller};
use audio2face3d_types::{
    AudioFormat, BlendshapeParameters, EmotionParameters, EmotionPostProcessing, FaceParameters,
    RequestOptions, Result,
};
use std::{collections::HashMap, time::Duration};

/// Deadline is not a header field. The transport driver must apply this budget.
#[derive(Debug)]
pub struct EncodedRequest {
    pub header: controller::AudioStreamHeader,
    pub timeout: Option<Duration>,
}

fn validate_profile(format: AudioFormat) -> Result<()> {
    if format.channels() != 1 || ![16000, 44100, 48000].contains(&format.sample_rate()) {
        return Err(unsupported(
            "controller input requires mono PCM16LE at 16/44.1/48 kHz",
        ));
    }
    Ok(())
}

pub fn encode_request(value: RequestOptions) -> Result<EncodedRequest> {
    value.validate()?;
    validate_profile(value.input_format)?;
    Ok(EncodedRequest {
        timeout: value.timeout,
        header: controller::AudioStreamHeader {
            audio_header: Some(encode_audio_format(value.input_format)?),
            face_params: value.face.map(encode_face),
            blendshape_params: value.blendshapes.map(|p| a2f::BlendShapeParameters {
                bs_weight_multipliers: p.multipliers.into_iter().collect(),
                bs_weight_offsets: p.offsets.into_iter().collect(),
                enable_clamping_bs_weight: p.clamp,
            }),
            emotion_params: value.emotion.map(|p| a2f::EmotionParameters {
                live_transition_time: p.transition_time,
                beginning_emotion: p.beginning.into_iter().collect(),
            }),
            emotion_post_processing_params: value.emotion_post_processing.map(|p| {
                a2f::EmotionPostProcessingParameters {
                    emotion_contrast: p.contrast,
                    live_blend_coef: p.smoothing,
                    enable_preferred_emotion: p.use_preferred,
                    preferred_emotion_strength: p.preferred_strength,
                    emotion_strength: p.strength,
                    max_emotions: p.max_emotions.map(|n| n as i32),
                }
            }),
        },
    })
}

/// The RPC deadline must be read separately; a header alone has no timeout.
pub fn decode_request(value: controller::AudioStreamHeader) -> Result<RequestOptions> {
    let format = decode_audio_format(
        value
            .audio_header
            .ok_or_else(|| protocol("missing audio header"))?,
    )?;
    validate_profile(format)?;
    let mut result = RequestOptions::new(format);
    result.face = value.face_params.map(decode_face).transpose()?;
    result.blendshapes = value.blendshape_params.map(|p| {
        let mut value = BlendshapeParameters::default();
        value.multipliers = p.bs_weight_multipliers.into_iter().collect();
        value.offsets = p.bs_weight_offsets.into_iter().collect();
        value.clamp = p.enable_clamping_bs_weight;
        value
    });
    result.emotion = value.emotion_params.map(|p| {
        let mut value = EmotionParameters::default();
        value.transition_time = p.live_transition_time;
        value.beginning = p.beginning_emotion.into_iter().collect();
        value
    });
    result.emotion_post_processing = value
        .emotion_post_processing_params
        .map(|p| {
            let mut value = EmotionPostProcessing::default();
            value.contrast = p.emotion_contrast;
            value.smoothing = p.live_blend_coef;
            value.use_preferred = p.enable_preferred_emotion;
            value.preferred_strength = p.preferred_emotion_strength;
            value.strength = p.emotion_strength;
            value.max_emotions = p
                .max_emotions
                .map(|n| u32::try_from(n).map_err(|_| protocol("negative max emotions")))
                .transpose()?;
            Ok::<_, audio2face3d_types::Error>(value)
        })
        .transpose()?;
    result.validate().map_err(remote)?;
    Ok(result)
}

fn encode_face(value: FaceParameters) -> a2f::FaceParameters {
    let mut float_params = HashMap::new();
    if let Some(v) = value.upper_face_smoothing {
        float_params.insert("upperFaceSmoothing".into(), v);
    }
    if let Some(v) = value.lower_face_smoothing {
        float_params.insert("lowerFaceSmoothing".into(), v);
    }
    if let Some(v) = value.upper_face_strength {
        float_params.insert("upperFaceStrength".into(), v);
    }
    if let Some(v) = value.lower_face_strength {
        float_params.insert("lowerFaceStrength".into(), v);
    }
    if let Some(v) = value.face_mask_level {
        float_params.insert("faceMaskLevel".into(), v);
    }
    if let Some(v) = value.face_mask_softness {
        float_params.insert("faceMaskSoftness".into(), v);
    }
    if let Some(v) = value.skin_strength {
        float_params.insert("skinStrength".into(), v);
    }
    if let Some(v) = value.blink_strength {
        float_params.insert("blinkStrength".into(), v);
    }
    if let Some(v) = value.blink_offset {
        float_params.insert("blinkOffset".into(), v);
    }
    if let Some(v) = value.eyelid_open_offset {
        float_params.insert("eyelidOpenOffset".into(), v);
    }
    if let Some(v) = value.lip_open_offset {
        float_params.insert("lipOpenOffset".into(), v);
    }
    if let Some(v) = value.tongue_strength {
        float_params.insert("tongueStrength".into(), v);
    }
    if let Some(v) = value.tongue_height_offset {
        float_params.insert("tongueHeightOffset".into(), v);
    }
    if let Some(v) = value.tongue_depth_offset {
        float_params.insert("tongueDepthOffset".into(), v);
    }
    a2f::FaceParameters {
        float_params,
        ..Default::default()
    }
}
fn decode_face(value: a2f::FaceParameters) -> Result<FaceParameters> {
    if !value.integer_params.is_empty() || !value.float_array_params.is_empty() {
        return Err(unsupported("integer/array face parameters are unsupported"));
    }
    let mut result = FaceParameters::default();
    for (name, value) in value.float_params {
        match name.as_str() {
            "upperFaceSmoothing" => result.upper_face_smoothing = Some(value),
            "lowerFaceSmoothing" => result.lower_face_smoothing = Some(value),
            "upperFaceStrength" => result.upper_face_strength = Some(value),
            "lowerFaceStrength" => result.lower_face_strength = Some(value),
            "faceMaskLevel" => result.face_mask_level = Some(value),
            "faceMaskSoftness" => result.face_mask_softness = Some(value),
            "skinStrength" => result.skin_strength = Some(value),
            "blinkStrength" => result.blink_strength = Some(value),
            "blinkOffset" => result.blink_offset = Some(value),
            "eyelidOpenOffset" => result.eyelid_open_offset = Some(value),
            "lipOpenOffset" => result.lip_open_offset = Some(value),
            "tongueStrength" => result.tongue_strength = Some(value),
            "tongueHeightOffset" => result.tongue_height_offset = Some(value),
            "tongueDepthOffset" => result.tongue_depth_offset = Some(value),
            _ => return Err(unsupported(format!("unknown face parameter: {name}"))),
        }
    }
    result.validate().map_err(remote)?;
    Ok(result)
}
