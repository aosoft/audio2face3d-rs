use super::{
    decode_audio_format, decode_emotion_trace, encode_audio_format, encode_emotion_trace, protocol,
    remote, unsupported,
};
use crate::wire::{animation, controller};
use audio2face3d_types::{
    AudioBlock, CurveFrame, CurveLayout, Diagnostic, LayoutId, MediaTime, OutputBatch, PcmBuffer,
    Result, Severity, StreamInfo,
};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, UNIX_EPOCH},
};

pub fn decode_stream_info(
    value: controller::AnimationDataStreamHeader,
    layout_id: LayoutId,
) -> Result<StreamInfo> {
    let audio = value.audio_header.map(decode_audio_format).transpose()?;
    let curves = value
        .skel_animation_header
        .map(|v| {
            if !v.joints.is_empty() {
                return Err(unsupported("joint layouts are unsupported"));
            }
            CurveLayout::new(layout_id, v.blend_shapes)
                .map(Arc::new)
                .map_err(remote)
        })
        .transpose()?;
    if audio.is_none() && curves.is_none() {
        return Err(protocol("output header has no supported media"));
    }
    let seconds = value.start_time_code_since_epoch;
    let duration =
        Duration::try_from_secs_f64(seconds).map_err(|_| protocol("invalid start time"))?;
    let mut info = StreamInfo::new(audio, curves);
    // This proto3 scalar has no presence bit: zero is the unspecified epoch.
    info.started_at = if seconds == 0.0 {
        None
    } else {
        Some(
            UNIX_EPOCH
                .checked_add(duration)
                .ok_or_else(|| protocol("start time overflow"))?,
        )
    };
    Ok(info)
}

pub fn encode_stream_info(value: StreamInfo) -> Result<controller::AnimationDataStreamHeader> {
    if value.audio_format.is_none() && value.curves.is_none() {
        return Err(audio2face3d_types::Error::invalid(
            "output header needs audio or curves",
        ));
    }
    let epoch = value
        .started_at
        .map(|v| {
            v.duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .map_err(|_| audio2face3d_types::Error::invalid("start time before epoch"))
        })
        .transpose()?
        .unwrap_or(0.0);
    Ok(controller::AnimationDataStreamHeader {
        audio_header: value.audio_format.map(encode_audio_format).transpose()?,
        skel_animation_header: value.curves.map(|layout| animation::SkelAnimationHeader {
            blend_shapes: match Arc::try_unwrap(layout) {
                Ok(layout) => layout.into_names(),
                Err(layout) => layout.names().to_vec(),
            },
            joints: vec![],
        }),
        start_time_code_since_epoch: epoch,
    })
}

/// Packet conversion only; the stream driver additionally checks continuity.
pub fn decode_animation(value: animation::AnimationData, info: &StreamInfo) -> Result<OutputBatch> {
    if value.camera.is_some() {
        return Err(unsupported("camera data is unsupported"));
    }
    let mut result = OutputBatch::default();
    if let Some(skel) = value.skel_animation {
        if !skel.translations.is_empty() || !skel.rotations.is_empty() || !skel.scales.is_empty() {
            return Err(unsupported("joint animation is unsupported"));
        }
        let mut last = None;
        for frame in skel.blend_shape_weights {
            let layout = info
                .curves
                .as_ref()
                .ok_or_else(|| protocol("curve data without a layout"))?;
            let time = MediaTime::from_seconds(frame.time_code).map_err(remote)?;
            if last.is_some_and(|p| time <= p) {
                return Err(protocol("non-increasing curve times"));
            }
            last = Some(time);
            result
                .curves
                .push(CurveFrame::new(layout.clone(), time, frame.values).map_err(remote)?);
        }
    }
    if let Some(audio) = value.audio {
        let format = info
            .audio_format
            .ok_or_else(|| protocol("audio data without a format"))?;
        let time = MediaTime::from_seconds(audio.time_code).map_err(remote)?;
        let position = time.nearest_sample(format.sample_rate()).map_err(remote)?;
        let quantized = MediaTime::from_samples(position, format.sample_rate()).map_err(remote)?;
        if time.as_nanos().abs_diff(quantized.as_nanos()) > 1 {
            return Err(protocol("audio time is not aligned to a sample"));
        }
        result.audio = Some(
            AudioBlock::new(
                format,
                position,
                PcmBuffer::from_vec(audio.audio_buffer).map_err(remote)?,
            )
            .map_err(remote)?,
        );
    }
    for (name, value) in value.metadata {
        if name == "emotion_aggregate" {
            result.emotion = Some(decode_emotion_trace(value)?);
        } else {
            result.diagnostics.push(Diagnostic {
                severity: Severity::Warning,
                message: format!("ignored optional metadata: {name} ({})", value.type_url),
            });
        }
    }
    Ok(result)
}

/// Diagnostics are separate events; this function refuses to silently discard them.
pub fn encode_animation(value: OutputBatch) -> Result<animation::AnimationData> {
    if !value.diagnostics.is_empty() {
        return Err(unsupported(
            "emit diagnostics separately from animation data",
        ));
    }
    let mut frames = Vec::with_capacity(value.curves.len());
    let mut layout: Option<Arc<CurveLayout>> = None;
    let mut last = None;
    for frame in value.curves {
        let (current, time, values) = frame.into_parts();
        if layout.as_ref().is_some_and(|l| **l != *current) {
            return Err(audio2face3d_types::Error::invalid("mixed curve layouts"));
        }
        if last.is_some_and(|p| time <= p) {
            return Err(audio2face3d_types::Error::invalid(
                "non-increasing curve times",
            ));
        }
        layout = Some(current);
        last = Some(time);
        frames.push(animation::FloatArrayWithTimeCode {
            time_code: time.as_seconds(),
            values,
        });
    }
    let audio = value
        .audio
        .map(|block| {
            let (format, position, pcm) = block.into_parts();
            Ok::<_, audio2face3d_types::Error>(animation::AudioWithTimeCode {
                time_code: MediaTime::from_samples(position, format.sample_rate())?.as_seconds(),
                audio_buffer: pcm.into_vec(),
            })
        })
        .transpose()?;
    let mut metadata = HashMap::new();
    if let Some(trace) = value.emotion {
        metadata.insert("emotion_aggregate".into(), encode_emotion_trace(trace));
    }
    Ok(animation::AnimationData {
        skel_animation: if frames.is_empty() {
            None
        } else {
            Some(animation::SkelAnimation {
                blend_shape_weights: frames,
                ..Default::default()
            })
        },
        audio,
        camera: None,
        metadata,
    })
}
