use crate::{audio::SAMPLE_RATE, config::MockPattern};
use audio2face3d_types::{
    AudioBlock, AudioFormat, CurveFrame, CurveLayout, LayoutId, MediaTime, OutputBatch, PcmBuffer,
    Result, SamplePosition,
};
use std::sync::{Arc, OnceLock};
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

pub fn layout() -> Arc<CurveLayout> {
    static LAYOUT: OnceLock<Arc<CurveLayout>> = OnceLock::new();
    LAYOUT
        .get_or_init(|| {
            Arc::new(
                CurveLayout::new(
                    LayoutId(0),
                    CURVE_NAMES.iter().map(|s| (*s).into()).collect(),
                )
                .expect("fixed unique curve names"),
            )
        })
        .clone()
}
pub(crate) fn frame(start: u64, pcm: Vec<u8>, values: Vec<f32>) -> Result<OutputBatch> {
    Ok(OutputBatch {
        audio: Some(AudioBlock::new(
            AudioFormat::MONO_16KHZ,
            SamplePosition(start),
            PcmBuffer::from_vec(pcm)?,
        )?),
        curves: vec![CurveFrame::new(
            layout(),
            MediaTime::from_samples(SamplePosition(start), SAMPLE_RATE as u32)?,
            values,
        )?],
        ..Default::default()
    })
}
pub fn diagnostic_frame(
    start: u64,
    pcm: Vec<u8>,
    pattern: MockPattern,
    curve: Option<&str>,
    weight: Option<f32>,
    jaw: Option<f32>,
) -> Result<OutputBatch> {
    let mut values = vec![0.0; CURVE_NAMES.len()];
    let index = match pattern {
        MockPattern::JawOpenPulse => 17,
        MockPattern::EyeBlinkLeft => 0,
        MockPattern::EyeBlinkRight => 7,
        MockPattern::MouthSmileLeft => 23,
        MockPattern::MouthSmileRight => 24,
    };
    let phase = (start % SAMPLE_RATE) as f32 / SAMPLE_RATE as f32;
    let index = curve
        .and_then(|name| CURVE_NAMES.iter().position(|candidate| *candidate == name))
        .unwrap_or(index);
    values[index] = weight.unwrap_or_else(|| 1.0 - (2.0 * phase - 1.0).abs());
    if let Some(jaw) = jaw {
        values[17] = jaw;
    }
    frame(start, pcm, values)
}

#[cfg(any(feature = "runtime", test))]
pub(crate) fn ordered_weights(mut weights: Vec<f32>, order: &[usize], clamp: bool) -> Vec<f32> {
    if order.iter().copied().eq(0..weights.len()) {
        if clamp {
            for value in &mut weights {
                *value = value.clamp(0.0, 1.0);
            }
        }
        weights
    } else {
        order
            .iter()
            .map(|&i| {
                if clamp {
                    weights[i].clamp(0.0, 1.0)
                } else {
                    weights[i]
                }
            })
            .collect()
    }
}
