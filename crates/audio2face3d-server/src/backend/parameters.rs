use crate::{animation::CURVE_NAMES, proto::controller::AudioStreamHeader};
use audio2face3d::common::GeometryConfig;
use tonic::Status;

pub fn face(config: &mut GeometryConfig, header: &AudioStreamHeader) -> Result<(), Status> {
    let Some(params) = &header.face_params else {
        return Ok(());
    };
    if !params.integer_params.is_empty() || !params.float_array_params.is_empty() {
        return Err(Status::invalid_argument(
            "face integer/array parameters are unsupported",
        ));
    }
    for (name, &value) in &params.float_params {
        if !value.is_finite() {
            return Err(Status::invalid_argument("face values must be finite"));
        }
        let target = match name.as_str() {
            "upperFaceSmoothing" => &mut config.upper_face_smoothing,
            "lowerFaceSmoothing" => &mut config.lower_face_smoothing,
            "upperFaceStrength" => &mut config.upper_face_strength,
            "lowerFaceStrength" => &mut config.lower_face_strength,
            "faceMaskLevel" => &mut config.face_mask_level,
            "faceMaskSoftness" => &mut config.face_mask_softness,
            "skinStrength" => &mut config.skin_strength,
            "blinkStrength" => &mut config.blink_strength,
            "blinkOffset" => &mut config.blink_offset,
            "eyelidOpenOffset" => &mut config.eyelid_open_offset,
            "lipOpenOffset" => &mut config.lip_open_offset,
            "tongueStrength" => &mut config.tongue_strength,
            "tongueHeightOffset" => &mut config.tongue_height_offset,
            "tongueDepthOffset" => &mut config.tongue_depth_offset,
            _ => {
                return Err(Status::invalid_argument(format!(
                    "unknown face parameter: {name}"
                )));
            }
        };
        *target = value;
    }
    Ok(())
}
const EXTENDED_TONGUE: [&str; 16] = [
    "TongueTipUp",
    "TongueTipDown",
    "TongueTipLeft",
    "TongueTipRight",
    "TongueRollUp",
    "TongueRollDown",
    "TongueRollLeft",
    "TongueRollRight",
    "TongueUp",
    "TongueDown",
    "TongueLeft",
    "TongueRight",
    "TongueIn",
    "TongueStretch",
    "TongueWide",
    "TongueNarrow",
];
pub fn blendshapes(
    header: &AudioStreamHeader,
    order: &[usize],
    multipliers: &mut [f32],
    offsets: &mut [f32],
) -> Result<bool, Status> {
    let Some(params) = &header.blendshape_params else {
        return Ok(false);
    };
    for (values, target) in [
        (&params.bs_weight_multipliers, multipliers),
        (&params.bs_weight_offsets, offsets),
    ] {
        for (name, &value) in values {
            if !value.is_finite() {
                return Err(Status::invalid_argument("BlendShape values must be finite"));
            }
            if let Some(index) = CURVE_NAMES.iter().position(|v| v == name) {
                target[order[index]] = value;
            } else if EXTENDED_TONGUE.contains(&name.as_str()) {
                tracing::debug!(name, "extended tongue curve omitted from 52-curve output");
            } else {
                return Err(Status::invalid_argument(format!(
                    "unknown BlendShape: {name}"
                )));
            }
        }
    }
    Ok(params.enable_clamping_bs_weight.unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_values_keep_model_defaults_and_explicit_zero_overrides() {
        let mut h = AudioStreamHeader::default();
        let order = (0..52).collect::<Vec<_>>();
        let mut multipliers = vec![0.75; 52];
        let mut offsets = vec![0.25; 52];
        assert!(!blendshapes(&h, &order, &mut multipliers, &mut offsets).unwrap());
        h.blendshape_params = Some(Default::default());
        let p = h.blendshape_params.as_mut().unwrap();
        p.bs_weight_multipliers.insert("JawOpen".into(), 0.0);
        p.bs_weight_offsets.insert("JawOpen".into(), 0.0);
        p.enable_clamping_bs_weight = Some(false);
        assert!(!blendshapes(&h, &order, &mut multipliers, &mut offsets).unwrap());
        let jaw = CURVE_NAMES.iter().position(|n| *n == "JawOpen").unwrap();
        assert_eq!((multipliers[jaw], offsets[jaw]), (0.0, 0.0));
        assert_eq!((multipliers[0], offsets[0]), (0.75, 0.25));
    }
}
