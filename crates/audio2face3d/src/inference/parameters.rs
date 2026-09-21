use crate::common::GeometryConfig;
use crate::inference::animation::CURVE_NAMES;
use crate::types::RequestOptions;
use crate::types::{Error, ErrorKind};

pub fn face(config: &mut GeometryConfig, header: &RequestOptions) -> Result<(), Error> {
    let Some(params) = &header.face else {
        return Ok(());
    };
    params.validate()?;
    if let Some(value) = params.upper_face_smoothing {
        config.upper_face_smoothing = value;
    }
    if let Some(value) = params.lower_face_smoothing {
        config.lower_face_smoothing = value;
    }
    if let Some(value) = params.upper_face_strength {
        config.upper_face_strength = value;
    }
    if let Some(value) = params.lower_face_strength {
        config.lower_face_strength = value;
    }
    if let Some(value) = params.face_mask_level {
        config.face_mask_level = value;
    }
    if let Some(value) = params.face_mask_softness {
        config.face_mask_softness = value;
    }
    if let Some(value) = params.skin_strength {
        config.skin_strength = value;
    }
    if let Some(value) = params.blink_strength {
        config.blink_strength = value;
    }
    if let Some(value) = params.blink_offset {
        config.blink_offset = value;
    }
    if let Some(value) = params.eyelid_open_offset {
        config.eyelid_open_offset = value;
    }
    if let Some(value) = params.lip_open_offset {
        config.lip_open_offset = value;
    }
    if let Some(value) = params.tongue_strength {
        config.tongue_strength = value;
    }
    if let Some(value) = params.tongue_height_offset {
        config.tongue_height_offset = value;
    }
    if let Some(value) = params.tongue_depth_offset {
        config.tongue_depth_offset = value;
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
    header: &RequestOptions,
    order: &[usize],
    multipliers: &mut [f32],
    offsets: &mut [f32],
) -> Result<bool, Error> {
    let Some(params) = &header.blendshapes else {
        return Ok(false);
    };
    for (values, target) in [
        (&params.multipliers, multipliers),
        (&params.offsets, offsets),
    ] {
        for (name, &value) in values {
            if !value.is_finite() {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    "BlendShape values must be finite",
                ));
            }
            if let Some(index) = CURVE_NAMES.iter().position(|v| v == name) {
                target[order[index]] = value;
            } else if EXTENDED_TONGUE.contains(&name.as_str()) {
                crate::logging::integration::log(crate::logging::LogLevel::Debug, || {
                    crate::logging::LogRecord::new(
                        "extended tongue curve omitted from 52-curve output",
                    )
                    .field("source", module_path!())
                    .field("name", name.as_str())
                });
            } else {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("unknown BlendShape: {name}"),
                ));
            }
        }
    }
    Ok(params.clamp.unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absent_values_keep_model_defaults_and_explicit_zero_overrides() {
        let mut h = RequestOptions::default();
        let order = (0..52).collect::<Vec<_>>();
        let mut multipliers = vec![0.75; 52];
        let mut offsets = vec![0.25; 52];
        assert!(!blendshapes(&h, &order, &mut multipliers, &mut offsets).unwrap());
        h.blendshapes = Some(Default::default());
        let p = h.blendshapes.as_mut().unwrap();
        p.multipliers.insert("JawOpen".into(), 0.0);
        p.offsets.insert("JawOpen".into(), 0.0);
        p.clamp = Some(false);
        assert!(!blendshapes(&h, &order, &mut multipliers, &mut offsets).unwrap());
        let jaw = CURVE_NAMES.iter().position(|n| *n == "JawOpen").unwrap();
        assert_eq!((multipliers[jaw], offsets[jaw]), (0.0, 0.0));
        assert_eq!((multipliers[0], offsets[0]), (0.75, 0.25));
    }
}
