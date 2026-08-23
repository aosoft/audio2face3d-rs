//! Strict parser for the SDK blendshape solver configuration.

use crate::{Audio2xError, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BlendshapeConfigRoot {
    pub blendshape_params: BlendshapeConfig,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BlendshapeConfig {
    #[serde(rename = "strengthL2regularization")]
    pub l2_regularization: f32,
    #[serde(rename = "strengthTemporalSmoothing")]
    pub temporal_regularization: f32,
    #[serde(rename = "strengthL1regularization")]
    pub l1_regularization: f32,
    #[serde(rename = "strengthSymmetry")]
    pub symmetry_regularization: f32,
    #[serde(rename = "numPoses")]
    pub num_poses: usize,
    #[serde(rename = "bsSolveActivePoses")]
    pub active_poses: Vec<i32>,
    #[serde(rename = "bsSolveCancelPoses")]
    pub cancel_poses: Vec<i32>,
    #[serde(rename = "bsSolveSymmetryPoses")]
    pub symmetry_poses: Vec<i32>,
    #[serde(rename = "bsWeightMultipliers")]
    pub multipliers: Vec<f32>,
    #[serde(rename = "bsWeightOffsets")]
    pub offsets: Vec<f32>,
    #[serde(rename = "templateBBSize", default = "default_template_bb_size")]
    pub template_bb_size: f32,
    #[serde(default = "default_tolerance")]
    pub tolerance: f32,
}

const fn default_template_bb_size() -> f32 {
    54.7
}

const fn default_tolerance() -> f32 {
    1.0e-10
}

impl BlendshapeConfig {
    pub fn validate(&self) -> Result<()> {
        if self.num_poses == 0 {
            return Err(invalid("blendshape numPoses must be non-zero"));
        }
        for (name, length) in [
            ("bsSolveActivePoses", self.active_poses.len()),
            ("bsSolveCancelPoses", self.cancel_poses.len()),
            ("bsSolveSymmetryPoses", self.symmetry_poses.len()),
            ("bsWeightMultipliers", self.multipliers.len()),
            ("bsWeightOffsets", self.offsets.len()),
        ] {
            if length != self.num_poses {
                return Err(invalid(&format!(
                    "{name} length {length} does not match numPoses {}",
                    self.num_poses
                )));
            }
        }
        if !self.active_poses.iter().any(|value| *value != 0) {
            return Err(invalid("blendshape active pose set is empty"));
        }
        validate_pairs("bsSolveCancelPoses", &self.cancel_poses)?;
        validate_pairs("bsSolveSymmetryPoses", &self.symmetry_poses)?;
        let regularization = [
            self.l1_regularization,
            self.l2_regularization,
            self.temporal_regularization,
            self.symmetry_regularization,
        ];
        if regularization
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid(
                "blendshape regularization strengths must be finite and non-negative",
            ));
        }
        if !self.template_bb_size.is_finite() || self.template_bb_size <= 0.0 {
            return Err(invalid("blendshape templateBBSize must be positive"));
        }
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(invalid("blendshape tolerance must be positive"));
        }
        if self
            .multipliers
            .iter()
            .chain(&self.offsets)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("blendshape multipliers/offsets must be finite"));
        }
        Ok(())
    }
}

fn validate_pairs(name: &str, values: &[i32]) -> Result<()> {
    let mut counts = HashMap::new();
    for value in values.iter().copied().filter(|value| *value >= 0) {
        *counts.entry(value).or_insert(0_usize) += 1;
    }
    if let Some((pair, count)) = counts.into_iter().find(|(_, count)| *count != 2) {
        return Err(invalid(&format!(
            "{name} pair id {pair} occurs {count} times instead of twice"
        )));
    }
    Ok(())
}

pub fn parse_blendshape_config(json: &str) -> Result<BlendshapeConfigRoot> {
    let value: BlendshapeConfigRoot = serde_json::from_str(json)
        .map_err(|error| invalid(&format!("invalid blendshape JSON/schema: {error}")))?;
    value.blendshape_params.validate()?;
    Ok(value)
}

pub fn load_blendshape_config(path: impl AsRef<Path>) -> Result<BlendshapeConfigRoot> {
    let path = path.as_ref();
    let json = fs::read_to_string(path)
        .map_err(|error| invalid(&format!("{}: {error}", path.display())))?;
    parse_blendshape_config(&json)
}

fn invalid(message: &str) -> Audio2xError {
    Audio2xError::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(active: &str, cancel: &str, symmetry: &str) -> String {
        format!(
            r#"{{"blendshape_params":{{"strengthL2regularization":3.5,"strengthTemporalSmoothing":0.0,"strengthL1regularization":1.0,"strengthSymmetry":100.0,"numPoses":3,"bsSolveActivePoses":{active},"bsSolveCancelPoses":{cancel},"bsSolveSymmetryPoses":{symmetry},"bsWeightMultipliers":[1.0,1.0,1.0],"bsWeightOffsets":[0.0,0.0,0.0]}}}}"#
        )
    }

    #[test]
    fn parses_sdk_encoding_and_defaults() {
        let parsed = parse_blendshape_config(&config("[1,1,0]", "[2,2,-1]", "[7,-1,7]")).unwrap();
        assert_eq!(parsed.blendshape_params.template_bb_size, 54.7);
        assert_eq!(parsed.blendshape_params.tolerance, 1.0e-10);
    }

    #[test]
    fn rejects_empty_active_set_and_malformed_pairs() {
        assert!(parse_blendshape_config(&config("[0,0,0]", "[-1,-1,-1]", "[-1,-1,-1]")).is_err());
        assert!(parse_blendshape_config(&config("[1,0,0]", "[3,-1,-1]", "[-1,-1,-1]")).is_err());
    }

    #[test]
    fn parses_installed_sdk_configs_when_available() {
        let Some(paths) = std::env::var_os("AUDIO2X_TEST_BLENDSHAPE_CONFIGS") else {
            return;
        };
        let mut parsed = 0;
        for path in std::env::split_paths(&paths) {
            load_blendshape_config(path).unwrap();
            parsed += 1;
        }
        assert!(parsed > 0);
    }
}
