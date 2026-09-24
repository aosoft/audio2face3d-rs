use crate::error::{Error, Result};
use audio2face3d_gui_core::rig;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub neutral: String,
    pub output_profile: String,
    pub unsupported_channels: Vec<String>,
    pub targets: BTreeMap<String, Vec<String>>,
    pub transform: Transform,
    pub geometry: Geometry,
    pub materials: Materials,
    pub expected: Option<Expected>,
    pub reference: Option<Reference>,
    #[serde(skip)]
    pub(crate) source_sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Transform {
    pub source_up: Axis,
    pub source_forward: Axis,
    pub fit_height_m: f32,
    pub center_m: [f32; 3],
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub enum Axis {
    #[serde(rename = "+X")]
    X,
    #[serde(rename = "-X")]
    NegX,
    #[serde(rename = "+Y")]
    Y,
    #[serde(rename = "-Y")]
    NegY,
    #[serde(rename = "+Z")]
    Z,
    #[serde(rename = "-Z")]
    NegZ,
}
impl Axis {
    pub fn vector(self) -> [f32; 3] {
        match self {
            Self::X => [1., 0., 0.],
            Self::NegX => [-1., 0., 0.],
            Self::Y => [0., 1., 0.],
            Self::NegY => [0., -1., 0.],
            Self::Z => [0., 0., 1.],
            Self::NegZ => [0., 0., -1.],
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Geometry {
    #[serde(default)]
    pub degenerate_pose_triangles: DegeneratePoseTriangles,
    pub split_by: SplitBy,
    pub exclude_materials: Vec<String>,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitBy {
    Material,
    ObjectGroupMaterial,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Materials {
    pub default_color: [f32; 4],
    pub colors: BTreeMap<String, [f32; 4]>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Expected {
    pub source_vertices: usize,
    pub source_faces: usize,
    pub mapped_channels: usize,
    pub unique_expression_files: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub name: String,
    pub repository: String,
    pub definition: String,
    pub reviewed_on: String,
    pub status: String,
}
impl Config {
    pub fn parse(text: &str) -> Result<Self> {
        use sha2::{Digest, Sha256};
        let mut config: Self = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        config.source_sha256 = Sha256::digest(text.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        config.validate()?;
        Ok(config)
    }
    pub fn expression_files(&self) -> BTreeSet<&str> {
        self.targets
            .values()
            .flatten()
            .map(String::as_str)
            .collect()
    }
    pub fn validate(&self) -> Result<()> {
        let fail = |s: &str| Error::Config(s.into());
        if self.schema_version != 1 || self.output_profile != rig::TESTER {
            return Err(fail("unsupported schema or output profile"));
        }
        let expected = rig::CHANNELS.into_iter().collect::<BTreeSet<_>>();
        let targets = self
            .targets
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        let unsupported = self
            .unsupported_channels
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if targets.is_empty()
            || unsupported.len() != self.unsupported_channels.len()
            || !targets.is_disjoint(&unsupported)
            || targets
                .union(&unsupported)
                .copied()
                .collect::<BTreeSet<_>>()
                != expected
        {
            return Err(fail(
                "targets and unsupported_channels must partition the 52 canonical names",
            ));
        }
        let mut files = BTreeSet::new();
        relative_path(&self.neutral)?;
        for paths in self.targets.values() {
            if paths.is_empty() {
                return Err(fail("empty target source list"));
            }
            for path in paths {
                relative_path(path)?;
                if path == &self.neutral || !files.insert(path) {
                    return Err(fail("duplicate expression file"));
                }
            }
        }
        if files.len() > 104 {
            return Err(fail("more than 104 expression files"));
        }
        let t = &self.transform;
        if t.source_up
            .vector()
            .iter()
            .zip(t.source_forward.vector())
            .any(|(a, b)| a * b != 0.)
            || !t.fit_height_m.is_finite()
            || t.fit_height_m <= 0.
            || t.center_m.iter().any(|x| !x.is_finite())
        {
            return Err(fail("invalid transform axes, height or center"));
        }
        for color in
            std::iter::once(&self.materials.default_color).chain(self.materials.colors.values())
        {
            if color[3] != 1.
                || color
                    .iter()
                    .any(|x| !x.is_finite() || !(0.0..=1.).contains(x))
            {
                return Err(fail("materials must be finite opaque RGBA in 0..1"));
            }
        }
        if let Some(e) = &self.expected
            && (e.mapped_channels != targets.len() || e.unique_expression_files != files.len())
        {
            return Err(fail("expected mapping counts do not match configuration"));
        }
        Ok(())
    }
}
pub(crate) fn relative_path(path: &str) -> Result<()> {
    use std::path::{Component, Path};
    if path.trim().is_empty()
        || path.contains(':')
        || path.contains('\\')
        || Path::new(path)
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(Error::Config(format!(
            "expected relative forward-slash input path: {path}"
        )));
    }
    Ok(())
}

/// Neutral geometry remains strict regardless of this explicit pose-only policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DegeneratePoseTriangles {
    #[default]
    Error,
    SkipNormalContribution,
}

#[cfg(test)]
mod tests {
    use super::*;
    const PRESET: &str = include_str!("../presets/ict-facekit.toml");
    #[test]
    fn preset_contract() {
        let c = Config::parse(PRESET).unwrap();
        assert_eq!(c.targets.len(), 51);
        assert_eq!(c.expression_files().len(), 53);
        assert_eq!(c.unsupported_channels, ["TongueOut"]);
    }
    #[test]
    fn invalid_config_is_rejected() {
        for s in [
            PRESET.replace("schema_version = 1", "schema_version = 1\nunknown = 2"),
            PRESET.replace("EyeBlinkLeft =", "EyeBlikLeft ="),
            PRESET.replace("[\"eyeBlink_L.obj\"]", "[]"),
            PRESET.replace("[\"TongueOut\"]", "[\"EyeBlinkLeft\"]"),
            PRESET.replace("eyeBlink_L.obj", "../eyeBlink_L.obj"),
            PRESET.replace("source_forward = \"+Z\"", "source_forward = \"-Y\""),
        ] {
            assert!(Config::parse(&s).is_err());
        }
    }
}
