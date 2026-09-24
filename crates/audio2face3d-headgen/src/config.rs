use audio2face3d_gui_core::{ModelError, model::Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub schema_version: u32,
    pub segments: u32,
    pub rings: u32,
    pub width: f32,
    pub height: f32,
    pub depth: f32,
    pub expression_scale: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: 1,
            segments: 32,
            rings: 20,
            width: 0.16,
            height: 0.24,
            depth: 0.14,
            expression_scale: 1.,
        }
    }
}
impl Config {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1
            || !(24..=64).contains(&self.segments)
            || !self.segments.is_multiple_of(2)
            || !(12..=40).contains(&self.rings)
            || !(0.10..=0.30).contains(&self.width)
            || !(0.15..=0.40).contains(&self.height)
            || !(0.08..=0.30).contains(&self.depth)
            || !(0.25..=2.0).contains(&self.expression_scale)
        {
            return Err(ModelError("invalid generator configuration".into()));
        }
        Ok(())
    }
    pub(crate) fn scale(&self, p: [f32; 3]) -> [f32; 3] {
        [
            p[0] * self.width / 0.16,
            p[1] * self.height / 0.24,
            p[2] * self.depth / 0.14,
        ]
    }
    pub(crate) fn unscale(&self, p: [f32; 3]) -> [f32; 3] {
        [
            p[0] * 0.16 / self.width,
            p[1] * 0.24 / self.height,
            p[2] * 0.14 / self.depth,
        ]
    }
}
