use crate::{
    config,
    error::{Error, Result},
};
use serde::Serialize;
pub fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|i| a[i] - b[i])
}
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
#[derive(Clone, Debug, Serialize)]
pub struct Transform {
    pub basis: [[f32; 3]; 3],
    pub scale: f32,
    pub source_center: [f32; 3],
    pub center: [f32; 3],
}
impl Transform {
    pub fn new(
        config: &config::Transform,
        positions: impl Iterator<Item = [f32; 3]>,
    ) -> Result<Self> {
        let up = config.source_up.vector();
        let forward = config.source_forward.vector();
        let basis = [cross(up, forward), up, forward];
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in positions {
            for i in 0..3 {
                let v = dot(basis[i], p);
                min[i] = min[i].min(v);
                max[i] = max[i].max(v);
            }
        }
        let height = max[1] - min[1];
        if !height.is_finite() || height <= 1e-12 {
            return Err(Error::Input("retained neutral has no finite height".into()));
        }
        let scale = config.fit_height_m / height;
        let source_center = std::array::from_fn(|i| min[i] + (max[i] - min[i]) * 0.5);
        if !scale.is_finite() || source_center.iter().any(|x| !x.is_finite()) {
            return Err(Error::Input("transform overflow".into()));
        }
        Ok(Self {
            basis,
            scale,
            source_center,
            center: config.center_m,
        })
    }
    pub fn position(&self, p: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|i| {
            (dot(self.basis[i], p) - self.source_center[i]) * self.scale + self.center[i]
        })
    }
    pub fn delta(&self, p: [f32; 3]) -> [f32; 3] {
        self.basis.map(|axis| dot(axis, p) * self.scale)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handedness_and_neutral_transform_are_shared() {
        let c = config::Transform {
            source_up: config::Axis::Z,
            source_forward: config::Axis::NegY,
            fit_height_m: 2.,
            center_m: [1., 2., 3.],
        };
        let t = Transform::new(&c, [[0., 0., 0.], [0., 0., 4.]].into_iter()).unwrap();
        assert_eq!(t.position([0., 0., 0.]), [1., 1., 3.]);
        assert_eq!(t.position([0., 0., 8.]), [1., 5., 3.]);
        assert_eq!(t.delta([2., 0., 0.]), [1., 0., 0.]);
        assert_eq!(dot(cross(t.basis[0], t.basis[1]), t.basis[2]), 1.);
    }
}
