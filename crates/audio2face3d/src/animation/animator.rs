use crate::common::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkinAnimatorParams {
    pub lower_face_smoothing: f32,
    pub upper_face_smoothing: f32,
    pub lower_face_strength: f32,
    pub upper_face_strength: f32,
    pub face_mask_level: f32,
    pub face_mask_softness: f32,
    pub skin_strength: f32,
    pub blink_strength: f32,
    pub eyelid_open_offset: f32,
    pub lip_open_offset: f32,
    pub blink_offset: f32,
}

#[derive(Debug, Clone)]
struct IirInterpolator {
    smoothing: f32,
    stages: Option<[Vec<f32>; 3]>,
}

impl IirInterpolator {
    fn new(smoothing: f32) -> Self {
        Self {
            smoothing,
            stages: None,
        }
    }

    fn update(&mut self, raw: &[f32], dt: f32) -> Vec<f32> {
        let stages = self
            .stages
            .get_or_insert_with(|| [raw.to_vec(), raw.to_vec(), raw.to_vec()]);
        stages[0].copy_from_slice(raw);
        if self.smoothing > 0.0 {
            let alpha = 1.0 - 0.5_f32.powf(dt / self.smoothing);
            for stage in 1..3 {
                let previous = stages[stage - 1].clone();
                for (value, previous) in stages[stage].iter_mut().zip(previous) {
                    *value += (previous - *value) * alpha;
                }
            }
        } else {
            stages[1] = stages[0].clone();
            stages[2] = stages[1].clone();
        }
        stages[2].clone()
    }
}

#[derive(Debug, Clone)]
pub struct SkinAnimator {
    params: SkinAnimatorParams,
    neutral_pose: Vec<f32>,
    lip_open_pose_delta: Vec<f32>,
    eye_close_pose_delta: Vec<f32>,
    face_mask_lower: Vec<f32>,
    lower: IirInterpolator,
    upper: IirInterpolator,
}

impl SkinAnimator {
    pub fn new(
        params: SkinAnimatorParams,
        neutral_pose: Vec<f32>,
        lip_open_pose_delta: Vec<f32>,
        eye_close_pose_delta: Vec<f32>,
    ) -> Result<Self> {
        if neutral_pose.is_empty() || !neutral_pose.len().is_multiple_of(3) {
            return Err(invalid("skin neutral pose must contain XYZ vertices"));
        }
        if lip_open_pose_delta.len() != neutral_pose.len()
            || eye_close_pose_delta.len() != neutral_pose.len()
        {
            return Err(invalid("skin pose arrays must have matching lengths"));
        }
        if params.face_mask_softness <= 0.0 {
            return Err(invalid("face mask softness must be positive"));
        }
        let ys: Vec<_> = neutral_pose.chunks_exact(3).map(|v| v[1]).collect();
        let min = ys.iter().copied().fold(f32::INFINITY, f32::min);
        let max = ys.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        let face_mask_lower = ys
            .into_iter()
            .map(|y| {
                let normalized = if range > 0.0 { (y - min) / range } else { 0.0 };
                1.0 / (1.0
                    + (-(params.face_mask_level - normalized) / params.face_mask_softness).exp())
            })
            .collect();
        Ok(Self {
            lower: IirInterpolator::new(params.lower_face_smoothing),
            upper: IirInterpolator::new(params.upper_face_smoothing),
            params,
            neutral_pose,
            lip_open_pose_delta,
            eye_close_pose_delta,
            face_mask_lower,
        })
    }

    pub fn neutral_pose(&self) -> &[f32] {
        &self.neutral_pose
    }

    pub fn reset(&mut self) {
        self.lower.stages = None;
        self.upper.stages = None;
    }

    pub fn animate(&mut self, input: &[f32], dt: f32) -> Result<Vec<f32>> {
        if input.len() != self.neutral_pose.len() {
            return Err(invalid("skin input length does not match neutral pose"));
        }
        if !dt.is_finite() {
            return Err(invalid("skin delta time must be finite"));
        }
        let raw: Vec<_> = input
            .iter()
            .zip(&self.eye_close_pose_delta)
            .zip(&self.lip_open_pose_delta)
            .map(|((&value, &eye), &lip)| {
                self.params.skin_strength * value
                    + eye
                        * (-self.params.eyelid_open_offset
                            + self.params.blink_offset * self.params.blink_strength)
                    + lip * self.params.lip_open_offset
            })
            .collect();
        let lower = self.lower.update(&raw, dt);
        let upper = self.upper.update(&raw, dt);
        Ok(self
            .neutral_pose
            .iter()
            .enumerate()
            .map(|(i, &neutral)| {
                let mask = self.face_mask_lower[i / 3];
                neutral
                    + upper[i] * self.params.upper_face_strength * (1.0 - mask)
                    + lower[i] * self.params.lower_face_strength * mask
            })
            .collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TongueAnimatorParams {
    pub tongue_strength: f32,
    pub tongue_height_offset: f32,
    pub tongue_depth_offset: f32,
}

#[derive(Debug, Clone)]
pub struct TongueAnimator {
    params: TongueAnimatorParams,
    neutral_pose: Vec<f32>,
}

impl TongueAnimator {
    pub fn new(params: TongueAnimatorParams, neutral_pose: Vec<f32>) -> Result<Self> {
        if neutral_pose.is_empty() || !neutral_pose.len().is_multiple_of(3) {
            return Err(invalid("tongue neutral pose must contain XYZ vertices"));
        }
        Ok(Self {
            params,
            neutral_pose,
        })
    }

    pub fn neutral_pose(&self) -> &[f32] {
        &self.neutral_pose
    }

    pub fn animate(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.len() != self.neutral_pose.len() {
            return Err(invalid("tongue input length does not match neutral pose"));
        }
        Ok(input
            .iter()
            .zip(&self.neutral_pose)
            .enumerate()
            .map(|(i, (&delta, &neutral))| {
                neutral
                    + delta * self.params.tongue_strength
                    + match i % 3 {
                        1 => self.params.tongue_height_offset,
                        2 => self.params.tongue_depth_offset,
                        _ => 0.0,
                    }
            })
            .collect())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyesAnimatorParams {
    pub eyeballs_strength: f32,
    pub saccade_strength: f32,
    pub right_eyeball_rotation_offset_x: f32,
    pub right_eyeball_rotation_offset_y: f32,
    pub left_eyeball_rotation_offset_x: f32,
    pub left_eyeball_rotation_offset_y: f32,
    pub saccade_seed: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EyesRotation {
    pub right: [f32; 3],
    pub left: [f32; 3],
}

#[derive(Debug, Clone)]
pub struct EyesAnimator {
    params: EyesAnimatorParams,
    saccade_rotation: Vec<f32>,
    frame_index: usize,
    live_time: f32,
}

impl EyesAnimator {
    pub fn new(params: EyesAnimatorParams, saccade_rotation: Vec<f32>) -> Result<Self> {
        if saccade_rotation.is_empty() || !saccade_rotation.len().is_multiple_of(2) {
            return Err(invalid("saccade rotation must contain XY pairs"));
        }
        let mut this = Self {
            params,
            saccade_rotation,
            frame_index: 0,
            live_time: 0.0,
        };
        this.increment_live_time(0.0)?;
        Ok(this)
    }

    fn frame_count(&self) -> usize {
        self.saccade_rotation.len() / 2
    }
    pub fn reset(&mut self) -> Result<()> {
        self.live_time = 0.0;
        self.increment_live_time(0.0)
    }
    pub fn set_frame_index(&mut self, frame: i32) {
        self.frame_index = (self.params.saccade_seed as i64 + i64::from(frame))
            .rem_euclid(self.frame_count() as i64) as usize;
    }
    pub fn increment_live_time(&mut self, dt: f32) -> Result<()> {
        if !dt.is_finite() {
            return Err(invalid("eyes delta time must be finite"));
        }
        let count = self.frame_count() as f32;
        self.live_time = (self.live_time + dt * 30.0).rem_euclid(count);
        self.frame_index = (self.params.saccade_seed + self.live_time).rem_euclid(count) as usize;
        Ok(())
    }
    pub fn compute_rotation(&self, result: [f32; 4]) -> EyesRotation {
        let s = &self.saccade_rotation[self.frame_index * 2..][..2];
        let (sx, sy) = (
            self.params.saccade_strength * s[0],
            self.params.saccade_strength * s[1],
        );
        EyesRotation {
            right: [
                self.params.right_eyeball_rotation_offset_x
                    + self.params.eyeballs_strength * result[0]
                    + sx,
                self.params.right_eyeball_rotation_offset_y
                    + self.params.eyeballs_strength * result[1]
                    + sy,
                0.0,
            ],
            left: [
                self.params.left_eyeball_rotation_offset_x
                    + self.params.eyeballs_strength * result[2]
                    + sx,
                self.params.left_eyeball_rotation_offset_y
                    + self.params.eyeballs_strength * result[3]
                    + sy,
                0.0,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tongue_applies_strength_and_yz_offsets() {
        let a = TongueAnimator::new(
            TongueAnimatorParams {
                tongue_strength: 2.0,
                tongue_height_offset: 10.0,
                tongue_depth_offset: 20.0,
            },
            vec![1.0, 2.0, 3.0],
        )
        .unwrap();
        assert_eq!(a.animate(&[0.5, 1.0, 1.5]).unwrap(), [2.0, 14.0, 26.0]);
    }

    #[test]
    fn skin_composes_and_resets_iir() {
        let p = SkinAnimatorParams {
            lower_face_smoothing: 1.0,
            upper_face_smoothing: 1.0,
            lower_face_strength: 1.0,
            upper_face_strength: 1.0,
            face_mask_level: 0.5,
            face_mask_softness: 0.1,
            skin_strength: 1.0,
            blink_strength: 0.0,
            eyelid_open_offset: 0.0,
            lip_open_offset: 0.0,
            blink_offset: 0.0,
        };
        let mut a =
            SkinAnimator::new(p, vec![0., 0., 0., 0., 1., 0.], vec![0.; 6], vec![0.; 6]).unwrap();
        assert_eq!(a.animate(&[1.; 6], 1.0).unwrap(), [1., 1., 1., 1., 2., 1.]);
        assert!(a.animate(&[0.; 6], 0.01).unwrap()[0] > 0.9);
        a.reset();
        assert_eq!(a.animate(&[0.; 6], 0.01).unwrap(), [0., 0., 0., 0., 1., 0.]);
    }

    #[test]
    fn eyes_wrap_seed_and_share_saccade() {
        let p = EyesAnimatorParams {
            eyeballs_strength: 2.0,
            saccade_strength: 0.5,
            right_eyeball_rotation_offset_x: 1.0,
            right_eyeball_rotation_offset_y: 2.0,
            left_eyeball_rotation_offset_x: 3.0,
            left_eyeball_rotation_offset_y: 4.0,
            saccade_seed: 1.0,
        };
        let mut a = EyesAnimator::new(p, vec![10., 20., 30., 40.]).unwrap();
        assert_eq!(
            a.compute_rotation([1., 2., 3., 4.]),
            EyesRotation {
                right: [18., 26., 0.],
                left: [24., 32., 0.]
            }
        );
        a.set_frame_index(-2);
        assert_eq!(a.compute_rotation([0.; 4]).right, [16., 22., 0.]);
        a.increment_live_time(1.0 / 30.0).unwrap();
        assert_eq!(a.compute_rotation([0.; 4]).right, [6., 12., 0.]);
    }
}
