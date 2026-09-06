//! SDK-facing host animator components and factories.

use crate::animation::{
    EyesAnimator, EyesAnimatorParams, EyesRotation, JawParameters, JawTransform, PcaReconstruction,
    SkinAnimator, SkinAnimatorParams, TongueAnimator, TongueAnimatorParams,
};
use crate::audio2face::{
    AnimatorEyesParams, AnimatorSkinParams, AnimatorTeethParams, AnimatorTongueParams,
};
use crate::audio2x::Result;

/// Host PCA reconstruction component corresponding to `nva2f::IAnimatorPcaReconstruction`.
#[derive(Debug, Clone)]
pub struct AnimatorPcaReconstruction {
    inner: PcaReconstruction,
}

impl AnimatorPcaReconstruction {
    pub fn shape_size(&self) -> usize {
        self.inner.shape_size()
    }

    pub fn shape_count(&self) -> usize {
        self.inner.shape_count()
    }

    pub fn reconstruct(&self, coefficients: &[f32], batch_size: usize) -> Result<Vec<f32>> {
        self.inner.reconstruct(coefficients, batch_size)
    }
}

/// Host skin animator corresponding to `nva2f::IAnimatorSkin`.
#[derive(Debug, Clone)]
pub struct AnimatorSkin {
    inner: SkinAnimator,
}

impl AnimatorSkin {
    pub fn neutral_pose(&self) -> &[f32] {
        self.inner.neutral_pose()
    }

    pub fn parameters(&self) -> AnimatorSkinParams {
        self.inner.parameters().into()
    }

    pub fn set_parameters(&mut self, parameters: AnimatorSkinParams) -> Result<()> {
        self.inner.set_parameters(parameters.into())
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn animate(&mut self, input: &[f32], dt: f32) -> Result<Vec<f32>> {
        self.inner.animate(input, dt)
    }

    pub fn animate_stateless(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.inner.animate_stateless(input)
    }
}

/// Host tongue animator corresponding to `nva2f::IAnimatorTongue`.
#[derive(Debug, Clone)]
pub struct AnimatorTongue {
    inner: TongueAnimator,
}

impl AnimatorTongue {
    pub fn neutral_pose(&self) -> &[f32] {
        self.inner.neutral_pose()
    }

    pub fn parameters(&self) -> AnimatorTongueParams {
        self.inner.parameters().into()
    }

    pub fn set_parameters(&mut self, parameters: AnimatorTongueParams) -> Result<()> {
        self.inner.set_parameters(parameters.into())
    }

    pub fn animate(&self, input: &[f32]) -> Result<Vec<f32>> {
        self.inner.animate(input)
    }
}

/// Host teeth animator corresponding to `nva2f::IAnimatorTeeth`.
///
/// The animator owns its tunable parameters. `JawTransform` remains the lower-level
/// rigid-transform helper and is not used as an SDK-facing type alias.
#[derive(Debug, Clone)]
pub struct AnimatorTeeth {
    transform: JawTransform,
    parameters: AnimatorTeethParams,
}

impl AnimatorTeeth {
    pub fn neutral_pose(&self) -> &[f32] {
        self.transform.neutral_pose()
    }

    pub fn parameters(&self) -> AnimatorTeethParams {
        self.parameters
    }

    pub fn set_parameters(&mut self, parameters: AnimatorTeethParams) -> Result<()> {
        JawParameters::from(parameters).validate()?;
        self.parameters = parameters;
        Ok(())
    }

    pub fn compute(&self, deltas: &[f32]) -> Result<[f32; 16]> {
        self.transform.compute(deltas, self.parameters.into())
    }
}

/// Host eyes animator corresponding to `nva2f::IAnimatorEyes`.
#[derive(Debug, Clone)]
pub struct AnimatorEyes {
    inner: EyesAnimator,
}

impl AnimatorEyes {
    pub fn parameters(&self) -> AnimatorEyesParams {
        self.inner.parameters().into()
    }

    pub fn set_parameters(&mut self, parameters: AnimatorEyesParams) -> Result<()> {
        self.inner.set_parameters(parameters.into())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.inner.reset()
    }

    pub fn set_frame_index(&mut self, frame: i32) {
        self.inner.set_frame_index(frame);
    }

    pub fn set_live_time(&mut self, time: f32) -> Result<()> {
        self.inner.set_live_time(time)
    }

    pub fn increment_live_time(&mut self, dt: f32) -> Result<()> {
        self.inner.increment_live_time(dt)
    }

    pub fn compute_rotation(&self, result: [f32; 4]) -> EyesRotation {
        self.inner.compute_rotation(result)
    }
}

pub fn create_animator_pca_reconstruction(
    shape_size: usize,
    shape_count: usize,
    shapes: Vec<f32>,
) -> Result<AnimatorPcaReconstruction> {
    Ok(AnimatorPcaReconstruction {
        inner: PcaReconstruction::new(shape_size, shape_count, shapes)?,
    })
}

pub fn create_animator_skin(
    parameters: AnimatorSkinParams,
    neutral_pose: Vec<f32>,
    lip_open_pose_delta: Vec<f32>,
    eye_close_pose_delta: Vec<f32>,
) -> Result<AnimatorSkin> {
    Ok(AnimatorSkin {
        inner: SkinAnimator::new(
            parameters.into(),
            neutral_pose,
            lip_open_pose_delta,
            eye_close_pose_delta,
        )?,
    })
}

pub fn create_animator_tongue(
    parameters: AnimatorTongueParams,
    neutral_pose: Vec<f32>,
) -> Result<AnimatorTongue> {
    Ok(AnimatorTongue {
        inner: TongueAnimator::new(parameters.into(), neutral_pose)?,
    })
}

pub fn create_animator_teeth(
    parameters: AnimatorTeethParams,
    neutral_pose: Vec<f32>,
) -> Result<AnimatorTeeth> {
    JawParameters::from(parameters).validate()?;
    Ok(AnimatorTeeth {
        transform: JawTransform::new(neutral_pose)?,
        parameters,
    })
}

pub fn create_animator_eyes(
    parameters: AnimatorEyesParams,
    saccade_rotation: Vec<f32>,
) -> Result<AnimatorEyes> {
    Ok(AnimatorEyes {
        inner: EyesAnimator::new(parameters.into(), saccade_rotation)?,
    })
}

impl From<AnimatorSkinParams> for SkinAnimatorParams {
    fn from(value: AnimatorSkinParams) -> Self {
        Self {
            lower_face_smoothing: value.lower_face_smoothing,
            upper_face_smoothing: value.upper_face_smoothing,
            lower_face_strength: value.lower_face_strength,
            upper_face_strength: value.upper_face_strength,
            face_mask_level: value.face_mask_level,
            face_mask_softness: value.face_mask_softness,
            skin_strength: value.skin_strength,
            blink_strength: value.blink_strength,
            eyelid_open_offset: value.eyelid_open_offset,
            lip_open_offset: value.lip_open_offset,
            blink_offset: value.blink_offset,
        }
    }
}

impl From<SkinAnimatorParams> for AnimatorSkinParams {
    fn from(value: SkinAnimatorParams) -> Self {
        Self {
            lower_face_smoothing: value.lower_face_smoothing,
            upper_face_smoothing: value.upper_face_smoothing,
            lower_face_strength: value.lower_face_strength,
            upper_face_strength: value.upper_face_strength,
            face_mask_level: value.face_mask_level,
            face_mask_softness: value.face_mask_softness,
            skin_strength: value.skin_strength,
            blink_strength: value.blink_strength,
            eyelid_open_offset: value.eyelid_open_offset,
            lip_open_offset: value.lip_open_offset,
            blink_offset: value.blink_offset,
        }
    }
}

impl From<AnimatorTongueParams> for TongueAnimatorParams {
    fn from(value: AnimatorTongueParams) -> Self {
        Self {
            tongue_strength: value.tongue_strength,
            tongue_height_offset: value.tongue_height_offset,
            tongue_depth_offset: value.tongue_depth_offset,
        }
    }
}

impl From<TongueAnimatorParams> for AnimatorTongueParams {
    fn from(value: TongueAnimatorParams) -> Self {
        Self {
            tongue_strength: value.tongue_strength,
            tongue_height_offset: value.tongue_height_offset,
            tongue_depth_offset: value.tongue_depth_offset,
        }
    }
}

impl From<AnimatorTeethParams> for JawParameters {
    fn from(value: AnimatorTeethParams) -> Self {
        Self {
            strength: value.lower_teeth_strength,
            height_offset: value.lower_teeth_height_offset,
            depth_offset: value.lower_teeth_depth_offset,
        }
    }
}

impl From<AnimatorEyesParams> for EyesAnimatorParams {
    fn from(value: AnimatorEyesParams) -> Self {
        Self {
            eyeballs_strength: value.eyeballs_strength,
            saccade_strength: value.saccade_strength,
            right_eyeball_rotation_offset_x: value.right_eyeball_rotation_offset_x,
            right_eyeball_rotation_offset_y: value.right_eyeball_rotation_offset_y,
            left_eyeball_rotation_offset_x: value.left_eyeball_rotation_offset_x,
            left_eyeball_rotation_offset_y: value.left_eyeball_rotation_offset_y,
            saccade_seed: value.saccade_seed,
        }
    }
}

impl From<EyesAnimatorParams> for AnimatorEyesParams {
    fn from(value: EyesAnimatorParams) -> Self {
        Self {
            eyeballs_strength: value.eyeballs_strength,
            saccade_strength: value.saccade_strength,
            right_eyeball_rotation_offset_x: value.right_eyeball_rotation_offset_x,
            right_eyeball_rotation_offset_y: value.right_eyeball_rotation_offset_y,
            left_eyeball_rotation_offset_x: value.left_eyeball_rotation_offset_x,
            left_eyeball_rotation_offset_y: value.left_eyeball_rotation_offset_y,
            saccade_seed: value.saccade_seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn teeth_animator_owns_parameters_separately_from_math_helper() {
        let parameters = AnimatorTeethParams {
            lower_teeth_strength: 1.0,
            lower_teeth_height_offset: 0.0,
            lower_teeth_depth_offset: 0.0,
        };
        let animator = create_animator_teeth(
            parameters,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        assert_eq!(animator.parameters(), parameters);
        assert_eq!(animator.neutral_pose().len(), 9);
    }
}
