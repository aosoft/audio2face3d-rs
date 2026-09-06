#![cfg_attr(not(feature = "tensorrt"), allow(dead_code))]

#[cfg(test)]
use crate::animation::SkinAnimatorParams;
use crate::animation::{
    EyesAnimator, EyesRotation, JawParameters, JawTransform, PcaReconstruction,
    RegressionResultLayout, SkinAnimator, TongueAnimator,
};
use crate::common::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct RegressionGeometry {
    pub skin: Vec<f32>,
    pub tongue: Vec<f32>,
    pub jaw_transform: [f32; 16],
    pub eyes_rotation: EyesRotation,
}

/// Component-wise post-processing used by interactive geometry executors.
pub(crate) trait LayeredGeometryPostprocessor {
    fn process_skin(&mut self, inference: &[f32], dt: f32, stateless: bool) -> Result<Vec<f32>>;
    fn process_tongue(&mut self, inference: &[f32]) -> Result<Vec<f32>>;
    fn process_teeth(&mut self, inference: &[f32]) -> Result<[f32; 16]>;
    fn process_eyes(&mut self, inference: &[f32], live_time: f32) -> Result<EyesRotation>;
    fn reset_layers(&mut self) -> Result<()>;

    #[cfg(test)]
    fn skin_parameters(&self) -> SkinAnimatorParams;
    #[cfg(test)]
    fn set_skin_parameters(&mut self, parameters: SkinAnimatorParams) -> Result<()>;
}

#[derive(Debug, Clone)]
pub(crate) struct RegressionPostprocessor {
    skin_pca: PcaReconstruction,
    tongue_pca: PcaReconstruction,
    skin: SkinAnimator,
    tongue: TongueAnimator,
    jaw: JawTransform,
    jaw_parameters: JawParameters,
    eyes: EyesAnimator,
}

impl RegressionPostprocessor {
    pub fn new(
        skin_pca: PcaReconstruction,
        tongue_pca: PcaReconstruction,
        skin: SkinAnimator,
        tongue: TongueAnimator,
        jaw: JawTransform,
        jaw_parameters: JawParameters,
        eyes: EyesAnimator,
    ) -> Self {
        Self {
            skin_pca,
            tongue_pca,
            skin,
            tongue,
            jaw,
            jaw_parameters,
            eyes,
        }
    }

    pub fn layout(&self) -> RegressionResultLayout {
        RegressionResultLayout {
            skin: self.skin_pca.shape_count(),
            tongue: self.tongue_pca.shape_count(),
            jaw: self.jaw.neutral_pose().len(),
            eyes: 4,
        }
    }

    pub fn process(&mut self, network_result: &[f32], dt: f32) -> Result<RegressionGeometry> {
        let slices = self.layout().split(network_result)?;
        let skin_delta = self.skin_pca.reconstruct(slices.skin, 1)?;
        let tongue_delta = self.tongue_pca.reconstruct(slices.tongue, 1)?;
        let eyes: [f32; 4] = slices.eyes.try_into().map_err(|_| {
            Error::InvalidSchema("regression eyes output must contain four values".into())
        })?;
        let geometry = RegressionGeometry {
            skin: self.skin.animate(&skin_delta, dt)?,
            tongue: self.tongue.animate(&tongue_delta)?,
            jaw_transform: self.jaw.compute(slices.jaw, self.jaw_parameters)?,
            eyes_rotation: self.eyes.compute_rotation(eyes),
        };
        self.eyes.increment_live_time(dt)?;
        Ok(geometry)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.skin.reset();
        self.eyes.reset()
    }
}

impl LayeredGeometryPostprocessor for RegressionPostprocessor {
    fn process_skin(&mut self, inference: &[f32], dt: f32, stateless: bool) -> Result<Vec<f32>> {
        let slices = self.layout().split(inference)?;
        let delta = self.skin_pca.reconstruct(slices.skin, 1)?;
        if stateless {
            self.skin.animate_stateless(&delta)
        } else {
            self.skin.animate(&delta, dt)
        }
    }

    fn process_tongue(&mut self, inference: &[f32]) -> Result<Vec<f32>> {
        let slices = self.layout().split(inference)?;
        self.tongue
            .animate(&self.tongue_pca.reconstruct(slices.tongue, 1)?)
    }

    fn process_teeth(&mut self, inference: &[f32]) -> Result<[f32; 16]> {
        let slices = self.layout().split(inference)?;
        self.jaw.compute(slices.jaw, self.jaw_parameters)
    }

    fn process_eyes(&mut self, inference: &[f32], live_time: f32) -> Result<EyesRotation> {
        let slices = self.layout().split(inference)?;
        let eyes = slices.eyes.try_into().map_err(|_| {
            Error::InvalidSchema("regression eyes output must contain four values".into())
        })?;
        self.eyes.set_live_time(live_time)?;
        Ok(self.eyes.compute_rotation(eyes))
    }

    fn reset_layers(&mut self) -> Result<()> {
        self.reset()
    }

    #[cfg(test)]
    fn skin_parameters(&self) -> SkinAnimatorParams {
        self.skin.parameters()
    }

    #[cfg(test)]
    fn set_skin_parameters(&mut self, parameters: SkinAnimatorParams) -> Result<()> {
        self.skin.set_parameters(parameters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{EyesAnimatorParams, SkinAnimatorParams, TongueAnimatorParams};

    #[test]
    fn runs_the_complete_regression_postprocess_order() {
        let skin_pca = PcaReconstruction::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let tongue_pca = PcaReconstruction::new(3, 1, vec![2.0, 3.0, 4.0]).unwrap();
        let skin = SkinAnimator::new(
            SkinAnimatorParams {
                lower_face_smoothing: 0.0,
                upper_face_smoothing: 0.0,
                lower_face_strength: 1.0,
                upper_face_strength: 1.0,
                face_mask_level: 0.5,
                face_mask_softness: 0.1,
                skin_strength: 1.0,
                blink_strength: 0.0,
                eyelid_open_offset: 0.0,
                lip_open_offset: 0.0,
                blink_offset: 0.0,
            },
            vec![0.0, 0.0, 0.0],
            vec![0.0; 3],
            vec![0.0; 3],
        )
        .unwrap();
        let tongue = TongueAnimator::new(
            TongueAnimatorParams {
                tongue_strength: 1.0,
                tongue_height_offset: 0.0,
                tongue_depth_offset: 0.0,
            },
            vec![0.0; 3],
        )
        .unwrap();
        let jaw = JawTransform::new(vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
        let eyes = EyesAnimator::new(
            EyesAnimatorParams {
                eyeballs_strength: 1.0,
                saccade_strength: 0.0,
                right_eyeball_rotation_offset_x: 0.0,
                right_eyeball_rotation_offset_y: 0.0,
                left_eyeball_rotation_offset_x: 0.0,
                left_eyeball_rotation_offset_y: 0.0,
                saccade_seed: 0.0,
            },
            vec![0.0, 0.0],
        )
        .unwrap();
        let mut post = RegressionPostprocessor::new(
            skin_pca,
            tongue_pca,
            skin,
            tongue,
            jaw,
            JawParameters::default(),
            eyes,
        );
        let result = post
            .process(
                &[
                    2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0,
                ],
                1.0 / 30.0,
            )
            .unwrap();
        assert_eq!(result.skin, [2.0, 4.0, 6.0]);
        assert_eq!(result.tongue, [6.0, 9.0, 12.0]);
        assert_eq!(result.eyes_rotation.right, [1.0, 2.0, 0.0]);
        assert_eq!(result.eyes_rotation.left, [3.0, 4.0, 0.0]);
    }
}
