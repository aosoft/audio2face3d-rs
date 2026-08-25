use crate::animation::{
    EyesAnimator, EyesRotation, JawParameters, JawTransform, PcaReconstruction, RegressionBackend,
    RegressionFrameInput, RegressionResultLayout, SkinAnimator, TongueAnimator,
};
use crate::common::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct RegressionGeometry {
    pub skin: Vec<f32>,
    pub tongue: Vec<f32>,
    pub jaw_transform: [f32; 16],
    pub eyes_rotation: EyesRotation,
}

pub struct PostprocessedRegressionBackend<B> {
    inference: B,
    tracks: Vec<RegressionPostprocessor>,
    dt: f32,
}

impl<B> PostprocessedRegressionBackend<B> {
    pub fn new(inference: B, tracks: Vec<RegressionPostprocessor>, dt: f32) -> Result<Self> {
        if tracks.is_empty() || !dt.is_finite() || dt <= 0.0 {
            return Err(Error::InvalidSchema(
                "postprocessed backend requires tracks and a positive finite dt".into(),
            ));
        }
        Ok(Self {
            inference,
            tracks,
            dt,
        })
    }
}

impl<B> RegressionBackend for PostprocessedRegressionBackend<B>
where
    B: RegressionBackend<Output = Vec<f32>>,
{
    type Output = RegressionGeometry;

    fn infer(&mut self, track: usize, input: &RegressionFrameInput) -> Result<Self::Output> {
        let result = self.inference.infer(track, input)?;
        self.tracks
            .get_mut(track)
            .ok_or_else(|| {
                Error::InvalidSchema(format!("postprocess track {track} is out of range"))
            })?
            .process(&result, self.dt)
    }

    fn infer_batch(
        &mut self,
        inputs: &[(usize, RegressionFrameInput)],
    ) -> Result<Vec<Self::Output>> {
        let results = self.inference.infer_batch(inputs)?;
        if results.len() != inputs.len() {
            return Err(Error::InvalidSchema("inference batch size mismatch".into()));
        }
        inputs
            .iter()
            .zip(results)
            .map(|((track, _), result)| {
                self.tracks
                    .get_mut(*track)
                    .ok_or_else(|| {
                        Error::InvalidSchema(format!("postprocess track {track} is out of range"))
                    })?
                    .process(&result, self.dt)
            })
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct RegressionPostprocessor {
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
