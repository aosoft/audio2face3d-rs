#![cfg_attr(not(feature = "tensorrt"), allow(dead_code))]

use crate::animation::{
    DiffusionPostprocessor, DiffusionResultLayout, EyesAnimator, EyesAnimatorParams, JawParameters,
    JawTransform, PcaReconstruction, RegressionPostprocessor, SkinAnimator, SkinAnimatorParams,
    TongueAnimator, TongueAnimatorParams,
};
use crate::common::{Error, GeometryConfig, NpzArchive, Result};
use std::path::Path;

/// Geometry post-processing arrays loaded from the original SDK NPZ format.
#[derive(Debug, Clone)]
pub struct GeometryModelData {
    pub skin_neutral_pose: Vec<f32>,
    pub tongue_neutral_pose: Vec<f32>,
    pub jaw_neutral_pose: Vec<f32>,
    pub skin_shapes: Option<Vec<f32>>,
    pub tongue_shapes: Option<Vec<f32>>,
    pub skin_lip_open_delta: Vec<f32>,
    pub skin_eye_close_delta: Vec<f32>,
    pub saccade_rotation: Vec<f32>,
}

impl GeometryModelData {
    pub fn load_regression(path: impl AsRef<Path>) -> Result<Self> {
        let mut archive = NpzArchive::open(path.as_ref())?;
        let data = Self {
            skin_neutral_pose: archive.f32("shapes_mean_skin")?,
            tongue_neutral_pose: archive.f32("shapes_mean_tongue")?,
            jaw_neutral_pose: archive.f32("neutral_jaw")?,
            skin_shapes: Some(archive.f32("shapes_matrix_skin")?),
            tongue_shapes: Some(archive.f32("shapes_matrix_tongue")?),
            skin_lip_open_delta: archive.f32("lip_open_pose_delta")?,
            skin_eye_close_delta: archive.f32("eye_close_pose_delta")?,
            saccade_rotation: archive.f32("saccade_rot_matrix")?,
        };
        data.validate()?;
        Ok(data)
    }

    pub fn load_diffusion(path: impl AsRef<Path>) -> Result<Self> {
        let mut archive = NpzArchive::open(path.as_ref())?;
        let data = Self {
            skin_neutral_pose: archive.f32("neutral_skin")?,
            tongue_neutral_pose: archive.f32("neutral_tongue")?,
            jaw_neutral_pose: archive.f32("neutral_jaw")?,
            skin_shapes: None,
            tongue_shapes: None,
            skin_lip_open_delta: archive.f32("lip_open_pose_delta")?,
            skin_eye_close_delta: archive.f32("eye_close_pose_delta")?,
            saccade_rotation: archive.f32("saccade_rot_matrix")?,
        };
        data.validate()?;
        Ok(data)
    }

    pub(crate) fn regression_postprocessor(
        &self,
        config: &GeometryConfig,
        skin_shape_count: usize,
        tongue_shape_count: usize,
    ) -> Result<RegressionPostprocessor> {
        let skin_shapes = self
            .skin_shapes
            .clone()
            .ok_or_else(|| invalid("regression skin PCA data is unavailable"))?;
        let tongue_shapes = self
            .tongue_shapes
            .clone()
            .ok_or_else(|| invalid("regression tongue PCA data is unavailable"))?;
        Ok(RegressionPostprocessor::new(
            PcaReconstruction::new(self.skin_neutral_pose.len(), skin_shape_count, skin_shapes)?,
            PcaReconstruction::new(
                self.tongue_neutral_pose.len(),
                tongue_shape_count,
                tongue_shapes,
            )?,
            self.skin_animator(config)?,
            self.tongue_animator(config)?,
            JawTransform::new(self.jaw_neutral_pose.clone())?,
            jaw_parameters(config),
            self.eyes_animator(config)?,
        ))
    }

    pub(crate) fn diffusion_postprocessor(
        &self,
        config: &GeometryConfig,
        layout: DiffusionResultLayout,
    ) -> Result<DiffusionPostprocessor> {
        DiffusionPostprocessor::new(
            layout,
            self.skin_animator(config)?,
            self.tongue_animator(config)?,
            JawTransform::new(self.jaw_neutral_pose.clone())?,
            jaw_parameters(config),
            self.eyes_animator(config)?,
        )
    }

    #[cfg(feature = "cuda")]
    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    pub(crate) fn gpu_postprocessor(
        &self,
        device: &std::sync::Arc<crate::cuda::GpuDevice>,
        stream: &crate::cuda::CudaStream,
        config: &GeometryConfig,
        track_count: usize,
        dt: f32,
    ) -> Result<crate::animation::GpuRegressionPostprocessor> {
        let skin = self.skin_animator(config)?;
        let tongue = self.tongue_animator(config)?;
        let eyes = self.eyes_animator(config)?;
        let track = crate::animation::GpuRegressionTrackParams {
            skin: skin.parameters(),
            tongue: tongue.parameters(),
            jaw: jaw_parameters(config),
            eyes: eyes.parameters(),
        };
        crate::animation::GpuRegressionPostprocessor::new(
            device,
            stream,
            crate::animation::GpuRegressionModel {
                skin_neutral_pose: &self.skin_neutral_pose,
                skin_lip_open_delta: &self.skin_lip_open_delta,
                skin_eye_close_delta: &self.skin_eye_close_delta,
                tongue_neutral_pose: &self.tongue_neutral_pose,
                jaw_neutral_pose: &self.jaw_neutral_pose,
                saccade_rotation: &self.saccade_rotation,
            },
            &vec![track; track_count],
            dt,
        )
    }

    #[cfg(feature = "cuda")]
    #[cfg_attr(not(feature = "tensorrt"), allow(dead_code))]
    pub(crate) fn gpu_regression_postprocessor(
        &self,
        device: &std::sync::Arc<crate::cuda::GpuDevice>,
        stream: &crate::cuda::CudaStream,
        config: &GeometryConfig,
        track_count: usize,
        dt: f32,
        raw_layout: crate::animation::RegressionResultLayout,
    ) -> Result<crate::animation::GpuRegressionPcaPostprocessor> {
        let skin = self.skin_animator(config)?;
        let tongue = self.tongue_animator(config)?;
        let eyes = self.eyes_animator(config)?;
        let track = crate::animation::GpuRegressionTrackParams {
            skin: skin.parameters(),
            tongue: tongue.parameters(),
            jaw: jaw_parameters(config),
            eyes: eyes.parameters(),
        };
        crate::animation::GpuRegressionPcaPostprocessor::new(
            device,
            stream,
            crate::animation::GpuRegressionModel {
                skin_neutral_pose: &self.skin_neutral_pose,
                skin_lip_open_delta: &self.skin_lip_open_delta,
                skin_eye_close_delta: &self.skin_eye_close_delta,
                tongue_neutral_pose: &self.tongue_neutral_pose,
                jaw_neutral_pose: &self.jaw_neutral_pose,
                saccade_rotation: &self.saccade_rotation,
            },
            &vec![track; track_count],
            dt,
            self.skin_shapes
                .as_deref()
                .ok_or_else(|| invalid("regression skin PCA data is unavailable"))?,
            self.tongue_shapes
                .as_deref()
                .ok_or_else(|| invalid("regression tongue PCA data is unavailable"))?,
            raw_layout,
        )
    }

    fn skin_animator(&self, config: &GeometryConfig) -> Result<SkinAnimator> {
        SkinAnimator::new(
            SkinAnimatorParams {
                lower_face_smoothing: config.lower_face_smoothing,
                upper_face_smoothing: config.upper_face_smoothing,
                lower_face_strength: config.lower_face_strength,
                upper_face_strength: config.upper_face_strength,
                face_mask_level: config.face_mask_level,
                face_mask_softness: config.face_mask_softness,
                skin_strength: config.skin_strength,
                blink_strength: config.blink_strength,
                eyelid_open_offset: config.eyelid_open_offset,
                lip_open_offset: config.lip_open_offset,
                blink_offset: config.blink_offset,
            },
            self.skin_neutral_pose.clone(),
            self.skin_lip_open_delta.clone(),
            self.skin_eye_close_delta.clone(),
        )
    }

    fn tongue_animator(&self, config: &GeometryConfig) -> Result<TongueAnimator> {
        TongueAnimator::new(
            TongueAnimatorParams {
                tongue_strength: config.tongue_strength,
                tongue_height_offset: config.tongue_height_offset,
                tongue_depth_offset: config.tongue_depth_offset,
            },
            self.tongue_neutral_pose.clone(),
        )
    }

    fn eyes_animator(&self, config: &GeometryConfig) -> Result<EyesAnimator> {
        EyesAnimator::new(
            EyesAnimatorParams {
                eyeballs_strength: config.eyeballs_strength,
                saccade_strength: config.saccade_strength,
                right_eyeball_rotation_offset_x: config.right_eye_rot_x_offset,
                right_eyeball_rotation_offset_y: config.right_eye_rot_y_offset,
                left_eyeball_rotation_offset_x: config.left_eye_rot_x_offset,
                left_eyeball_rotation_offset_y: config.left_eye_rot_y_offset,
                saccade_seed: config.eye_saccade_seed as f32,
            },
            self.saccade_rotation.clone(),
        )
    }

    fn validate(&self) -> Result<()> {
        let skin = self.skin_neutral_pose.len();
        if skin == 0
            || self.tongue_neutral_pose.is_empty()
            || self.jaw_neutral_pose.is_empty()
            || self.skin_lip_open_delta.len() != skin
            || self.skin_eye_close_delta.len() != skin
            || self.saccade_rotation.is_empty()
            || !self.saccade_rotation.len().is_multiple_of(2)
        {
            return Err(invalid("geometry model data dimensions are invalid"));
        }
        Ok(())
    }
}

fn jaw_parameters(config: &GeometryConfig) -> JawParameters {
    JawParameters {
        strength: config.lower_teeth_strength,
        height_offset: config.lower_teeth_height_offset,
        depth_offset: config.lower_teeth_depth_offset,
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}
