//! Model-driven single-track interactive geometry execution.

use crate::animation::{
    BlendshapeData, BlendshapeInvalidationLayer, CpuBlendshapeSolver, DiffusionContract,
    DiffusionPostprocessor, GeometryInvalidationLayer, GeometryModelData,
    InteractiveBlendshapeLayer, InteractiveBlendshapeWeights, InteractiveDiffusionExecutor,
    InteractiveGeometryInterrupt, InteractiveGeometryMetadata, InteractiveGeometryStatus,
    InteractiveRegressionExecutor, RegressionContract, RegressionGeometry, RegressionPostprocessor,
    TensorRtDiffusionBackend, TensorRtRegressionBackend,
};
use crate::common::{
    AudioAccumulator, EmotionAccumulator, Error, GeometryAudioParameters, GeometryParameters,
    ModelDataPaths, NetworkDocument, Result, load_blendshape_config,
};
use crate::cuda::GpuDevice;
use crate::{Model, ModelKind, ModelParameters};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteractivePipelineOptions {
    pub device_ordinal: i32,
    pub frame_rate_numerator: usize,
    pub frame_rate_denominator: usize,
    pub preview_inferences: usize,
    pub diffusion_seed: u64,
}

impl Default for InteractivePipelineOptions {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            frame_rate_numerator: 30,
            frame_rate_denominator: 1,
            preview_inferences: 0,
            diffusion_seed: 0,
        }
    }
}

/// Model-driven Rust counterpart of the original single-track A2F interactive executor.
pub enum InteractiveGeometryExecutorBundle {
    Regression(InteractiveRegressionExecutor<TensorRtRegressionBackend, RegressionPostprocessor>),
    Diffusion(InteractiveDiffusionExecutor<TensorRtDiffusionBackend, DiffusionPostprocessor>),
}

/// Interactive geometry plus CPU Skin/Tongue BlendShape solving.
pub struct InteractiveBlendshapeExecutorBundle {
    geometry: InteractiveGeometryExecutorBundle,
    blendshape: InteractiveBlendshapeLayer,
}

impl InteractiveBlendshapeExecutorBundle {
    pub fn load(model: &Model, options: InteractivePipelineOptions) -> Result<Self> {
        Ok(Self {
            geometry: InteractiveGeometryExecutorBundle::load(model, options)?,
            blendshape: InteractiveBlendshapeLayer::new(
                load_blendshape_component(model, "skin")?,
                load_blendshape_component(model, "tongue")?,
            ),
        })
    }

    pub fn geometry(&self) -> &InteractiveGeometryExecutorBundle {
        &self.geometry
    }

    /// Invalidates cached weights before exposing mutable geometry parameters.
    pub fn geometry_mut(&mut self) -> &mut InteractiveGeometryExecutorBundle {
        self.blendshape.invalidate_geometry();
        &mut self.geometry
    }

    pub fn blendshape(&self) -> &InteractiveBlendshapeLayer {
        &self.blendshape
    }

    pub fn blendshape_mut(&mut self) -> &mut InteractiveBlendshapeLayer {
        &mut self.blendshape
    }

    pub fn invalidate_geometry(&mut self, layer: GeometryInvalidationLayer) {
        self.geometry.invalidate(layer);
        if layer != GeometryInvalidationLayer::None {
            self.blendshape.invalidate_geometry();
        }
    }

    pub fn invalidate_blendshape(&mut self, layer: BlendshapeInvalidationLayer) {
        self.blendshape.invalidate(layer);
    }

    pub fn compute_frame<C>(
        &mut self,
        frame: usize,
        mut callback: C,
    ) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &InteractiveBlendshapeWeights) -> bool,
    {
        let total_frames = self.geometry.total_frames()?;
        let blendshape = &mut self.blendshape;
        let mut solve_error = None;
        let status = self.geometry.compute_frame(frame, |metadata, geometry| {
            match blendshape.compute_frame(frame, total_frames, geometry) {
                Ok(weights) => callback(metadata, &weights),
                Err(error) => {
                    solve_error = Some(error);
                    false
                }
            }
        })?;
        solve_error.map_or(Ok(status), Err)
    }

    pub fn compute_all_frames<C>(&mut self, mut callback: C) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &InteractiveBlendshapeWeights) -> bool,
    {
        self.blendshape
            .begin_all_frames(self.geometry.total_frames()?)?;
        let blendshape = &mut self.blendshape;
        let mut solve_error = None;
        let status = self.geometry.compute_all_frames(|metadata, geometry| {
            match blendshape.compute_next_frame(metadata.frame, geometry) {
                Ok(weights) => callback(metadata, &weights),
                Err(error) => {
                    solve_error = Some(error);
                    false
                }
            }
        })?;
        solve_error.map_or(Ok(status), Err)
    }
}

impl InteractiveGeometryExecutorBundle {
    pub fn load(model: &Model, options: InteractivePipelineOptions) -> Result<Self> {
        if options.frame_rate_numerator == 0 || options.frame_rate_denominator == 0 {
            return Err(invalid("interactive frame rate contains zero"));
        }
        let device = GpuDevice::new(options.device_ordinal)?;
        let ModelParameters::Geometry(config) = model.parameters(0)? else {
            return Err(invalid("interactive geometry config is unavailable"));
        };
        match (model.kind(), model.network()) {
            (ModelKind::Regression, NetworkDocument::Geometry(network)) => {
                let GeometryParameters::Regression(parameters) = &network.params else {
                    return Err(invalid("regression parameters are unavailable"));
                };
                let GeometryAudioParameters::Regression(audio_parameters) = &network.audio_params
                else {
                    return Err(invalid("regression audio parameters are unavailable"));
                };
                let contract = RegressionContract::new(
                    parameters,
                    audio_parameters,
                    options.frame_rate_numerator,
                    options.frame_rate_denominator,
                )?;
                let backend =
                    TensorRtRegressionBackend::load(device, model.engine_path(), contract.clone())?;
                let data = GeometryModelData::load_regression(model.model_data_path(0)?)?;
                let postprocessor = data.regression_postprocessor(
                    config,
                    parameters.num_shapes_skin,
                    parameters.num_shapes_tongue,
                )?;
                let emotions = EmotionAccumulator::new(
                    parameters.explicit_emotions.len(),
                    options.frame_rate_numerator,
                )
                .map_err(accumulator_error)?;
                emotions
                    .accumulate(0, &parameters.default_emotion)
                    .map_err(accumulator_error)?;
                Ok(Self::Regression(InteractiveRegressionExecutor::new(
                    backend,
                    contract,
                    postprocessor,
                    AudioAccumulator::new(audio_parameters.buffer_len, 0)?,
                    emotions,
                    vec![0.0; parameters.implicit_emotion_len],
                    config.input_strength,
                )?))
            }
            (ModelKind::Diffusion, NetworkDocument::Geometry(network)) => {
                let GeometryParameters::Diffusion(parameters) = &network.params else {
                    return Err(invalid("diffusion parameters are unavailable"));
                };
                let GeometryAudioParameters::Diffusion(audio_parameters) = &network.audio_params
                else {
                    return Err(invalid("diffusion audio parameters are unavailable"));
                };
                let contract = DiffusionContract::new(parameters, audio_parameters)?;
                let backend =
                    TensorRtDiffusionBackend::load(device, model.engine_path(), contract.clone())?;
                let data = GeometryModelData::load_diffusion(model.model_data_path(0)?)?;
                let postprocessor = data.diffusion_postprocessor(config, contract.result_layout)?;
                let emotions =
                    EmotionAccumulator::new(parameters.emotions.len(), contract.center_frames)
                        .map_err(accumulator_error)?;
                emotions
                    .accumulate(0, &parameters.default_emotion)
                    .map_err(accumulator_error)?;
                Ok(Self::Diffusion(InteractiveDiffusionExecutor::new(
                    backend,
                    contract,
                    postprocessor,
                    AudioAccumulator::new(audio_parameters.buffer_len, 0)?,
                    emotions,
                    0,
                    config.input_strength,
                    options.preview_inferences,
                    options.diffusion_seed,
                )?))
            }
            _ => Err(invalid(
                "interactive geometry requires a Regression or Diffusion model",
            )),
        }
    }

    pub fn kind(&self) -> ModelKind {
        match self {
            Self::Regression(_) => ModelKind::Regression,
            Self::Diffusion(_) => ModelKind::Diffusion,
        }
    }

    pub fn audio(&self) -> &AudioAccumulator {
        match self {
            Self::Regression(executor) => executor.audio(),
            Self::Diffusion(executor) => executor.audio(),
        }
    }

    pub fn emotions(&self) -> &EmotionAccumulator {
        match self {
            Self::Regression(executor) => executor.emotions(),
            Self::Diffusion(executor) => executor.emotions(),
        }
    }

    pub fn interrupt_handle(&self) -> InteractiveGeometryInterrupt {
        match self {
            Self::Regression(executor) => executor.interrupt_handle(),
            Self::Diffusion(executor) => executor.interrupt_handle(),
        }
    }

    pub fn total_frames(&self) -> Result<usize> {
        match self {
            Self::Regression(executor) => executor.total_frames(),
            Self::Diffusion(executor) => executor.total_frames(),
        }
    }

    pub fn sampling_rate(&self) -> usize {
        match self {
            Self::Regression(executor) => executor.sampling_rate(),
            Self::Diffusion(executor) => executor.sampling_rate(),
        }
    }

    pub fn frame_rate(&self) -> (usize, usize) {
        match self {
            Self::Regression(executor) => executor.frame_rate(),
            Self::Diffusion(executor) => executor.frame_rate(),
        }
    }

    pub fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        match self {
            Self::Regression(executor) => executor.frame_timestamp(frame),
            Self::Diffusion(executor) => executor.frame_timestamp(frame),
        }
    }

    pub fn invalidate(&mut self, layer: GeometryInvalidationLayer) {
        match self {
            Self::Regression(executor) => executor.invalidate(layer),
            Self::Diffusion(executor) => executor.invalidate(layer),
        }
    }

    pub fn is_valid(&self, layer: GeometryInvalidationLayer) -> bool {
        match self {
            Self::Regression(executor) => executor.is_valid(layer),
            Self::Diffusion(executor) => executor.is_valid(layer),
        }
    }

    pub fn compute_frame<C>(
        &mut self,
        frame: usize,
        callback: C,
    ) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &RegressionGeometry) -> bool,
    {
        match self {
            Self::Regression(executor) => executor.compute_frame(frame, callback),
            Self::Diffusion(executor) => executor.compute_frame(frame, callback),
        }
    }

    pub fn compute_all_frames<C>(&mut self, callback: C) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &RegressionGeometry) -> bool,
    {
        match self {
            Self::Regression(executor) => executor.compute_all_frames(callback),
            Self::Diffusion(executor) => executor.compute_all_frames(callback),
        }
    }
}

fn accumulator_error(error: impl std::fmt::Display) -> Error {
    invalid(format!(
        "interactive accumulator initialization failed: {error}"
    ))
}

fn load_blendshape_component(model: &Model, name: &str) -> Result<Option<CpuBlendshapeSolver>> {
    let Some(paths) = component_paths(model, name)? else {
        return Ok(None);
    };
    let data = BlendshapeData::load_npz(&paths.data)?;
    let config = load_blendshape_config(&paths.config)?.blendshape_params;
    let mut solver = CpuBlendshapeSolver::from_config(data, &config)?;
    solver.prepare()?;
    Ok(Some(solver))
}

fn component_paths<'a>(model: &'a Model, name: &str) -> Result<Option<&'a ModelDataPaths>> {
    Ok(model.blendshape_paths(0)?.get(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_installed_geometry_models_interactively_when_configured() {
        let Some(paths) = std::env::var_os("AUDIO2FACE3D_TEST_FACADE_MODELS") else {
            return;
        };
        for root in std::env::split_paths(&paths) {
            let model = Model::load(root.join("model.json")).unwrap();
            if model.kind() == ModelKind::Emotion {
                continue;
            }
            let mut geometry = InteractiveGeometryExecutorBundle::load(
                &model,
                InteractivePipelineOptions::default(),
            )
            .unwrap();
            geometry.audio().accumulate(&vec![0.0; 1_600]).unwrap();
            geometry.audio().close().unwrap();
            geometry.emotions().close().unwrap();
            let total = geometry.total_frames().unwrap();
            assert!(total > 0);
            let status = geometry
                .compute_frame(total / 2, |metadata, frame| {
                    assert_eq!(metadata.frame, total / 2);
                    assert!(!frame.skin.is_empty());
                    assert!(!frame.tongue.is_empty());
                    true
                })
                .unwrap();
            assert!(matches!(
                status,
                InteractiveGeometryStatus::Complete { frames: 1, .. }
            ));
            geometry
                .compute_all_frames(|_, _| true)
                .expect("interactive all-frame execution must complete");
            assert!(geometry.is_valid(GeometryInvalidationLayer::All));

            let mut blendshape = InteractiveBlendshapeExecutorBundle::load(
                &model,
                InteractivePipelineOptions::default(),
            )
            .unwrap();
            blendshape
                .geometry()
                .audio()
                .accumulate(&vec![0.0; 1_600])
                .unwrap();
            blendshape.geometry().audio().close().unwrap();
            blendshape.geometry().emotions().close().unwrap();
            blendshape
                .compute_frame(0, |_, weights| {
                    assert!(!weights.skin.is_empty());
                    assert!(!weights.tongue.is_empty());
                    true
                })
                .unwrap();
        }
    }
}
