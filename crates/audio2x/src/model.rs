use audio2x_core::{
    Audio2xError, ConfigDocument, EmotionPostProcessingConfig, GeometryAudioParameters,
    GeometryConfig, GeometryParameters, ModelDocument, NetworkDocument, Result, load_config,
    load_model, load_network,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Regression,
    Diffusion,
    Emotion,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ModelParameters {
    Geometry(GeometryConfig),
    Emotion(EmotionPostProcessingConfig),
}

/// Parsed model descriptor with all relative paths resolved against model.json.
#[derive(Clone, Debug)]
pub struct Audio2xModel {
    descriptor_path: PathBuf,
    engine_path: PathBuf,
    network: NetworkDocument,
    parameters: Vec<ModelParameters>,
    kind: ModelKind,
}

impl Audio2xModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let descriptor_path = path.as_ref().to_owned();
        let descriptor = load_model(&descriptor_path)?;
        let (network_path, engine_path, config_paths) = match descriptor {
            ModelDocument::Single(model) => (
                model.network_info_path,
                model.network_path,
                vec![model.model_config_path],
            ),
            ModelDocument::Multi(model) => (
                model.network_info_path,
                model.network_path,
                model.model_config_paths,
            ),
        };
        let network = load_network(network_path)?;
        let kind = match &network {
            NetworkDocument::Geometry(network) => match network.params {
                GeometryParameters::Regression(_) => ModelKind::Regression,
                GeometryParameters::Diffusion(_) => ModelKind::Diffusion,
            },
            NetworkDocument::Emotion(_) => ModelKind::Emotion,
        };
        let mut parameters = Vec::with_capacity(config_paths.len());
        for path in config_paths {
            let value = match load_config(path)? {
                ConfigDocument::Geometry(config) if kind != ModelKind::Emotion => {
                    ModelParameters::Geometry(config.config)
                }
                ConfigDocument::Emotion(config) if kind == ModelKind::Emotion => {
                    ModelParameters::Emotion(config.post_processing_config)
                }
                _ => return Err(invalid("model network and configuration types differ")),
            };
            parameters.push(value);
        }
        if parameters.is_empty() {
            return Err(invalid("model has no configuration"));
        }
        if !engine_path.is_file() {
            return Err(invalid(format!(
                "TensorRT engine is missing: {}; acquire the model explicitly before loading",
                engine_path.display()
            )));
        }
        Ok(Self {
            descriptor_path,
            engine_path,
            network,
            parameters,
            kind,
        })
    }

    pub const fn kind(&self) -> ModelKind {
        self.kind
    }

    pub fn descriptor_path(&self) -> &Path {
        &self.descriptor_path
    }

    pub fn engine_path(&self) -> &Path {
        &self.engine_path
    }

    pub fn network(&self) -> &NetworkDocument {
        &self.network
    }

    pub fn parameters(&self, index: usize) -> Result<&ModelParameters> {
        self.parameters
            .get(index)
            .ok_or_else(|| invalid("model parameter index is out of range"))
    }

    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }

    pub fn set_parameters(&mut self, index: usize, parameters: ModelParameters) -> Result<()> {
        let matches = matches!(
            (self.kind, &parameters),
            (
                ModelKind::Regression | ModelKind::Diffusion,
                ModelParameters::Geometry(_)
            ) | (ModelKind::Emotion, ModelParameters::Emotion(_))
        );
        if !matches {
            return Err(invalid("parameter type does not match model pipeline"));
        }
        validate_parameters(&parameters)?;
        *self
            .parameters
            .get_mut(index)
            .ok_or_else(|| invalid("model parameter index is out of range"))? = parameters;
        Ok(())
    }

    pub fn sample_rate(&self) -> usize {
        match &self.network {
            NetworkDocument::Geometry(network) => match &network.audio_params {
                GeometryAudioParameters::Regression(value) => value.samplerate,
                GeometryAudioParameters::Diffusion(value) => value.samplerate,
            },
            NetworkDocument::Emotion(network) => network.audio_params.samplerate,
        }
    }
}

fn validate_parameters(parameters: &ModelParameters) -> Result<()> {
    let finite = match parameters {
        ModelParameters::Geometry(value) => [
            value.input_strength,
            value.upper_face_smoothing,
            value.lower_face_smoothing,
            value.upper_face_strength,
            value.lower_face_strength,
            value.face_mask_level,
            value.face_mask_softness,
            value.skin_strength,
            value.blink_strength,
            value.lower_teeth_strength,
            value.lower_teeth_height_offset,
            value.lower_teeth_depth_offset,
            value.lip_open_offset,
            value.tongue_strength,
            value.tongue_height_offset,
            value.tongue_depth_offset,
            value.eyeballs_strength,
            value.saccade_strength,
            value.right_eye_rot_x_offset,
            value.right_eye_rot_y_offset,
            value.left_eye_rot_x_offset,
            value.left_eye_rot_y_offset,
            value.eyelid_open_offset,
        ]
        .into_iter()
        .all(f32::is_finite),
        ModelParameters::Emotion(value) => {
            value.output_emotion_length != 0
                && value.preferred_emotion.len() == value.output_emotion_length
                && value.max_emotions <= value.emotion_correspondence.len()
                && value.preferred_emotion.iter().all(|v| v.is_finite())
                && [
                    value.emotion_contrast,
                    value.live_blend_coef,
                    value.preferred_emotion_strength,
                    value.fixed_dt,
                    value.transition_smoothing,
                    value.emotion_strength,
                ]
                .into_iter()
                .all(f32::is_finite)
        }
    };
    if finite {
        Ok(())
    } else {
        Err(invalid("model parameters contain invalid values"))
    }
}

fn invalid(message: impl Into<String>) -> Audio2xError {
    Audio2xError::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_installed_models_and_rejects_cross_pipeline_parameters() {
        let Some(paths) = std::env::var_os("AUDIO2X_TEST_MODEL_DIRS") else {
            return;
        };
        for root in std::env::split_paths(&paths) {
            let mut model = Audio2xModel::load(root.join("model.json")).unwrap();
            assert!(model.engine_path().is_file());
            assert!(model.sample_rate() > 0);
            let wrong = match model.kind() {
                ModelKind::Emotion => ModelParameters::Geometry(match model.parameters(0) {
                    Ok(ModelParameters::Geometry(value)) => value.clone(),
                    _ => continue,
                }),
                _ => ModelParameters::Emotion(EmotionPostProcessingConfig {
                    output_emotion_length: 1,
                    emotion_contrast: 1.0,
                    live_blend_coef: 0.0,
                    preferred_emotion_strength: 0.0,
                    fixed_dt: 0.1,
                    transition_smoothing: 0.1,
                    enable_preferred_emotion: false,
                    preferred_emotion: vec![0.0],
                    emotion_strength: 1.0,
                    max_emotions: 0,
                    emotion_correspondence: Default::default(),
                }),
            };
            assert!(model.set_parameters(0, wrong).is_err());
        }
    }
}
