//! Strict parsers for the SDK's runtime JSON files.
//! NPZ remains a test/model-data adapter and is not parsed by this module.

use crate::common::{Error, Result};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ModelDocument {
    Single(SingleModelDocument),
    Multi(MultiModelDocument),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SingleModelDocument {
    #[serde(rename = "networkInfoPath")]
    pub network_info_path: PathBuf,
    #[serde(rename = "networkPath")]
    pub network_path: PathBuf,
    #[serde(rename = "emotionDatabasePath")]
    pub emotion_database_path: Option<PathBuf>,
    #[serde(rename = "modelConfigPath")]
    pub model_config_path: PathBuf,
    #[serde(rename = "modelDataPath")]
    pub model_data_path: Option<PathBuf>,
    #[serde(rename = "blendshapePaths")]
    pub blendshape_paths: Option<HashMap<String, ModelDataPaths>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MultiModelDocument {
    #[serde(rename = "networkInfoPath")]
    pub network_info_path: PathBuf,
    #[serde(rename = "networkPath")]
    pub network_path: PathBuf,
    #[serde(rename = "modelConfigPaths")]
    pub model_config_paths: Vec<PathBuf>,
    #[serde(rename = "modelDataPaths")]
    pub model_data_paths: Vec<PathBuf>,
    #[serde(rename = "blendshapePaths")]
    pub blendshape_paths: Vec<HashMap<String, ModelDataPaths>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelDataPaths {
    pub config: PathBuf,
    pub data: PathBuf,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ConfigDocument {
    Geometry(GeometryConfigRoot),
    Emotion(EmotionConfigRoot),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GeometryConfigRoot {
    pub config: GeometryConfig,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GeometryConfig {
    pub input_strength: f32,
    pub upper_face_smoothing: f32,
    pub lower_face_smoothing: f32,
    pub upper_face_strength: f32,
    pub lower_face_strength: f32,
    pub face_mask_level: f32,
    pub face_mask_softness: f32,
    #[serde(default)]
    pub source_shot: Option<String>,
    #[serde(default)]
    pub source_frame: Option<i64>,
    pub skin_strength: f32,
    pub blink_strength: f32,
    pub lower_teeth_strength: f32,
    pub lower_teeth_height_offset: f32,
    pub lower_teeth_depth_offset: f32,
    pub lip_open_offset: f32,
    pub tongue_strength: f32,
    pub tongue_height_offset: f32,
    pub tongue_depth_offset: f32,
    pub eyeballs_strength: f32,
    pub saccade_strength: f32,
    pub right_eye_rot_x_offset: f32,
    pub right_eye_rot_y_offset: f32,
    pub left_eye_rot_x_offset: f32,
    pub left_eye_rot_y_offset: f32,
    pub eyelid_open_offset: f32,
    pub eye_saccade_seed: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EmotionConfigRoot {
    pub post_processing_config: EmotionPostProcessingConfig,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EmotionPostProcessingConfig {
    pub output_emotion_length: usize,
    pub emotion_contrast: f32,
    pub live_blend_coef: f32,
    pub preferred_emotion_strength: f32,
    pub fixed_dt: f32,
    pub transition_smoothing: f32,
    pub enable_preferred_emotion: bool,
    pub preferred_emotion: Vec<f32>,
    pub emotion_strength: f32,
    pub max_emotions: usize,
    pub emotion_correspondence: HashMap<String, i64>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum NetworkDocument {
    Geometry(Box<GeometryNetwork>),
    Emotion(EmotionNetwork),
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GeometryNetwork {
    pub id: NetworkId,
    pub params: GeometryParameters,
    pub audio_params: GeometryAudioParameters,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NetworkId {
    #[serde(rename = "type")]
    pub model_type: String,
    pub actor: String,
    pub version: String,
    pub output: String,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum GeometryParameters {
    Regression(RegressionParameters),
    Diffusion(DiffusionParameters),
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RegressionParameters {
    pub implicit_emotion_len: usize,
    pub explicit_emotions: Vec<String>,
    pub default_emotion: Vec<f32>,
    pub num_shapes_skin: usize,
    pub num_shapes_tongue: usize,
    pub num_verts_skin: usize,
    pub num_verts_tongue: usize,
    pub result_jaw_size: usize,
    pub result_eyes_size: usize,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DiffusionParameters {
    pub emotions: Vec<String>,
    pub default_emotion: Vec<f32>,
    pub identities: Vec<String>,
    pub skin_size: usize,
    pub tongue_size: usize,
    pub jaw_size: usize,
    pub eyes_size: usize,
    pub num_diffusion_steps: usize,
    pub num_gru_layers: usize,
    pub gru_latent_dim: usize,
    pub num_frames_left_truncate: usize,
    pub num_frames_right_truncate: usize,
    pub num_frames_center: usize,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum GeometryAudioParameters {
    Regression(RegressionAudioParameters),
    Diffusion(DiffusionAudioParameters),
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RegressionAudioParameters {
    pub buffer_len: usize,
    pub buffer_ofs: usize,
    pub samplerate: usize,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DiffusionAudioParameters {
    pub buffer_len: usize,
    pub padding_left: usize,
    pub padding_right: usize,
    pub samplerate: usize,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EmotionNetwork {
    pub audio_params: EmotionAudioParameters,
    pub emotions: Vec<String>,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EmotionAudioParameters {
    pub samplerate: usize,
}

impl ModelDocument {
    pub fn resolve_paths(&mut self, base: &Path) {
        match self {
            Self::Single(model) => {
                model.network_info_path = resolve(base, &model.network_info_path);
                model.network_path = resolve(base, &model.network_path);
                model.model_config_path = resolve(base, &model.model_config_path);
                model.emotion_database_path = model
                    .emotion_database_path
                    .take()
                    .map(|path| resolve(base, &path));
                model.model_data_path = model
                    .model_data_path
                    .take()
                    .map(|path| resolve(base, &path));
                if let Some(paths) = &mut model.blendshape_paths {
                    resolve_model_data_paths(base, paths.values_mut());
                }
            }
            Self::Multi(model) => {
                model.network_info_path = resolve(base, &model.network_info_path);
                model.network_path = resolve(base, &model.network_path);
                model
                    .model_config_paths
                    .iter_mut()
                    .for_each(|path| *path = resolve(base, path));
                model
                    .model_data_paths
                    .iter_mut()
                    .for_each(|path| *path = resolve(base, path));
                for paths in &mut model.blendshape_paths {
                    resolve_model_data_paths(base, paths.values_mut());
                }
            }
        }
    }

    pub fn validate(&self) -> Result<()> {
        if let Self::Multi(model) = self
            && (model.model_config_paths.is_empty()
                || model.model_config_paths.len() != model.model_data_paths.len()
                || model.model_config_paths.len() != model.blendshape_paths.len())
        {
            return Err(invalid("multi-model path counts do not match"));
        }
        Ok(())
    }
}

impl NetworkDocument {
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Geometry(network) => {
                let expected_type = match &network.params {
                    GeometryParameters::Regression(_) => "regression",
                    GeometryParameters::Diffusion(_) => "diffusion",
                };
                if network.id.model_type != expected_type {
                    return Err(invalid("network id type does not match parameters"));
                }
                let (emotions, defaults) = match &network.params {
                    GeometryParameters::Regression(p) => (&p.explicit_emotions, &p.default_emotion),
                    GeometryParameters::Diffusion(p) => (&p.emotions, &p.default_emotion),
                };
                if emotions.is_empty() || emotions.len() != defaults.len() {
                    return Err(invalid("emotion names/default values mismatch"));
                }
                if emotions.iter().collect::<HashSet<_>>().len() != emotions.len() {
                    return Err(invalid("duplicate emotion name"));
                }
                let samplerate = match network.audio_params {
                    GeometryAudioParameters::Regression(ref p) => p.samplerate,
                    GeometryAudioParameters::Diffusion(ref p) => p.samplerate,
                };
                if samplerate == 0 {
                    return Err(invalid("samplerate must be non-zero"));
                }
                match (&network.params, &network.audio_params) {
                    (
                        GeometryParameters::Regression(parameters),
                        GeometryAudioParameters::Regression(audio),
                    ) => {
                        if audio.buffer_len == 0 || audio.buffer_ofs > audio.buffer_len {
                            return Err(invalid("invalid regression audio window"));
                        }
                        parameters
                            .implicit_emotion_len
                            .checked_add(parameters.explicit_emotions.len())
                            .ok_or_else(|| invalid("emotion dimension overflow"))?;
                    }
                    (
                        GeometryParameters::Diffusion(parameters),
                        GeometryAudioParameters::Diffusion(audio),
                    ) => {
                        if audio.buffer_len == 0 {
                            return Err(invalid("invalid diffusion audio window"));
                        }
                        parameters
                            .num_frames_left_truncate
                            .checked_add(parameters.num_frames_center)
                            .and_then(|v| v.checked_add(parameters.num_frames_right_truncate))
                            .ok_or_else(|| invalid("diffusion frame count overflow"))?;
                    }
                    _ => return Err(invalid("network parameters and audio parameters mismatch")),
                }
            }
            Self::Emotion(network)
                if network.audio_params.samplerate == 0 || network.emotions.is_empty() =>
            {
                return Err(invalid("invalid emotion network"));
            }
            Self::Emotion(_) => {}
        }
        Ok(())
    }
}

impl ConfigDocument {
    pub fn validate(&self) -> Result<()> {
        if let Self::Emotion(root) = self {
            let config = &root.post_processing_config;
            if config.output_emotion_length == 0
                || config.preferred_emotion.len() != config.output_emotion_length
                || config.max_emotions > config.emotion_correspondence.len()
                || config.emotion_correspondence.values().any(|index| {
                    *index < -1
                        || usize::try_from(*index)
                            .is_ok_and(|index| index >= config.output_emotion_length)
                })
            {
                return Err(invalid("invalid emotion post-processing dimensions"));
            }
        }
        Ok(())
    }
}

pub fn parse_model(json: &str) -> Result<ModelDocument> {
    let value: ModelDocument = serde_json::from_str(json).map_err(json_error)?;
    value.validate()?;
    Ok(value)
}
pub fn parse_config(json: &str) -> Result<ConfigDocument> {
    let value: ConfigDocument = serde_json::from_str(json).map_err(json_error)?;
    value.validate()?;
    Ok(value)
}
pub fn parse_network(json: &str) -> Result<NetworkDocument> {
    let value: NetworkDocument = serde_json::from_str(json).map_err(json_error)?;
    value.validate()?;
    Ok(value)
}
pub fn load_model(path: impl AsRef<Path>) -> Result<ModelDocument> {
    let path = path.as_ref();
    let mut value = parse_model(&read(path)?)?;
    value.resolve_paths(path.parent().unwrap_or(Path::new(".")));
    Ok(value)
}
pub fn load_config(path: impl AsRef<Path>) -> Result<ConfigDocument> {
    let path = path.as_ref();
    parse_config(&read(path)?)
}
pub fn load_network(path: impl AsRef<Path>) -> Result<NetworkDocument> {
    let path = path.as_ref();
    parse_network(&read(path)?)
}
fn read(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|e| invalid(&format!("{}: {e}", path.display())))
}
fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}
fn resolve_model_data_paths<'a>(base: &Path, paths: impl Iterator<Item = &'a mut ModelDataPaths>) {
    for path in paths {
        path.config = resolve(base, &path.config);
        path.data = resolve(base, &path.data);
    }
}
fn json_error(error: serde_json::Error) -> Error {
    invalid(&format!("invalid JSON/model schema: {error}"))
}
fn invalid(message: &str) -> Error {
    Error::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_regression_network_schema() {
        let json = r#"{"id":{"type":"regression","actor":"mark","version":"2.3","output":"geometry"},"params":{"implicit_emotion_len":1,"explicit_emotions":["joy"],"default_emotion":[0.0],"num_shapes_skin":1,"num_shapes_tongue":1,"num_verts_skin":1,"num_verts_tongue":1,"result_jaw_size":1,"result_eyes_size":1},"audio_params":{"buffer_len":2,"buffer_ofs":1,"samplerate":16000}}"#;
        assert!(matches!(
            parse_network(json).unwrap(),
            NetworkDocument::Geometry(_)
        ));
    }
    #[test]
    fn strict_rejects_unknown_and_missing() {
        assert!(parse_model(r#"{"networkPath":"n","unknown":1}"#).is_err());
        assert!(parse_model(r#"{"networkPath":"n"}"#).is_err());
    }

    #[test]
    fn validates_relative_paths_counts_shapes_and_overflow() {
        let mut model =
            parse_model(r#"{"networkInfoPath":"n","networkPath":"e","modelConfigPath":"c"}"#)
                .unwrap();
        model.resolve_paths(Path::new("root"));
        let ModelDocument::Single(model) = model else {
            panic!("expected single model")
        };
        assert_eq!(model.network_info_path, Path::new("root").join("n"));

        assert!(parse_model(r#"{"networkInfoPath":"n","networkPath":"e","modelConfigPaths":["c"],"modelDataPaths":[],"blendshapePaths":[]}"#).is_err());
        assert!(parse_config(r#"{"post_processing_config":{"output_emotion_length":2,"emotion_contrast":1.0,"live_blend_coef":0.5,"preferred_emotion_strength":0.5,"fixed_dt":0.1,"transition_smoothing":0.5,"enable_preferred_emotion":false,"preferred_emotion":[0.0],"emotion_strength":1.0,"max_emotions":0,"emotion_correspondence":{}}}"#).is_err());

        let overflow = format!(
            r#"{{"id":{{"type":"regression","actor":"a","version":"1","output":"geometry"}},"params":{{"implicit_emotion_len":{},"explicit_emotions":["joy"],"default_emotion":[0.0],"num_shapes_skin":1,"num_shapes_tongue":1,"num_verts_skin":1,"num_verts_tongue":1,"result_jaw_size":1,"result_eyes_size":1}},"audio_params":{{"buffer_len":2,"buffer_ofs":1,"samplerate":16000}}}}"#,
            usize::MAX
        );
        assert!(parse_network(&overflow).is_err());
    }

    #[test]
    fn parses_installed_sdk_models_when_available() {
        let Some(roots) = std::env::var_os("AUDIO2FACE3D_TEST_MODEL_DIRS") else {
            return;
        };
        for root in std::env::split_paths(&roots) {
            let model = load_model(root.join("model.json")).unwrap();
            match &model {
                ModelDocument::Single(model) => {
                    load_config(&model.model_config_path).unwrap();
                    load_network(&model.network_info_path).unwrap();
                }
                ModelDocument::Multi(model) => {
                    for path in &model.model_config_paths {
                        load_config(path).unwrap();
                    }
                    load_network(&model.network_info_path).unwrap();
                }
            }
        }
    }
}
