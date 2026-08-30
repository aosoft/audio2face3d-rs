use crate::common::{
    AudioAccumulator, ConfigDocument, EmotionAccumulator, EmotionNetwork,
    EmotionPostProcessingConfig, Error, NetworkDocument, Result, load_config, load_network,
};
use crate::emotion::{
    EmotionCallbackMetadata, EmotionExecutionStatus, EmotionPostProcessData,
    EmotionPostProcessParameters, PostProcessEmotionContract, PostProcessEmotionExecutor,
    PostProcessEmotionTrack,
};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Deserialize)]
struct PostProcessModelDocument {
    #[serde(rename = "networkInfoPath")]
    network_info_path: PathBuf,
    #[serde(rename = "modelConfigPath")]
    model_config_path: PathBuf,
}

/// Inference-free Audio2Emotion model metadata.
///
/// Unlike [`crate::Model`], this reader intentionally ignores `networkPath`
/// and never requires a TensorRT engine. The descriptor needs only
/// `networkInfoPath` and `modelConfigPath`, matching the original
/// `ReadPostProcessModelInfo` contract.
#[derive(Clone, Debug)]
pub struct PostProcessEmotionModel {
    descriptor_path: Option<PathBuf>,
    network_info_path: PathBuf,
    config_path: PathBuf,
    network: EmotionNetwork,
    config: EmotionPostProcessingConfig,
}

impl PostProcessEmotionModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let descriptor_path = path.as_ref().to_owned();
        let document: PostProcessModelDocument = serde_json::from_str(
            &fs::read_to_string(&descriptor_path)
                .map_err(|error| invalid(format!("{}: {error}", descriptor_path.display())))?,
        )
        .map_err(|error| invalid(format!("invalid post-process model descriptor: {error}")))?;
        let base = descriptor_path.parent().unwrap_or(Path::new("."));
        let network_info_path = resolve(base, &document.network_info_path);
        let config_path = resolve(base, &document.model_config_path);
        Self::load_resolved(Some(descriptor_path), network_info_path, config_path)
    }

    pub fn load_files(
        network_info_path: impl AsRef<Path>,
        config_path: impl AsRef<Path>,
    ) -> Result<Self> {
        Self::load_resolved(
            None,
            network_info_path.as_ref().to_owned(),
            config_path.as_ref().to_owned(),
        )
    }

    fn load_resolved(
        descriptor_path: Option<PathBuf>,
        network_info_path: PathBuf,
        config_path: PathBuf,
    ) -> Result<Self> {
        let network = match load_network(&network_info_path)? {
            NetworkDocument::Emotion(network) => network,
            NetworkDocument::Geometry(_) => {
                return Err(invalid(
                    "post-process model network info is not Audio2Emotion",
                ));
            }
        };
        let config = match load_config(&config_path)? {
            ConfigDocument::Emotion(config) => config.post_processing_config,
            ConfigDocument::Geometry(_) => {
                return Err(invalid("post-process model config is not Audio2Emotion"));
            }
        };
        EmotionPostProcessData::from_model(&network, &config)?;
        Ok(Self {
            descriptor_path,
            network_info_path,
            config_path,
            network,
            config,
        })
    }

    pub fn descriptor_path(&self) -> Option<&Path> {
        self.descriptor_path.as_deref()
    }

    pub fn network_info_path(&self) -> &Path {
        &self.network_info_path
    }

    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    pub fn network(&self) -> &EmotionNetwork {
        &self.network
    }

    pub fn config(&self) -> &EmotionPostProcessingConfig {
        &self.config
    }

    pub fn sample_rate(&self) -> usize {
        self.network.audio_params.samplerate
    }

    pub fn post_process(&self) -> Result<(EmotionPostProcessData, EmotionPostProcessParameters)> {
        EmotionPostProcessData::from_model(&self.network, &self.config)
    }

    pub fn contract(
        &self,
        frame_rate_numerator: usize,
        frame_rate_denominator: usize,
    ) -> Result<PostProcessEmotionContract> {
        PostProcessEmotionContract::new(
            self.sample_rate(),
            frame_rate_numerator,
            frame_rate_denominator,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostProcessEmotionBundleOptions {
    pub track_count: usize,
    pub frame_rate_numerator: usize,
    pub frame_rate_denominator: usize,
}

impl PostProcessEmotionBundleOptions {
    pub const fn new(
        track_count: usize,
        frame_rate_numerator: usize,
        frame_rate_denominator: usize,
    ) -> Self {
        Self {
            track_count,
            frame_rate_numerator,
            frame_rate_denominator,
        }
    }
}

/// Owns a post-process model, its accumulators, and the streaming executor.
///
/// No CUDA stream is required because this Rust implementation reuses the
/// host post-processor. The ownership and accumulator access pattern otherwise
/// mirrors the original post-process executor bundle.
pub struct PostProcessEmotionExecutorBundle {
    model: PostProcessEmotionModel,
    executor: PostProcessEmotionExecutor,
    audio_accumulators: Vec<AudioAccumulator>,
    preferred_emotion_accumulators: Vec<EmotionAccumulator>,
}

impl PostProcessEmotionExecutorBundle {
    pub fn load(
        descriptor: impl AsRef<Path>,
        options: PostProcessEmotionBundleOptions,
    ) -> Result<Self> {
        let model = PostProcessEmotionModel::load(descriptor)?;
        Self::from_model(model, options)
    }

    pub fn from_model(
        model: PostProcessEmotionModel,
        options: PostProcessEmotionBundleOptions,
    ) -> Result<Self> {
        if options.track_count == 0 {
            return Err(invalid("post-process bundle track count must be non-zero"));
        }
        let contract =
            model.contract(options.frame_rate_numerator, options.frame_rate_denominator)?;
        let (data, parameters) = model.post_process()?;
        let output_length = data.output_emotion_length;
        let executor =
            PostProcessEmotionExecutor::new(contract, data, parameters, options.track_count)?;
        let audio_accumulators = (0..options.track_count)
            .map(|_| AudioAccumulator::new(model.sample_rate(), 0))
            .collect::<Result<Vec<_>>>()?;
        let preferred_emotion_accumulators = (0..options.track_count)
            .map(|_| {
                EmotionAccumulator::new(output_length, 300).map_err(|error| {
                    invalid(format!(
                        "preferred emotion accumulator creation failed: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            model,
            executor,
            audio_accumulators,
            preferred_emotion_accumulators,
        })
    }

    pub fn model(&self) -> &PostProcessEmotionModel {
        &self.model
    }

    pub fn executor(&self) -> &PostProcessEmotionExecutor {
        &self.executor
    }

    pub fn executor_mut(&mut self) -> &mut PostProcessEmotionExecutor {
        &mut self.executor
    }

    pub fn audio_accumulator(&self, track: usize) -> Result<&AudioAccumulator> {
        self.audio_accumulators
            .get(track)
            .ok_or_else(|| invalid("post-process audio accumulator track is out of range"))
    }

    pub fn preferred_emotion_accumulator(&self, track: usize) -> Result<&EmotionAccumulator> {
        self.preferred_emotion_accumulators
            .get(track)
            .ok_or_else(|| invalid("preferred emotion accumulator track is out of range"))
    }

    pub fn execute<C>(&mut self, callback: C) -> Result<EmotionExecutionStatus>
    where
        C: FnMut(EmotionCallbackMetadata, &[f32]) -> bool,
    {
        let tracks = self
            .audio_accumulators
            .iter()
            .zip(&self.preferred_emotion_accumulators)
            .map(|(audio, preferred)| PostProcessEmotionTrack {
                audio,
                preferred_emotions: Some(preferred),
            })
            .collect::<Vec<_>>();
        self.executor.execute(&tracks, callback)
    }
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);

    struct Fixture {
        root: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "audio2face3d-postprocess-model-{}-{id}",
                std::process::id()
            ));
            fs::create_dir_all(&root).unwrap();
            fs::write(
                root.join("model.json"),
                r#"{"networkInfoPath":"network_info.json","networkPath":"missing.plan","modelConfigPath":"config.json"}"#,
            )
            .unwrap();
            fs::write(
                root.join("network_info.json"),
                r#"{"audio_params":{"samplerate":16000},"emotions":["joy","neutral"]}"#,
            )
            .unwrap();
            fs::write(
                root.join("config.json"),
                r#"{"post_processing_config":{"output_emotion_length":2,"emotion_contrast":1.0,"live_blend_coef":0.0,"preferred_emotion_strength":0.5,"fixed_dt":9.0,"transition_smoothing":0.0,"enable_preferred_emotion":false,"preferred_emotion":[0.0,0.0],"emotion_strength":1.0,"max_emotions":2,"emotion_correspondence":{"joy":0,"neutral":1}}}"#,
            )
            .unwrap();
            Self { root }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn model_reader_requires_no_engine_and_resolves_relative_paths() {
        let fixture = Fixture::new();
        let model = PostProcessEmotionModel::load(fixture.root.join("model.json")).unwrap();
        assert_eq!(model.sample_rate(), 16_000);
        assert_eq!(model.network().emotions, ["joy", "neutral"]);
        assert_eq!(
            model.network_info_path(),
            fixture.root.join("network_info.json")
        );
        assert_eq!(model.config_path(), fixture.root.join("config.json"));
        assert_eq!(
            model.post_process().unwrap().0.emotion_correspondence,
            [0, 1]
        );
        assert!(!fixture.root.join("missing.plan").exists());
        let from_files = PostProcessEmotionModel::load_files(
            fixture.root.join("network_info.json"),
            fixture.root.join("config.json"),
        )
        .unwrap();
        assert_eq!(from_files.sample_rate(), model.sample_rate());
    }

    #[test]
    fn bundle_owns_model_accumulators_and_executor() {
        let fixture = Fixture::new();
        let mut bundle = PostProcessEmotionExecutorBundle::load(
            fixture.root.join("model.json"),
            PostProcessEmotionBundleOptions::new(2, 30, 1),
        )
        .unwrap();
        assert!(
            PostProcessEmotionExecutorBundle::load(
                fixture.root.join("model.json"),
                PostProcessEmotionBundleOptions::new(0, 30, 1),
            )
            .is_err()
        );
        assert!(bundle.audio_accumulator(2).is_err());
        assert!(bundle.preferred_emotion_accumulator(2).is_err());
        assert_eq!(bundle.executor().track_count(), 2);
        assert_eq!(
            bundle
                .preferred_emotion_accumulator(0)
                .unwrap()
                .state()
                .emotion_size,
            2
        );
        for track in 0..2 {
            bundle
                .audio_accumulator(track)
                .unwrap()
                .accumulate(&[0.0; 534])
                .unwrap();
            bundle.audio_accumulator(track).unwrap().close().unwrap();
        }
        let mut callbacks = 0;
        while !matches!(
            bundle
                .execute(|_, _| {
                    callbacks += 1;
                    true
                })
                .unwrap(),
            EmotionExecutionStatus::Complete
        ) {}
        assert_eq!(callbacks, 4);
    }
}
