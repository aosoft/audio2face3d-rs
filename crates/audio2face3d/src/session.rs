use crate::animation::{
    DiffusionContract, DiffusionExecutionStatus, DiffusionExecutor, DiffusionTrack, PumpStatus,
    RegressionContract, RegressionExecutor, RegressionTrack, TensorRtDiffusionBackend,
    TensorRtRegressionBackend,
};
use crate::common::{
    AudioAccumulator, EmotionAccumulator, Error, GeometryAudioParameters, GeometryParameters,
    NetworkDocument, Result,
};
use crate::cuda::GpuDevice;
use crate::emotion::{
    ClassifierContract, EmotionExecutionStatus, EmotionExecutor, EmotionPostProcessData,
    EmotionPostProcessParameters, EmotionTrack, TensorRtClassifierBackend,
};
use crate::{Model, ModelKind, ModelParameters};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineOptions {
    pub device_ordinal: i32,
    pub track_count: usize,
    pub frame_rate_numerator: usize,
    pub frame_rate_denominator: usize,
    pub emotion_buffer_length: usize,
    pub inferences_to_skip: usize,
    pub diffusion_seed: u64,
}

impl Default for PipelineOptions {
    fn default() -> Self {
        Self {
            device_ordinal: 0,
            track_count: 1,
            frame_rate_numerator: 30,
            frame_rate_denominator: 1,
            emotion_buffer_length: 60_000,
            inferences_to_skip: 0,
            diffusion_seed: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TrackParameters {
    Regression {
        input_strength: f32,
        implicit_emotion: Vec<f32>,
    },
    Diffusion {
        input_strength: f32,
        identity_index: usize,
    },
    Emotion {
        input_strength: f32,
        post_process: EmotionPostProcessParameters,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CallbackMetadata {
    pub kind: ModelKind,
    pub track: usize,
    pub inference: Option<usize>,
    pub frame: usize,
    pub timestamp: i64,
    pub next_timestamp: i64,
}

#[derive(Clone, Copy, Debug)]
pub enum PipelineOutput<'a> {
    Geometry(&'a [f32]),
    Emotion(&'a [f32]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineStatus {
    AwaitingInput,
    Executed { tracks: usize, frames: usize },
    Complete,
    Interrupted,
}

struct RegressionOwnedTrack {
    audio: AudioAccumulator,
    emotions: EmotionAccumulator,
    implicit_emotion: Vec<f32>,
    input_strength: f32,
}

struct DiffusionOwnedTrack {
    audio: AudioAccumulator,
    emotions: EmotionAccumulator,
    identity_index: usize,
    input_strength: f32,
}

struct EmotionOwnedTrack {
    audio: AudioAccumulator,
    input_strength: f32,
}

pub struct TensorRtPipeline {
    inner: PipelineInner,
}

enum PipelineInner {
    Regression {
        executor: Box<RegressionExecutor>,
        backend: TensorRtRegressionBackend,
        tracks: Vec<RegressionOwnedTrack>,
    },
    Diffusion {
        executor: Box<DiffusionExecutor>,
        backend: TensorRtDiffusionBackend,
        tracks: Vec<DiffusionOwnedTrack>,
    },
    Emotion {
        executor: Box<EmotionExecutor>,
        backend: TensorRtClassifierBackend,
        tracks: Vec<EmotionOwnedTrack>,
    },
}

impl TensorRtPipeline {
    pub fn load(model: &Model, options: PipelineOptions) -> Result<Self> {
        if options.track_count == 0 || options.frame_rate_denominator == 0 {
            return Err(invalid("pipeline options contain zero dimensions"));
        }
        let device = GpuDevice::new(options.device_ordinal)?;
        match (model.kind(), model.network()) {
            (ModelKind::Regression, NetworkDocument::Geometry(network)) => {
                let GeometryParameters::Regression(parameters) = &network.params else {
                    return Err(invalid("regression model parameters are unavailable"));
                };
                let GeometryAudioParameters::Regression(audio) = &network.audio_params else {
                    return Err(invalid("regression audio parameters are unavailable"));
                };
                let contract = RegressionContract::new(
                    parameters,
                    audio,
                    options.frame_rate_numerator,
                    options.frame_rate_denominator,
                )?;
                let backend =
                    TensorRtRegressionBackend::load(device, model.engine_path(), contract.clone())?;
                let input_strength = geometry_input_strength(model, 0)?;
                let tracks = (0..options.track_count)
                    .map(|_| {
                        let emotions = EmotionAccumulator::new(
                            parameters.explicit_emotions.len(),
                            options.frame_rate_numerator,
                        )
                        .map_err(accumulator_error)?;
                        emotions
                            .accumulate(0, &parameters.default_emotion)
                            .map_err(accumulator_error)?;
                        emotions.close().map_err(accumulator_error)?;
                        Ok(RegressionOwnedTrack {
                            audio: AudioAccumulator::new(audio.buffer_len, 0)?,
                            emotions,
                            implicit_emotion: vec![0.0; parameters.implicit_emotion_len],
                            input_strength,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self {
                    inner: PipelineInner::Regression {
                        executor: Box::new(RegressionExecutor::new(contract, options.track_count)?),
                        backend,
                        tracks,
                    },
                })
            }
            (ModelKind::Diffusion, NetworkDocument::Geometry(network)) => {
                let GeometryParameters::Diffusion(parameters) = &network.params else {
                    return Err(invalid("diffusion model parameters are unavailable"));
                };
                let GeometryAudioParameters::Diffusion(audio) = &network.audio_params else {
                    return Err(invalid("diffusion audio parameters are unavailable"));
                };
                let contract = DiffusionContract::new(parameters, audio)?;
                let backend =
                    TensorRtDiffusionBackend::load(device, model.engine_path(), contract.clone())?;
                let tracks = (0..options.track_count)
                    .map(|index| {
                        let parameter_index = index.min(model.parameter_count() - 1);
                        let emotions = EmotionAccumulator::new(
                            parameters.emotions.len(),
                            contract.center_frames,
                        )
                        .map_err(accumulator_error)?;
                        emotions
                            .accumulate(0, &parameters.default_emotion)
                            .map_err(accumulator_error)?;
                        emotions.close().map_err(accumulator_error)?;
                        Ok(DiffusionOwnedTrack {
                            audio: AudioAccumulator::new(audio.buffer_len, 0)?,
                            emotions,
                            identity_index: index % parameters.identities.len(),
                            input_strength: geometry_input_strength(model, parameter_index)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self {
                    inner: PipelineInner::Diffusion {
                        executor: Box::new(DiffusionExecutor::new(
                            contract,
                            options.track_count,
                            options.diffusion_seed,
                        )?),
                        backend,
                        tracks,
                    },
                })
            }
            (ModelKind::Emotion, NetworkDocument::Emotion(network)) => {
                let ModelParameters::Emotion(config) = model.parameters(0)? else {
                    return Err(invalid("emotion post-process parameters are unavailable"));
                };
                let (data, parameters) = EmotionPostProcessData::from_model(network, config)?;
                let contract = ClassifierContract::new(
                    options.emotion_buffer_length,
                    network.audio_params.samplerate,
                    network.emotions.len(),
                    options.frame_rate_numerator,
                    options.frame_rate_denominator,
                    options.inferences_to_skip,
                )?;
                let backend =
                    TensorRtClassifierBackend::load(device, model.engine_path(), contract.clone())?;
                backend.validate_track_count(options.track_count)?;
                let tracks = (0..options.track_count)
                    .map(|_| {
                        Ok(EmotionOwnedTrack {
                            audio: AudioAccumulator::new(network.audio_params.samplerate, 0)?,
                            input_strength: 1.0,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Self {
                    inner: PipelineInner::Emotion {
                        executor: Box::new(EmotionExecutor::new(
                            contract,
                            data,
                            parameters,
                            options.track_count,
                        )?),
                        backend,
                        tracks,
                    },
                })
            }
            _ => Err(invalid("model kind and network document differ")),
        }
    }

    pub fn kind(&self) -> ModelKind {
        match self.inner {
            PipelineInner::Regression { .. } => ModelKind::Regression,
            PipelineInner::Diffusion { .. } => ModelKind::Diffusion,
            PipelineInner::Emotion { .. } => ModelKind::Emotion,
        }
    }

    pub fn track_count(&self) -> usize {
        match &self.inner {
            PipelineInner::Regression { tracks, .. } => tracks.len(),
            PipelineInner::Diffusion { tracks, .. } => tracks.len(),
            PipelineInner::Emotion { tracks, .. } => tracks.len(),
        }
    }

    pub fn accumulate_audio(&self, track: usize, samples: &[f32]) -> Result<()> {
        match &self.inner {
            PipelineInner::Regression { tracks, .. } => {
                regression_track(tracks, track)?.audio.accumulate(samples)
            }
            PipelineInner::Diffusion { tracks, .. } => {
                diffusion_track(tracks, track)?.audio.accumulate(samples)
            }
            PipelineInner::Emotion { tracks, .. } => {
                emotion_track(tracks, track)?.audio.accumulate(samples)
            }
        }
    }

    pub fn close_audio(&self, track: usize) -> Result<()> {
        match &self.inner {
            PipelineInner::Regression { tracks, .. } => {
                regression_track(tracks, track)?.audio.close()
            }
            PipelineInner::Diffusion { tracks, .. } => {
                diffusion_track(tracks, track)?.audio.close()
            }
            PipelineInner::Emotion { tracks, .. } => emotion_track(tracks, track)?.audio.close(),
        }
    }

    pub fn set_track_parameters(&mut self, track: usize, value: TrackParameters) -> Result<()> {
        match (&mut self.inner, value) {
            (
                PipelineInner::Regression { tracks, .. },
                TrackParameters::Regression {
                    input_strength,
                    implicit_emotion,
                },
            ) => {
                finite_strength(input_strength)?;
                let track = regression_track_mut(tracks, track)?;
                if implicit_emotion.len() != track.implicit_emotion.len()
                    || implicit_emotion.iter().any(|value| !value.is_finite())
                {
                    return Err(invalid(
                        "regression implicit emotion dimensions are invalid",
                    ));
                }
                track.input_strength = input_strength;
                track.implicit_emotion = implicit_emotion;
                Ok(())
            }
            (
                PipelineInner::Diffusion { tracks, .. },
                TrackParameters::Diffusion {
                    input_strength,
                    identity_index,
                },
            ) => {
                finite_strength(input_strength)?;
                let track = diffusion_track_mut(tracks, track)?;
                track.input_strength = input_strength;
                track.identity_index = identity_index;
                Ok(())
            }
            (
                PipelineInner::Emotion {
                    executor, tracks, ..
                },
                TrackParameters::Emotion {
                    input_strength,
                    post_process,
                },
            ) => {
                finite_strength(input_strength)?;
                executor.set_parameters(track, post_process)?;
                emotion_track_mut(tracks, track)?.input_strength = input_strength;
                Ok(())
            }
            _ => Err(invalid("track parameters do not match pipeline")),
        }
    }

    pub fn execute<C>(&mut self, mut callback: C) -> Result<PipelineStatus>
    where
        C: for<'output> FnMut(CallbackMetadata, PipelineOutput<'output>) -> bool,
    {
        match &mut self.inner {
            PipelineInner::Regression {
                executor,
                backend,
                tracks,
            } => {
                let views = tracks
                    .iter()
                    .map(|track| RegressionTrack {
                        audio: &track.audio,
                        emotions: &track.emotions,
                        implicit_emotion: &track.implicit_emotion,
                        input_strength: track.input_strength,
                    })
                    .collect::<Vec<_>>();
                Ok(
                    match executor.pump(&views, backend, |metadata, output| {
                        callback(
                            CallbackMetadata {
                                kind: ModelKind::Regression,
                                track: metadata.track,
                                inference: None,
                                frame: metadata.frame,
                                timestamp: metadata.timestamp,
                                next_timestamp: metadata.next_timestamp,
                            },
                            PipelineOutput::Geometry(output),
                        )
                    })? {
                        PumpStatus::AwaitingInput => PipelineStatus::AwaitingInput,
                        PumpStatus::Complete => PipelineStatus::Complete,
                        PumpStatus::Interrupted => PipelineStatus::Interrupted,
                    },
                )
            }
            PipelineInner::Diffusion {
                executor,
                backend,
                tracks,
            } => {
                let views = tracks
                    .iter()
                    .map(|track| DiffusionTrack {
                        audio: &track.audio,
                        emotions: &track.emotions,
                        identity_index: track.identity_index,
                        input_strength: track.input_strength,
                    })
                    .collect::<Vec<_>>();
                let mut frames = 0;
                Ok(
                    match executor.execute(&views, backend, |metadata, output| {
                        frames += 1;
                        callback(
                            CallbackMetadata {
                                kind: ModelKind::Diffusion,
                                track: metadata.track,
                                inference: Some(metadata.inference),
                                frame: metadata.frame,
                                timestamp: metadata.timestamp,
                                next_timestamp: metadata.next_timestamp,
                            },
                            PipelineOutput::Geometry(output),
                        )
                    })? {
                        DiffusionExecutionStatus::AwaitingInput => PipelineStatus::AwaitingInput,
                        DiffusionExecutionStatus::Complete => PipelineStatus::Complete,
                        DiffusionExecutionStatus::Executed { tracks } => {
                            PipelineStatus::Executed { tracks, frames }
                        }
                    },
                )
            }
            PipelineInner::Emotion {
                executor,
                backend,
                tracks,
            } => {
                let views = tracks
                    .iter()
                    .map(|track| EmotionTrack {
                        audio: &track.audio,
                        preferred_emotions: None,
                        input_strength: track.input_strength,
                    })
                    .collect::<Vec<_>>();
                Ok(
                    match executor.execute(&views, backend, |metadata, output| {
                        callback(
                            CallbackMetadata {
                                kind: ModelKind::Emotion,
                                track: metadata.track,
                                inference: None,
                                frame: metadata.frame,
                                timestamp: metadata.timestamp,
                                next_timestamp: metadata.next_timestamp,
                            },
                            PipelineOutput::Emotion(output),
                        )
                    })? {
                        EmotionExecutionStatus::AwaitingInput => PipelineStatus::AwaitingInput,
                        EmotionExecutionStatus::Complete => PipelineStatus::Complete,
                        EmotionExecutionStatus::Executed { tracks, frames } => {
                            PipelineStatus::Executed { tracks, frames }
                        }
                    },
                )
            }
        }
    }
}

fn geometry_input_strength(model: &Model, index: usize) -> Result<f32> {
    match model.parameters(index)? {
        ModelParameters::Geometry(value) => {
            finite_strength(value.input_strength)?;
            Ok(value.input_strength)
        }
        _ => Err(invalid("geometry configuration is unavailable")),
    }
}

fn finite_strength(value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid("input strength must be finite"))
    }
}

fn accumulator_error(error: impl std::fmt::Display) -> Error {
    invalid(format!("track accumulator initialization failed: {error}"))
}

fn regression_track(
    tracks: &[RegressionOwnedTrack],
    index: usize,
) -> Result<&RegressionOwnedTrack> {
    tracks
        .get(index)
        .ok_or_else(|| invalid("track is out of range"))
}
fn regression_track_mut(
    tracks: &mut [RegressionOwnedTrack],
    index: usize,
) -> Result<&mut RegressionOwnedTrack> {
    tracks
        .get_mut(index)
        .ok_or_else(|| invalid("track is out of range"))
}
fn diffusion_track(tracks: &[DiffusionOwnedTrack], index: usize) -> Result<&DiffusionOwnedTrack> {
    tracks
        .get(index)
        .ok_or_else(|| invalid("track is out of range"))
}
fn diffusion_track_mut(
    tracks: &mut [DiffusionOwnedTrack],
    index: usize,
) -> Result<&mut DiffusionOwnedTrack> {
    tracks
        .get_mut(index)
        .ok_or_else(|| invalid("track is out of range"))
}
fn emotion_track(tracks: &[EmotionOwnedTrack], index: usize) -> Result<&EmotionOwnedTrack> {
    tracks
        .get(index)
        .ok_or_else(|| invalid("track is out of range"))
}
fn emotion_track_mut(
    tracks: &mut [EmotionOwnedTrack],
    index: usize,
) -> Result<&mut EmotionOwnedTrack> {
    tracks
        .get_mut(index)
        .ok_or_else(|| invalid("track is out of range"))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_installed_models_through_facade_when_configured() {
        let Some(paths) = std::env::var_os("AUDIO2FACE3D_TEST_FACADE_MODELS") else {
            return;
        };
        for root in std::env::split_paths(&paths) {
            let model = Model::load(root.join("model.json")).unwrap();
            let mut pipeline = TensorRtPipeline::load(&model, PipelineOptions::default()).unwrap();
            let parameters = match model.network() {
                NetworkDocument::Geometry(network) => match &network.params {
                    GeometryParameters::Regression(value) => TrackParameters::Regression {
                        input_strength: 0.5,
                        implicit_emotion: vec![0.0; value.implicit_emotion_len],
                    },
                    GeometryParameters::Diffusion(_) => TrackParameters::Diffusion {
                        input_strength: 0.5,
                        identity_index: 0,
                    },
                },
                NetworkDocument::Emotion(network) => {
                    let ModelParameters::Emotion(config) = model.parameters(0).unwrap() else {
                        unreachable!()
                    };
                    let (_, post_process) =
                        EmotionPostProcessData::from_model(network, config).unwrap();
                    TrackParameters::Emotion {
                        input_strength: 0.5,
                        post_process,
                    }
                }
            };
            pipeline.set_track_parameters(0, parameters).unwrap();
            pipeline.accumulate_audio(0, &vec![0.0; 1_600]).unwrap();
            pipeline.close_audio(0).unwrap();
            let mut callbacks = 0;
            loop {
                let status = pipeline
                    .execute(|_, _| {
                        callbacks += 1;
                        true
                    })
                    .unwrap();
                match status {
                    PipelineStatus::Executed { .. } => {}
                    PipelineStatus::Complete => break,
                    other => panic!("unexpected facade status: {other:?}"),
                }
            }
            assert!(callbacks > 0);
        }
    }
}
