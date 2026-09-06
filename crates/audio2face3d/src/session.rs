use crate::animation::{
    DiffusionContract, DiffusionExecutionStatus, DiffusionExecutor, DiffusionPostprocessor,
    DiffusionTrack, EyesRotation, GeometryModelData, PumpStatus, RegressionContract,
    RegressionExecutor, RegressionGeometry, RegressionPostprocessor, RegressionTrack,
    TensorRtDiffusionBackend, TensorRtRegressionBackend,
};
use crate::common::{
    AudioAccumulator, EmotionAccumulator, Error, GeometryAudioParameters, GeometryParameters,
    NetworkDocument, Result,
};
use crate::cuda::{CudaStream, GpuDevice};
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

#[derive(Clone, Debug, PartialEq)]
pub struct GeometryFrame {
    pub skin: Vec<f32>,
    pub tongue: Vec<f32>,
    pub jaw_transform: [f32; 16],
    pub eyes_rotation: EyesRotation,
}

impl GeometryFrame {
    pub fn packed(&self) -> Vec<f32> {
        let mut values = Vec::with_capacity(self.skin.len() + self.tongue.len() + 22);
        values.extend_from_slice(&self.skin);
        values.extend_from_slice(&self.tongue);
        values.extend_from_slice(&self.jaw_transform);
        values.extend_from_slice(&self.eyes_rotation.right);
        values.extend_from_slice(&self.eyes_rotation.left);
        values
    }
}

impl From<RegressionGeometry> for GeometryFrame {
    fn from(value: RegressionGeometry) -> Self {
        Self {
            skin: value.skin,
            tongue: value.tongue,
            jaw_transform: value.jaw_transform,
            eyes_rotation: value.eyes_rotation,
        }
    }
}

enum GeometryPostprocessors {
    Regression(Vec<RegressionPostprocessor>),
    Diffusion(Vec<DiffusionPostprocessor>),
}

/// Owns a geometry pipeline and model-specific post-processors.
pub struct GeometryExecutorBundle {
    pipeline: TensorRtPipeline,
    postprocessors: GeometryPostprocessors,
    model_data: Vec<GeometryModelData>,
    dt: f32,
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

/// Borrowed geometry components owned by a [`TensorRtPipeline`].
///
/// The enum keeps the executor and its matching TensorRT backend under one
/// borrow, so callers cannot accidentally combine components from different
/// model kinds or outlive the pipeline that owns them.
pub enum GeometryPipelineComponents<'a> {
    Regression {
        executor: &'a RegressionExecutor,
        backend: &'a TensorRtRegressionBackend,
    },
    Diffusion {
        executor: &'a DiffusionExecutor,
        backend: &'a TensorRtDiffusionBackend,
    },
}

/// Mutably borrowed geometry components owned by a [`TensorRtPipeline`].
pub enum GeometryPipelineComponentsMut<'a> {
    Regression {
        executor: &'a mut RegressionExecutor,
        backend: &'a mut TensorRtRegressionBackend,
    },
    Diffusion {
        executor: &'a mut DiffusionExecutor,
        backend: &'a mut TensorRtDiffusionBackend,
    },
}

/// Borrowed per-track inputs and parameters owned by a pipeline.
pub enum PipelineTrackComponents<'a> {
    Regression {
        audio: &'a AudioAccumulator,
        emotions: &'a EmotionAccumulator,
        implicit_emotion: &'a [f32],
        input_strength: f32,
    },
    Diffusion {
        audio: &'a AudioAccumulator,
        emotions: &'a EmotionAccumulator,
        identity_index: usize,
        input_strength: f32,
    },
    Emotion {
        audio: &'a AudioAccumulator,
        input_strength: f32,
    },
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

    /// Returns the CUDA stream owned by the geometry backend.
    ///
    /// The pipeline is also used for emotion models, which do not expose a
    /// geometry stream through this facade; callers must therefore handle the
    /// model-kind error instead of receiving a stream with unrelated
    /// ownership semantics.
    pub fn cuda_stream(&self) -> Result<&CudaStream> {
        match self.geometry_components()? {
            GeometryPipelineComponents::Regression { backend, .. } => Ok(backend.stream()),
            GeometryPipelineComponents::Diffusion { backend, .. } => Ok(backend.stream()),
        }
    }

    /// Borrows the audio accumulator owned by one pipeline track.
    pub fn audio_accumulator(&self, track: usize) -> Result<&AudioAccumulator> {
        match self.track_components(track)? {
            PipelineTrackComponents::Regression { audio, .. }
            | PipelineTrackComponents::Diffusion { audio, .. }
            | PipelineTrackComponents::Emotion { audio, .. } => Ok(audio),
        }
    }

    /// Borrows the emotion accumulator owned by one geometry track.
    pub fn emotion_accumulator(&self, track: usize) -> Result<&EmotionAccumulator> {
        match self.track_components(track)? {
            PipelineTrackComponents::Regression { emotions, .. }
            | PipelineTrackComponents::Diffusion { emotions, .. } => Ok(emotions),
            PipelineTrackComponents::Emotion { .. } => Err(invalid(
                "emotion pipelines have no geometry emotion accumulator",
            )),
        }
    }

    /// Borrows the low-level geometry executor/backend pair without
    /// transferring ownership. Emotion pipelines reject this accessor.
    pub fn geometry_components(&self) -> Result<GeometryPipelineComponents<'_>> {
        match &self.inner {
            PipelineInner::Regression {
                executor, backend, ..
            } => Ok(GeometryPipelineComponents::Regression { executor, backend }),
            PipelineInner::Diffusion {
                executor, backend, ..
            } => Ok(GeometryPipelineComponents::Diffusion { executor, backend }),
            PipelineInner::Emotion { .. } => {
                Err(invalid("emotion pipelines have no geometry components"))
            }
        }
    }

    /// Mutably borrows the matching low-level executor/backend pair.
    pub fn geometry_components_mut(&mut self) -> Result<GeometryPipelineComponentsMut<'_>> {
        match &mut self.inner {
            PipelineInner::Regression {
                executor, backend, ..
            } => Ok(GeometryPipelineComponentsMut::Regression { executor, backend }),
            PipelineInner::Diffusion {
                executor, backend, ..
            } => Ok(GeometryPipelineComponentsMut::Diffusion { executor, backend }),
            PipelineInner::Emotion { .. } => {
                Err(invalid("emotion pipelines have no geometry components"))
            }
        }
    }

    /// Borrows one track's accumulators and current parameter values.
    pub fn track_components(&self, track: usize) -> Result<PipelineTrackComponents<'_>> {
        match &self.inner {
            PipelineInner::Regression { tracks, .. } => {
                let track = regression_track(tracks, track)?;
                Ok(PipelineTrackComponents::Regression {
                    audio: &track.audio,
                    emotions: &track.emotions,
                    implicit_emotion: &track.implicit_emotion,
                    input_strength: track.input_strength,
                })
            }
            PipelineInner::Diffusion { tracks, .. } => {
                let track = diffusion_track(tracks, track)?;
                Ok(PipelineTrackComponents::Diffusion {
                    audio: &track.audio,
                    emotions: &track.emotions,
                    identity_index: track.identity_index,
                    input_strength: track.input_strength,
                })
            }
            PipelineInner::Emotion { tracks, .. } => {
                let track = emotion_track(tracks, track)?;
                Ok(PipelineTrackComponents::Emotion {
                    audio: &track.audio,
                    input_strength: track.input_strength,
                })
            }
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

impl GeometryExecutorBundle {
    pub fn builder(
        model: &Model,
        options: PipelineOptions,
    ) -> Result<crate::GeometryExecutorBundleBuilder<Self>> {
        crate::GeometryExecutorBundleBuilder::from_model(model, options)
    }

    pub fn load(model: &Model, options: PipelineOptions) -> Result<Self> {
        if model.kind() == ModelKind::Emotion {
            return Err(invalid(
                "geometry bundle requires a Regression or Diffusion model",
            ));
        }
        let dt = options.frame_rate_denominator as f32 / options.frame_rate_numerator as f32;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(invalid("geometry bundle frame rate is invalid"));
        }
        let mut model_data = Vec::with_capacity(options.track_count);
        let postprocessors = match model.network() {
            NetworkDocument::Geometry(network) => match &network.params {
                GeometryParameters::Regression(parameters) => {
                    let mut processors = Vec::with_capacity(options.track_count);
                    for track in 0..options.track_count {
                        let index = track.min(model.parameter_count() - 1);
                        let data = GeometryModelData::load_regression(
                            model
                                .model_data_path(index)
                                .or_else(|_| model.model_data_path(0))?,
                        )?;
                        let ModelParameters::Geometry(config) = model.parameters(index)? else {
                            return Err(invalid("regression geometry config is unavailable"));
                        };
                        processors.push(data.regression_postprocessor(
                            config,
                            parameters.num_shapes_skin,
                            parameters.num_shapes_tongue,
                        )?);
                        model_data.push(data);
                    }
                    GeometryPostprocessors::Regression(processors)
                }
                GeometryParameters::Diffusion(parameters) => {
                    let contract = DiffusionContract::new(
                        parameters,
                        match &network.audio_params {
                            GeometryAudioParameters::Diffusion(audio) => audio,
                            _ => return Err(invalid("diffusion audio config is unavailable")),
                        },
                    )?;
                    let mut processors = Vec::with_capacity(options.track_count);
                    for track in 0..options.track_count {
                        let index = track.min(model.parameter_count() - 1);
                        let data = GeometryModelData::load_diffusion(
                            model
                                .model_data_path(index)
                                .or_else(|_| model.model_data_path(0))?,
                        )?;
                        let ModelParameters::Geometry(config) = model.parameters(index)? else {
                            return Err(invalid("diffusion geometry config is unavailable"));
                        };
                        processors
                            .push(data.diffusion_postprocessor(config, contract.result_layout)?);
                        model_data.push(data);
                    }
                    GeometryPostprocessors::Diffusion(processors)
                }
            },
            _ => return Err(invalid("geometry network is unavailable")),
        };
        Ok(Self {
            pipeline: TensorRtPipeline::load(model, options)?,
            postprocessors,
            model_data,
            dt,
        })
    }

    pub fn pipeline(&self) -> &TensorRtPipeline {
        &self.pipeline
    }

    pub fn pipeline_mut(&mut self) -> &mut TensorRtPipeline {
        &mut self.pipeline
    }

    /// Formal owning-executor accessor for bundle consumers.
    pub fn executor(&self) -> &TensorRtPipeline {
        &self.pipeline
    }

    /// Returns the stream owned by this geometry executor.
    pub fn cuda_stream(&self) -> Result<&CudaStream> {
        self.pipeline.cuda_stream()
    }

    /// Borrows a track's shared audio accumulator.
    pub fn audio_accumulator(&self, track: usize) -> Result<&AudioAccumulator> {
        self.pipeline.audio_accumulator(track)
    }

    /// Borrows a track's shared emotion accumulator.
    pub fn emotion_accumulator(&self, track: usize) -> Result<&EmotionAccumulator> {
        self.pipeline.emotion_accumulator(track)
    }

    pub fn model_data(&self, track: usize) -> Result<&GeometryModelData> {
        self.model_data
            .get(track)
            .ok_or_else(|| invalid("geometry bundle track is out of range"))
    }

    pub fn track_count(&self) -> usize {
        self.pipeline.track_count()
    }

    pub fn accumulate_audio(&self, track: usize, samples: &[f32]) -> Result<()> {
        self.pipeline.accumulate_audio(track, samples)
    }

    pub fn close_audio(&self, track: usize) -> Result<()> {
        self.pipeline.close_audio(track)
    }

    pub fn set_track_parameters(&mut self, track: usize, value: TrackParameters) -> Result<()> {
        self.pipeline.set_track_parameters(track, value)
    }

    pub fn execute<C>(&mut self, mut callback: C) -> Result<PipelineStatus>
    where
        C: for<'frame> FnMut(CallbackMetadata, &'frame GeometryFrame) -> bool,
    {
        let processors = &mut self.postprocessors;
        let dt = self.dt;
        let mut postprocess_error = None;
        let status = self.pipeline.execute(|metadata, output| {
            let PipelineOutput::Geometry(raw) = output else {
                postprocess_error = Some(invalid("geometry bundle received an emotion output"));
                return false;
            };
            let result = match processors {
                GeometryPostprocessors::Regression(processors) => processors
                    .get_mut(metadata.track)
                    .ok_or_else(|| invalid("regression postprocessor track is out of range"))
                    .and_then(|processor| processor.process(raw, dt)),
                GeometryPostprocessors::Diffusion(processors) => processors
                    .get_mut(metadata.track)
                    .ok_or_else(|| invalid("diffusion postprocessor track is out of range"))
                    .and_then(|processor| processor.process(raw, dt)),
            };
            match result {
                Ok(frame) => {
                    let frame = GeometryFrame::from(frame);
                    callback(metadata, &frame)
                }
                Err(error) => {
                    postprocess_error = Some(error);
                    false
                }
            }
        })?;
        if let Some(error) = postprocess_error {
            Err(error)
        } else {
            Ok(status)
        }
    }

    pub fn reset_postprocess(&mut self, track: usize) -> Result<()> {
        match &mut self.postprocessors {
            GeometryPostprocessors::Regression(processors) => processors
                .get_mut(track)
                .ok_or_else(|| invalid("regression postprocessor track is out of range"))?
                .reset(),
            GeometryPostprocessors::Diffusion(processors) => processors
                .get_mut(track)
                .ok_or_else(|| invalid("diffusion postprocessor track is out of range"))?
                .reset(),
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
    fn geometry_frame_packs_original_result_component_order() {
        let frame = GeometryFrame {
            skin: vec![1.0, 2.0],
            tongue: vec![3.0],
            jaw_transform: [4.0; 16],
            eyes_rotation: EyesRotation {
                right: [5.0; 3],
                left: [6.0; 3],
            },
        };
        let packed = frame.packed();
        assert_eq!(&packed[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(&packed[3..19], &[4.0; 16]);
        assert_eq!(&packed[19..22], &[5.0; 3]);
        assert_eq!(&packed[22..25], &[6.0; 3]);
    }

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
