//! Regression + host BlendShape adapter, owned by an independent control worker.
use crate::inference::{
    Backend, Cancellation, EngineFuture,
    animation::CURVE_NAMES,
    audio::SAMPLE_RATE,
    config::Config,
    worker::{Worker, wait},
};
use crate::types::{Error, ErrorKind};
use crate::types::{InputChunk, OutputBatch, RequestOptions};
use crate::{
    Model, ModelKind, ModelParameters,
    animation::BlendshapeData,
    audio2face::{
        BlendshapeExecutor, BlendshapeSolveComponentParameters,
        BlendshapeSolveExecutorCreationParameters, BlendshapeSolverConfigView,
        BlendshapeSolverDataView, BlendshapeSolverParams, FaceExecutor, GeometryExecutionOption,
        GeometryExecutorCreationParameters, GeometryTrackResources, HostBlendshapeSolveExecutor,
        HostBlendshapeSolveExecutorCreationParameters,
        regression::{
            RegressionGeometryExecutorCreationParameters, RegressionGeometryExecutorFactory,
        },
    },
    audio2x::{AudioAccumulator, CallbackMetadata, EmotionAccumulator, ExecutionState, FrameRate},
    common::{
        GeometryAudioParameters, GeometryParameters, NetworkDocument, load_blendshape_config,
    },
};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

fn internal(error: impl std::fmt::Display) -> Error {
    Error::new(ErrorKind::Inference, format!("regression backend: {error}"))
}

pub struct Runtime {
    executor: HostBlendshapeSolveExecutor,
    order: Vec<usize>,
    clamp: bool,
    emotion: super::emotion::Stage,
}

fn load_sync(config: &Config, header: &RequestOptions) -> Result<Runtime, Error> {
    let loaded_at = std::time::Instant::now();
    let path = config
        .model
        .as_ref()
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "--model is required"))?;
    let model = Model::load(path).map_err(internal)?;
    if model.kind() != ModelKind::Regression || model.sample_rate() != SAMPLE_RATE as usize {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "expected a 16000 Hz regression model",
        ));
    }
    let NetworkDocument::Geometry(network) = model.network() else {
        return Err(internal("missing geometry network"));
    };
    let (
        GeometryParameters::Regression(parameters),
        GeometryAudioParameters::Regression(audio_parameters),
    ) = (&network.params, &network.audio_params)
    else {
        return Err(internal("missing regression parameters"));
    };
    let ModelParameters::Geometry(geometry_config) = model.parameters(0).map_err(internal)? else {
        return Err(internal("missing geometry configuration"));
    };
    // This runs on the existing inference control worker, before any native engine is created.
    crate::logging::integration::LogScope::capture()
        .context()
        .initialize_native()
        .map_err(|error| Error::new(ErrorKind::RuntimeUnavailable, error.to_string()))?;
    let mut geometry_config = geometry_config.clone();
    super::parameters::face(&mut geometry_config, header)?;
    let audio = Arc::new(AudioAccumulator::new(audio_parameters.buffer_len, 0).map_err(internal)?);
    let emotions = Arc::new(
        EmotionAccumulator::new(parameters.explicit_emotions.len(), 30).map_err(internal)?,
    );
    let emotion = super::emotion::Stage::load(
        config,
        header,
        &parameters.explicit_emotions,
        &parameters.default_emotion,
        emotions.clone(),
    )?;
    let bundle = wait(RegressionGeometryExecutorFactory::load_with_config(
        RegressionGeometryExecutorCreationParameters {
            model_path: path.clone(),
            common: GeometryExecutorCreationParameters {
                tracks: vec![GeometryTrackResources { audio, emotions }],
                device_ordinal: i32::try_from(config.device).map_err(|_| {
                    Error::new(ErrorKind::InvalidInput, "device ordinal is out of range")
                })?,
                execution_option: GeometryExecutionOption::SKIN,
            },
            input_strength: geometry_config.input_strength,
            frame_rate: FrameRate::new(30, 1).map_err(internal)?,
            source_emotion_shot: geometry_config.source_shot.clone(),
            source_emotion_frame: geometry_config
                .source_frame
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(0),
        },
        geometry_config,
    ))
    .map_err(internal)?;
    let paths = model.blendshape_paths(0).map_err(internal)?;
    let skin_paths = paths
        .get("skin")
        .ok_or_else(|| internal("missing skin blendshape paths"))?;
    let data = BlendshapeData::load_npz(&skin_paths.data).map_err(internal)?;
    let mut bs = load_blendshape_config(&skin_paths.config)
        .map_err(internal)?
        .blendshape_params;
    if data.pose_names.len() != CURVE_NAMES.len() {
        return Err(internal("skin must contain exactly 52 non-neutral poses"));
    }
    let order = CURVE_NAMES
        .iter()
        .map(|wire| {
            let model_name = wire[..1].to_ascii_lowercase() + &wire[1..];
            data.pose_names
                .iter()
                .position(|name| name == &model_name)
                .ok_or_else(|| internal(format!("missing model pose {model_name}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let clamp =
        super::parameters::blendshapes(header, &order, &mut bs.multipliers, &mut bs.offsets)?;
    let names = data
        .pose_names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let executor = bundle
        .try_into_host_blendshape(HostBlendshapeSolveExecutorCreationParameters {
            components: BlendshapeSolveExecutorCreationParameters {
                skin: Some(BlendshapeSolveComponentParameters {
                    params: BlendshapeSolverParams {
                        l1_regularization: bs.l1_regularization,
                        l2_regularization: bs.l2_regularization,
                        symmetry_regularization: bs.symmetry_regularization,
                        temporal_regularization: bs.temporal_regularization,
                        template_bounding_box_size: bs.template_bb_size,
                        tolerance: bs.tolerance,
                    },
                    config: BlendshapeSolverConfigView {
                        active_poses: &bs.active_poses,
                        cancel_poses: &bs.cancel_poses,
                        symmetry_poses: &bs.symmetry_poses,
                        multipliers: &bs.multipliers,
                        offsets: &bs.offsets,
                    },
                    data: BlendshapeSolverDataView {
                        neutral_pose: &data.neutral_pose,
                        delta_poses: &data.delta_poses,
                        pose_mask: data.pose_mask.as_deref(),
                        pose_names: &names,
                    },
                }),
                tongue: None,
            },
            job_runner: None,
        })
        .map_err(|failure| internal(failure.error))?;
    if executor.weight_count() != 52 {
        return Err(internal("solver output must contain 52 weights"));
    }
    tracing::info!(model = %path.display(), device = config.device, load_ms = loaded_at.elapsed().as_millis(), "regression model and host solver ready");
    Ok(Runtime {
        executor,
        order,
        clamp,
        emotion,
    })
}

struct CapturedFrame {
    metadata: CallbackMetadata,
    weights: Vec<f32>,
}
struct Tick {
    state: ExecutionState,
    frames: Vec<CapturedFrame>,
}

impl Runtime {
    fn tick(&mut self) -> Result<Tick, Error> {
        // Input is fed in 534-sample increments, so a normal tick schedules at
        // most a few frames; closing adds only the model's lookahead tail.
        self.emotion.tick()?;
        let captured = Arc::new(Mutex::new(Ok(Vec::new())));
        let output = captured.clone();
        let execution = self
            .executor
            .execute(Arc::new(move |event| {
                let Ok(mut slot) = output.lock() else {
                    return;
                };
                if let Ok(frames) = &mut *slot {
                    match event {
                        Ok(frame) if frames.len() < 64 => frames.push(CapturedFrame {
                            metadata: frame.metadata,
                            weights: frame.weights.to_vec(),
                        }),
                        Ok(_) => *slot = Err(internal("callback batch exceeds 64 frames")),
                        Err(error) => *slot = Err(internal(error)),
                    }
                }
            }))
            .map_err(internal)?;
        // Do not select cancellation against wait_all: dropping Execution does
        // not cancel GPU/CPU jobs or their borrowed resources.
        let report = wait(execution.wait_all()).map_err(internal)?;
        let mut frames = captured
            .lock()
            .map_err(internal)?
            .as_mut()
            .map_err(|e| e.clone())?
            .drain(..)
            .collect::<Vec<_>>();
        frames.sort_by_key(|frame| frame.metadata.frame_index);
        let next = self
            .executor
            .next_audio_sample_to_read(0)
            .map_err(internal)?;
        let audio = self.executor.audio_accumulator(0).map_err(internal)?;
        audio
            .drop_samples_before(next.min(audio.nb_accumulated_samples()))
            .map_err(internal)?;
        Ok(Tick {
            state: report.state,
            frames,
        })
    }
}

pub struct RegressionState {
    runtime: Option<Runtime>,
    pcm: VecDeque<u8>,
    pending: VecDeque<CapturedFrame>,
    received: u64,
    fed: u64,
    emitted: u64,
    next_index: usize,
    finished: bool,
    closed: bool,
    complete: bool,
    max_samples: u64,
}

impl RegressionState {
    pub fn new(runtime: Runtime, config: &Config) -> Self {
        Self {
            runtime: Some(runtime),
            pcm: VecDeque::new(),
            pending: VecDeque::new(),
            received: 0,
            fed: 0,
            emitted: 0,
            next_index: 0,
            finished: false,
            closed: false,
            complete: false,
            max_samples: u64::from(config.max_audio_seconds) * SAMPLE_RATE,
        }
    }

    fn output(&mut self, frame: CapturedFrame) -> Result<Option<OutputBatch>, Error> {
        let m = frame.metadata;
        if m.track_index != 0
            || m.frame_index != self.next_index
            || m.timestamp < 0
            || m.next_timestamp <= m.timestamp
        {
            return Err(internal(format!("invalid callback metadata: {m:?}")));
        }
        self.next_index += 1;
        let start = m.timestamp as u64;
        if start >= self.received {
            return Ok(None);
        }
        if start != self.emitted {
            return Err(internal("non-contiguous native audio timestamps"));
        }
        let end = (m.next_timestamp as u64).min(self.received);
        let bytes = ((end - start) * 2) as usize;
        if bytes > self.pcm.len()
            || frame.weights.len() != 52
            || !frame.weights.iter().all(|v| v.is_finite())
        {
            return Err(internal("invalid native PCM interval or weights"));
        }
        let order = &self
            .runtime
            .as_ref()
            .ok_or_else(|| internal("runtime unavailable"))?
            .order;
        let clamp = self.runtime.as_ref().unwrap().clamp;
        let weights = crate::inference::animation::ordered_weights(frame.weights, order, clamp);
        let metadata = self
            .runtime
            .as_mut()
            .unwrap()
            .emotion
            .metadata(start as i64)?;
        let pcm = if bytes == self.pcm.len() {
            std::mem::take(&mut self.pcm).into()
        } else {
            self.pcm.drain(..bytes).collect()
        };
        self.emitted = end;
        let mut batch = crate::inference::animation::frame(start, pcm, weights)?;
        batch.emotion = Some(metadata);
        Ok(Some(batch))
    }
}

impl RegressionState {
    fn push(&mut self, input: InputChunk) -> Result<(), Error> {
        let (pcm, emotions) = input.into_parts();
        let audio_buffer = pcm.into_vec();
        if !audio_buffer.len().is_multiple_of(2) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "PCM chunk must contain whole 16-bit samples",
            ));
        }
        let total = self.received + (audio_buffer.len() / 2) as u64;
        if total > self.max_samples {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "audio duration limit exceeded",
            ));
        }
        self.runtime
            .as_mut()
            .ok_or_else(|| internal("runtime unavailable"))?
            .emotion
            .push_keys(emotions)?;
        self.received = total;
        if self.pcm.is_empty() {
            self.pcm = audio_buffer.into();
        } else {
            self.pcm.extend(audio_buffer);
        }
        Ok(())
    }

    fn next_frame(&mut self, cancel: &Cancellation) -> Result<Option<OutputBatch>, Error> {
        loop {
            if cancel.is_cancelled() {
                return Err(Error::new(
                    ErrorKind::Cancelled,
                    "inference cancelled after draining started jobs",
                ));
            }
            if let Some(frame) = self.pending.pop_front() {
                if let Some(output) = self.output(frame)? {
                    return Ok(Some(output));
                }
                continue;
            }
            if self.complete {
                if self.emitted != self.received {
                    return Err(internal("native completion left unreturned audio"));
                }
                return Ok(None);
            }
            let mut runtime = self
                .runtime
                .take()
                .ok_or_else(|| internal("runtime unavailable"))?;
            let has_input = self.fed < self.received;
            if has_input {
                let count = (self.received - self.fed).min(534) as usize;
                let mut scratch = [0.0f32; 534];
                let samples = &mut scratch[..count];
                crate::inference::audio::normalized_samples(
                    &self.pcm,
                    ((self.fed - self.emitted) * 2) as usize,
                    samples,
                );
                if let Err(error) = runtime
                    .executor
                    .audio_accumulator(0)
                    .and_then(|audio| audio.accumulate(samples))
                {
                    self.runtime = Some(runtime);
                    return Err(internal(error));
                }
                if let Err(error) = runtime.emotion.feed(samples) {
                    self.runtime = Some(runtime);
                    return Err(error);
                }
                self.fed += count as u64;
            } else if self.finished && !self.closed {
                if let Err(error) = runtime
                    .executor
                    .audio_accumulator(0)
                    .and_then(|audio| audio.close())
                {
                    self.runtime = Some(runtime);
                    return Err(internal(error));
                }
                if let Err(error) = runtime.emotion.finish() {
                    self.runtime = Some(runtime);
                    return Err(error);
                }
                self.closed = true;
            } else if !self.closed {
                self.runtime = Some(runtime);
                return Ok(None);
            }
            let result = runtime.tick();
            self.runtime = Some(runtime);
            let tick = result?;
            self.complete = tick.state == ExecutionState::Complete;
            if self.closed
                && tick.state == ExecutionState::AwaitingInput
                && self.runtime.as_ref().unwrap().emotion.is_complete()
            {
                return Err(internal("executor awaiting input after close"));
            }
            if self.closed
                && !self.complete
                && tick.frames.is_empty()
                && self.runtime.as_ref().unwrap().emotion.is_complete()
            {
                return Err(internal("executor made no progress after close"));
            }
            self.pending.extend(tick.frames);
        }
    }

    fn finish(&mut self) -> Result<(), Error> {
        if self.received == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "audio clip must contain at least one sample",
            ));
        }
        self.finished = true;
        Ok(())
    }

    fn close(&mut self) -> Result<(), Error> {
        self.pending.clear();
        self.pcm.clear();
        if let Some(runtime) = self.runtime.take() {
            // The native destructor also drains jobs when execute itself failed.
            drop(runtime);
        }
        Ok(())
    }
}

/// The handle never owns native state: abandonment still cleans up on the worker.
pub struct RegressionBackend {
    worker: Worker<RegressionState>,
}
impl RegressionBackend {
    pub async fn load(config: Config, options: RequestOptions) -> Result<Self, Error> {
        let worker = Worker::start(move || {
            let runtime = load_sync(&config, &options)?;
            Ok(RegressionState::new(runtime, &config))
        })
        .await?;
        Ok(Self { worker })
    }
}
impl Backend for RegressionBackend {
    fn push(&mut self, input: InputChunk) -> EngineFuture<'_, ()> {
        Box::pin(self.worker.call(move |state| state.push(input)))
    }
    fn next_frame<'a>(
        &'a mut self,
        cancel: &'a Cancellation,
    ) -> EngineFuture<'a, Option<OutputBatch>> {
        let cancel = cancel.clone();
        Box::pin(self.worker.call(move |state| state.next_frame(&cancel)))
    }
    fn finish(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(self.worker.call(RegressionState::finish))
    }
    fn close(&mut self) -> EngineFuture<'_, ()> {
        Box::pin(self.worker.call(RegressionState::close))
    }
    fn success_message(&self) -> &'static str {
        "Regression audio processing completed successfully."
    }
}

#[cfg(test)]
mod ownership_profile {
    use super::*;
    #[test]
    #[ignore = "requires A2F_MODEL and CUDA/TensorRT"]
    fn native_weight_order() {
        let config = Config {
            backend: crate::inference::BackendKind::Regression,
            model: Some(std::env::var_os("A2F_MODEL").expect("A2F_MODEL").into()),
            ..Config::default()
        };
        let runtime = load_sync(&config, &RequestOptions::default()).unwrap();
        println!(
            "NATIVE_ORDER channels={} identity={}",
            runtime.order.len(),
            runtime.order.iter().copied().eq(0..52)
        );
        assert_eq!(runtime.order.len(), 52);
    }
}
