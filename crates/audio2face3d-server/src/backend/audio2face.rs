//! Regression + host BlendShape adapter. CUDA work stays on blocking workers.
use super::Backend;
use crate::{
    animation::CURVE_NAMES,
    audio::SAMPLE_RATE,
    config::Config,
    proto::{
        a2f::AudioWithEmotion,
        animation::{AnimationData, AudioWithTimeCode, FloatArrayWithTimeCode, SkelAnimation},
    },
};
use audio2face3d::{
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
use tonic::Status;

fn internal(error: impl std::fmt::Display) -> Status {
    Status::internal(format!("regression backend: {error}"))
}

pub struct Runtime {
    executor: HostBlendshapeSolveExecutor,
    order: Vec<usize>,
    clamp: bool,
    emotion: super::emotion::Stage,
}

pub async fn load(config: Config) -> Result<Runtime, Status> {
    load_with_header(config, Default::default()).await
}
pub async fn load_with_header(
    config: Config,
    header: crate::proto::controller::AudioStreamHeader,
) -> Result<Runtime, Status> {
    let handle = tokio::runtime::Handle::current();
    tokio::task::spawn_blocking(move || load_sync(&config, &header, &handle))
        .await
        .map_err(internal)?
}

fn load_sync(
    config: &Config,
    header: &crate::proto::controller::AudioStreamHeader,
    handle: &tokio::runtime::Handle,
) -> Result<Runtime, Status> {
    let loaded_at = std::time::Instant::now();
    let path = config
        .model
        .as_ref()
        .ok_or_else(|| Status::invalid_argument("--model is required"))?;
    let model = Model::load(path).map_err(internal)?;
    if model.kind() != ModelKind::Regression || model.sample_rate() != SAMPLE_RATE as usize {
        return Err(Status::invalid_argument(
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
        handle,
    )?;
    let bundle = handle
        .block_on(RegressionGeometryExecutorFactory::load_with_config(
            RegressionGeometryExecutorCreationParameters {
                model_path: path.clone(),
                common: GeometryExecutorCreationParameters {
                    tracks: vec![GeometryTrackResources { audio, emotions }],
                    device_ordinal: i32::try_from(config.device)
                        .map_err(|_| Status::invalid_argument("device ordinal is out of range"))?,
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
    fn tick(&mut self, handle: &tokio::runtime::Handle) -> Result<Tick, Status> {
        // Input is fed in 534-sample increments, so a normal tick schedules at
        // most a few frames; closing adds only the model's lookahead tail.
        self.emotion.tick(handle)?;
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
        let report = handle.block_on(execution.wait_all()).map_err(internal)?;
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

pub struct RegressionBackend {
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

impl RegressionBackend {
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

    fn output(&mut self, frame: CapturedFrame) -> Result<Option<AnimationData>, Status> {
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
        let weights = order
            .iter()
            .map(|&i| {
                if clamp {
                    frame.weights[i].clamp(0.0, 1.0)
                } else {
                    frame.weights[i]
                }
            })
            .collect();
        let metadata = self
            .runtime
            .as_mut()
            .unwrap()
            .emotion
            .metadata(start as i64)?;
        let pcm = self.pcm.drain(..bytes).collect();
        self.emitted = end;
        let time_code = start as f64 / SAMPLE_RATE as f64;
        Ok(Some(AnimationData {
            metadata,
            skel_animation: Some(SkelAnimation {
                blend_shape_weights: vec![FloatArrayWithTimeCode {
                    time_code,
                    values: weights,
                }],
                ..Default::default()
            }),
            audio: Some(AudioWithTimeCode {
                time_code,
                audio_buffer: pcm,
            }),
            ..Default::default()
        }))
    }
}

#[tonic::async_trait]
impl Backend for RegressionBackend {
    fn push(&mut self, input: AudioWithEmotion) -> Result<(), Status> {
        if !input.audio_buffer.len().is_multiple_of(2) {
            return Err(Status::invalid_argument(
                "PCM chunk must contain whole 16-bit samples",
            ));
        }
        let total = self.received + (input.audio_buffer.len() / 2) as u64;
        if total > self.max_samples {
            return Err(Status::resource_exhausted("audio duration limit exceeded"));
        }
        self.runtime
            .as_mut()
            .ok_or_else(|| internal("runtime unavailable"))?
            .emotion
            .push_keys(input.emotions)?;
        self.received = total;
        self.pcm.extend(input.audio_buffer);
        Ok(())
    }

    async fn next_frame(
        &mut self,
        cancel: &tokio_util::sync::CancellationToken,
    ) -> Result<Option<AnimationData>, Status> {
        loop {
            if cancel.is_cancelled() {
                return Err(Status::cancelled(
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
                let bytes = self
                    .pcm
                    .iter()
                    .skip(((self.fed - self.emitted) * 2) as usize)
                    .take(count * 2)
                    .copied()
                    .collect::<Vec<_>>();
                let samples = bytes
                    .chunks_exact(2)
                    .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
                    .collect::<Vec<_>>();
                if let Err(error) = runtime
                    .executor
                    .audio_accumulator(0)
                    .and_then(|audio| audio.accumulate(&samples))
                {
                    self.runtime = Some(runtime);
                    return Err(internal(error));
                }
                if let Err(error) = runtime.emotion.feed(&samples) {
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
            let handle = tokio::runtime::Handle::current();
            let (runtime, result) = tokio::task::spawn_blocking(move || {
                let result = runtime.tick(&handle);
                (runtime, result)
            })
            .await
            .map_err(internal)?;
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

    fn finish(&mut self) -> Result<(), Status> {
        if self.received == 0 {
            return Err(Status::invalid_argument(
                "audio clip must contain at least one sample",
            ));
        }
        self.finished = true;
        Ok(())
    }

    async fn close(&mut self) -> Result<(), Status> {
        self.pending.clear();
        self.pcm.clear();
        if let Some(runtime) = self.runtime.take() {
            // The native destructor also drains jobs when execute itself failed.
            tokio::task::spawn_blocking(move || drop(runtime))
                .await
                .map_err(internal)?;
        }
        Ok(())
    }

    fn success_message(&self) -> &'static str {
        "Regression audio processing completed successfully."
    }
}
