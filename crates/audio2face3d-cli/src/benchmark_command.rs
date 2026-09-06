use crate::async_util::block_on;
use audio2face3d::animation::{
    BlendshapeData, DiffusionBackend, DiffusionContract, DiffusionFrameInput, GpuBlendshapeSolver,
    InteractiveGpuBlendshapeLayer, RegressionBackend, RegressionContract, RegressionFrameInput,
    RegressionGeometry, TensorRtDiffusionBackend, TensorRtRegressionBackend,
};
use audio2face3d::audio2emotion::post_process::{
    PostProcessData, PostProcessParams, PostProcessor,
};
use audio2face3d::audio2face::{
    BlendshapeSolveComponentParameters, BlendshapeSolverConfigView, BlendshapeSolverDataView,
    BlendshapeSolverParams, CpuBlendshapeSolver, DeviceBlendshapeSolveInteractiveExecutor,
    create_blendshape_solver,
};
use audio2face3d::common::{
    Error, GeometryAudioParameters, GeometryParameters, NetworkDocument, Result,
    load_blendshape_config,
};
use audio2face3d::cuda::{CudaStream, DeviceBuffer, GpuDevice};
use audio2face3d::emotion::{ClassifierBackend, ClassifierContract, TensorRtClassifierBackend};
use audio2face3d::{BenchmarkRunner, Model, ModelKind, ModelParameters};
use audio2face3d_cli::reference::sha256_file;
use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::sync::Arc;

pub fn run(
    descriptor: &Path,
    tracks: usize,
    precision: &str,
    scope: &str,
    iterations: usize,
    engine: Option<&Path>,
    output: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if precision != "fp32" && precision != "fp16" {
        return Err("precision must be fp32 or fp16".into());
    }
    let model = Model::load(descriptor)?;
    let workload = RefCell::new(None);
    let report = BenchmarkRunner {
        warmup_iterations: 5,
        measured_iterations: iterations,
    }
    .run(
        || {
            *workload.borrow_mut() = Some(Workload::load(
                &model,
                engine.unwrap_or_else(|| model.engine_path()),
                tracks,
                scope,
            )?);
            Ok(())
        },
        || {
            Model::load(descriptor)?;
            Ok(())
        },
        || {
            workload
                .borrow_mut()
                .as_mut()
                .expect("build phase ran")
                .infer()
        },
        || {
            workload
                .borrow_mut()
                .as_mut()
                .expect("build phase ran")
                .post_process()
        },
        || {
            let mut workload = workload.borrow_mut();
            let workload = workload.as_mut().expect("build phase ran");
            workload.infer()?;
            workload.post_process()
        },
        gpu_memory_mib,
    )?;
    let engine = engine.unwrap_or_else(|| model.engine_path());
    let document = serde_json::json!({
        "schema_version": 1,
        "pipeline": format!("{:?}", model.kind()).to_ascii_lowercase(),
        "precision": precision,
        "tracks": tracks,
        "scope": scope,
        "model": descriptor.display().to_string(),
        "model_sha256": sha256_file(descriptor)?,
        "engine": engine.display().to_string(),
        "engine_sha256": sha256_file(engine)?,
        "revision": option_env!("GIT_COMMIT"),
        "environment": {
            "target": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "cuda_path": std::env::var("CUDA_PATH").ok(),
            "tensorrt_root": std::env::var("TENSORRT_ROOT_DIR").ok(),
        },
        "report": report.to_json(),
    });
    let mut bytes = serde_json::to_vec_pretty(&document)?;
    bytes.push(b'\n');
    if let Some(output) = output {
        fs::write(output, &bytes)?;
    }
    print!("{}", String::from_utf8(bytes)?);
    Ok(())
}

enum Workload {
    Regression {
        backend: TensorRtRegressionBackend,
        inputs: Vec<(usize, RegressionFrameInput)>,
        outputs: Vec<Vec<f32>>,
    },
    Diffusion {
        backend: TensorRtDiffusionBackend,
        inputs: Vec<(usize, DiffusionFrameInput)>,
        outputs: Vec<Vec<f32>>,
    },
    Emotion {
        backend: TensorRtClassifierBackend,
        inputs: Vec<(usize, Vec<f32>)>,
        processors: Vec<PostProcessor>,
        outputs: Vec<Vec<f32>>,
    },
    CpuBlendshape {
        solver: Box<CpuBlendshapeSolver>,
        target: Vec<f32>,
        output: Vec<f32>,
    },
    GpuBlendshape {
        solver: Box<GpuBlendshapeSolver>,
        target: DeviceBuffer<f32>,
        output: DeviceBuffer<f32>,
        host_output: Vec<f32>,
        stream: CudaStream,
        _device: Arc<GpuDevice>,
    },
    InteractiveGpuReplay {
        executor: Box<DeviceBlendshapeSolveInteractiveExecutor>,
        geometry: RegressionGeometry,
    },
}

impl Workload {
    fn load(model: &Model, engine: &Path, tracks: usize, scope: &str) -> Result<Self> {
        if scope != "raw-network" {
            return Self::load_blendshape(model, tracks, scope);
        }
        let device = GpuDevice::new(0)?;
        match (model.kind(), model.network()) {
            (ModelKind::Regression, NetworkDocument::Geometry(network)) => {
                let GeometryParameters::Regression(parameters) = &network.params else {
                    unreachable!()
                };
                let GeometryAudioParameters::Regression(audio) = &network.audio_params else {
                    unreachable!()
                };
                let contract = RegressionContract::new(parameters, audio, 30, 1)?;
                let inputs = (0..tracks)
                    .map(|track| {
                        (
                            track,
                            RegressionFrameInput {
                                timestamp: 0,
                                next_timestamp: 533,
                                audio: vec![0.0; contract.audio_size],
                                emotion: vec![0.0; contract.emotion_size],
                            },
                        )
                    })
                    .collect();
                Ok(Self::Regression {
                    backend: TensorRtRegressionBackend::load(device, engine, contract)?,
                    inputs,
                    outputs: Vec::new(),
                })
            }
            (ModelKind::Diffusion, NetworkDocument::Geometry(network)) => {
                let GeometryParameters::Diffusion(parameters) = &network.params else {
                    unreachable!()
                };
                let GeometryAudioParameters::Diffusion(audio) = &network.audio_params else {
                    unreachable!()
                };
                let contract = DiffusionContract::new(parameters, audio)?;
                let inputs = (0..tracks)
                    .map(|track| {
                        (
                            track,
                            DiffusionFrameInput {
                                audio: vec![0.0; contract.audio_size],
                                emotions: vec![0.0; contract.center_frames * contract.emotion_size],
                                identity: {
                                    let mut value = vec![0.0; contract.identity_size];
                                    value[track % contract.identity_size] = 1.0;
                                    value
                                },
                                noise: vec![0.0; contract.noise_size().unwrap_or(0)],
                                input_latents: vec![0.0; contract.state_size().unwrap_or(0)],
                            },
                        )
                    })
                    .collect();
                Ok(Self::Diffusion {
                    backend: TensorRtDiffusionBackend::load(device, engine, contract)?,
                    inputs,
                    outputs: Vec::new(),
                })
            }
            (ModelKind::Emotion, NetworkDocument::Emotion(network)) => {
                let ModelParameters::Emotion(config) = model.parameters(0)? else {
                    unreachable!()
                };
                let data = PostProcessData {
                    inference_emotion_length: network.emotions.len(),
                    output_emotion_length: config.output_emotion_length,
                    emotion_correspondence: network
                        .emotions
                        .iter()
                        .map(|name| {
                            config
                                .emotion_correspondence
                                .get(name)
                                .copied()
                                .ok_or_else(|| {
                                    Error::InvalidSchema(format!(
                                        "emotion correspondence is missing for {name}"
                                    ))
                                })
                                .and_then(|value| {
                                    i32::try_from(value).map_err(|_| {
                                        Error::InvalidSchema(format!(
                                            "emotion correspondence is out of range: {value}"
                                        ))
                                    })
                                })
                        })
                        .collect::<Result<Vec<_>>>()?,
                };
                let parameters = PostProcessParams {
                    emotion_contrast: config.emotion_contrast,
                    max_emotions: config.max_emotions,
                    beginning_emotion: vec![0.0; config.output_emotion_length],
                    preferred_emotion: config.preferred_emotion.clone(),
                    live_blend_coefficient: config.live_blend_coef,
                    enable_preferred_emotion: config.enable_preferred_emotion,
                    preferred_emotion_strength: config.preferred_emotion_strength,
                    live_transition_time: config.transition_smoothing,
                    fixed_dt: config.fixed_dt,
                    emotion_strength: config.emotion_strength,
                };
                let contract = ClassifierContract::new(
                    60_000,
                    network.audio_params.samplerate,
                    network.emotions.len(),
                    30,
                    1,
                    0,
                )?;
                Ok(Self::Emotion {
                    backend: TensorRtClassifierBackend::load(device, engine, contract.clone())?,
                    inputs: (0..tracks)
                        .map(|track| (track, vec![0.0; contract.buffer_length]))
                        .collect(),
                    processors: (0..tracks)
                        .map(|_| PostProcessor::new(data.clone(), parameters.clone()))
                        .collect::<Result<Vec<_>>>()?,
                    outputs: Vec::new(),
                })
            }
            _ => unreachable!(),
        }
    }

    fn load_blendshape(model: &Model, tracks: usize, scope: &str) -> Result<Self> {
        if tracks != 1 {
            return Err(Error::InvalidSchema(
                "BlendShape benchmark scopes currently require one track".into(),
            ));
        }
        if model.kind() == ModelKind::Emotion {
            return Err(Error::InvalidSchema(
                "BlendShape benchmark scopes require a geometry model".into(),
            ));
        }
        let paths = model
            .blendshape_paths(0)?
            .get("skin")
            .ok_or_else(|| Error::InvalidSchema("model has no skin BlendShape data".into()))?;
        let data = BlendshapeData::load_npz(&paths.data)?;
        let target = data.evaluate_pose(&vec![0.25; data.pose_count()])?;
        let config = load_blendshape_config(&paths.config)?.blendshape_params;
        match scope {
            "blendshape-cpu" => {
                let pose_names = data
                    .pose_names
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                let solver = create_blendshape_solver(BlendshapeSolveComponentParameters {
                    params: BlendshapeSolverParams {
                        l1_regularization: config.l1_regularization,
                        l2_regularization: config.l2_regularization,
                        symmetry_regularization: config.symmetry_regularization,
                        temporal_regularization: config.temporal_regularization,
                        template_bounding_box_size: config.template_bb_size,
                        tolerance: config.tolerance,
                    },
                    config: BlendshapeSolverConfigView {
                        active_poses: &config.active_poses,
                        cancel_poses: &config.cancel_poses,
                        symmetry_poses: &config.symmetry_poses,
                        multipliers: &config.multipliers,
                        offsets: &config.offsets,
                    },
                    data: BlendshapeSolverDataView {
                        neutral_pose: &data.neutral_pose,
                        delta_poses: &data.delta_poses,
                        pose_mask: data.pose_mask.as_deref(),
                        pose_names: &pose_names,
                    },
                })?;
                Ok(Self::CpuBlendshape {
                    solver: Box::new(solver),
                    target,
                    output: Vec::new(),
                })
            }
            "blendshape-gpu" | "interactive-gpu-replay" => {
                let device = GpuDevice::new(0)?;
                let stream = device.create_stream()?;
                let solver = GpuBlendshapeSolver::new(&device, &stream, data, &config)?;
                if scope == "interactive-gpu-replay" {
                    let layer = InteractiveGpuBlendshapeLayer::new(
                        Arc::clone(&device),
                        stream,
                        Some(solver),
                        None,
                        2,
                    )?;
                    let geometry = RegressionGeometry {
                        skin: target,
                        tongue: Vec::new(),
                        jaw_transform: [0.0; 16],
                        eyes_rotation: audio2face3d::animation::EyesRotation {
                            right: [0.0; 3],
                            left: [0.0; 3],
                        },
                    };
                    let mut executor = DeviceBlendshapeSolveInteractiveExecutor::from_layer(
                        layer,
                        model.sample_rate(),
                        audio2face3d::audio2x::FrameRate::new(30, 1)?,
                    );
                    block_on(executor.compute_frame(0, 1, &geometry, |_| true))?;
                    Ok(Self::InteractiveGpuReplay {
                        executor: Box::new(executor),
                        geometry,
                    })
                } else {
                    let mut target_device = device.allocate(target.len())?;
                    target_device.copy_from(&target, &stream)?;
                    let output_count = solver.pose_count();
                    Ok(Self::GpuBlendshape {
                        solver: Box::new(solver),
                        target: target_device,
                        output: device.allocate(output_count)?,
                        host_output: vec![0.0; output_count],
                        stream,
                        _device: device,
                    })
                }
            }
            _ => Err(Error::InvalidSchema(format!(
                "unknown benchmark scope `{scope}`"
            ))),
        }
    }

    fn infer(&mut self) -> Result<()> {
        match self {
            Self::Regression {
                backend,
                inputs,
                outputs,
            } => *outputs = backend.infer_batch(inputs)?,
            Self::Diffusion {
                backend,
                inputs,
                outputs,
            } => {
                *outputs = backend
                    .infer_batch(inputs)?
                    .into_iter()
                    .map(|value| value.prediction)
                    .collect()
            }
            Self::Emotion {
                backend,
                inputs,
                outputs,
                ..
            } => *outputs = backend.infer_batch(inputs)?,
            Self::CpuBlendshape {
                solver,
                target,
                output,
            } => *output = solver.solve(target)?,
            Self::GpuBlendshape {
                solver,
                target,
                output,
                stream,
                ..
            } => solver.solve_async(target, output, stream)?.synchronize()?,
            Self::InteractiveGpuReplay { executor, geometry } => {
                block_on(executor.compute_frame(0, 1, geometry, |_| true))?;
            }
        }
        Ok(())
    }

    fn post_process(&mut self) -> Result<()> {
        match self {
            Self::Emotion {
                processors,
                outputs,
                ..
            } => {
                for (processor, output) in processors.iter_mut().zip(outputs.iter()) {
                    processor.process(output)?;
                }
            }
            Self::Regression { outputs, .. } | Self::Diffusion { outputs, .. } => {
                let _: f32 = outputs.iter().flatten().copied().sum();
            }
            Self::CpuBlendshape { output, .. } => {
                std::hint::black_box(output.iter().copied().sum::<f32>());
            }
            Self::GpuBlendshape {
                output,
                host_output,
                stream,
                ..
            } => output.copy_to(host_output, stream)?,
            Self::InteractiveGpuReplay { executor, .. } => {
                std::hint::black_box(executor.layer().copy_cached_frame_to_host(0)?);
            }
        }
        Ok(())
    }
}

fn gpu_memory_mib() -> Option<u64> {
    query_memory(
        "--query-compute-apps=used_memory",
        "--format=csv,noheader,nounits",
    )
    .or_else(|| query_memory("--query-gpu=memory.used", "--format=csv,noheader,nounits"))
}

fn query_memory(query: &str, format: &str) -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([query, format])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .filter_map(|line| line.trim().parse::<u64>().ok())
                .max()
        })
        .flatten()
}
