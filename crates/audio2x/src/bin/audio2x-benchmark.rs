use audio2emotion::{
    ClassifierBackend, ClassifierContract, EmotionPostProcessData, EmotionPostProcessor,
    TensorRtClassifierBackend,
};
use audio2face::{
    DiffusionBackend, DiffusionContract, DiffusionFrameInput, RegressionBackend,
    RegressionContract, RegressionFrameInput, TensorRtDiffusionBackend, TensorRtRegressionBackend,
};
use audio2x::core::{GeometryAudioParameters, GeometryParameters, NetworkDocument, Result};
use audio2x::cuda::GpuDevice;
use audio2x::{Audio2xModel, BenchmarkRunner, ModelKind, ModelParameters};
use std::cell::RefCell;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let descriptor = arguments.next().map(PathBuf::from).ok_or(
        "usage: audio2x-benchmark <model.json> <tracks> <precision> [iterations] [engine]",
    )?;
    let tracks = parse_usize(arguments.next(), "tracks")?.unwrap_or(1);
    let precision = arguments
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "fp32".into());
    if precision != "fp32" && precision != "fp16" {
        return Err("precision must be fp32 or fp16".into());
    }
    let iterations = parse_usize(arguments.next(), "iterations")?.unwrap_or(20);
    let model = Audio2xModel::load(&descriptor)?;
    let engine = arguments.next().map(PathBuf::from);
    let workload = RefCell::new(None);
    let report = BenchmarkRunner {
        warmup_iterations: 5,
        measured_iterations: iterations,
    }
    .run(
        || {
            *workload.borrow_mut() = Some(Workload::load(
                &model,
                engine.as_deref().unwrap_or_else(|| model.engine_path()),
                tracks,
            )?);
            Ok(())
        },
        || {
            Audio2xModel::load(&descriptor)?;
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
    let output = serde_json::json!({
        "schema_version": 1,
        "pipeline": format!("{:?}", model.kind()).to_ascii_lowercase(),
        "precision": precision,
        "tracks": tracks,
        "engine": engine.as_deref().unwrap_or_else(|| model.engine_path()).display().to_string(),
        "report": report.to_json(),
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
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
        processors: Vec<EmotionPostProcessor>,
        outputs: Vec<Vec<f32>>,
    },
}

impl Workload {
    fn load(model: &Audio2xModel, engine: &Path, tracks: usize) -> Result<Self> {
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
                let (data, parameters) = EmotionPostProcessData::from_model(network, config)?;
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
                        .map(|_| EmotionPostProcessor::new(data.clone(), parameters.clone()))
                        .collect::<Result<Vec<_>>>()?,
                    outputs: Vec::new(),
                })
            }
            _ => unreachable!(),
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
        }
        Ok(())
    }
}

fn parse_usize(value: Option<std::ffi::OsString>, name: &str) -> Result<Option<usize>, String> {
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse()
                .map_err(|_| format!("{name} must be an integer"))
        })
        .transpose()
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
