use crate::common::Result;
use serde_json::{Value, json};
use std::time::Instant;

#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
use crate::animation::{
    DiffusionBackend, DiffusionContract, DiffusionFrameInput, RegressionBackend,
    RegressionContract, RegressionFrameInput, TensorRtDiffusionBackend, TensorRtRegressionBackend,
};
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
use crate::audio2emotion::post_process::{PostProcessData, PostProcessParams, PostProcessor};
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
use crate::common::{Error, GeometryAudioParameters, GeometryParameters, NetworkDocument};
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
use crate::cuda::GpuDevice;
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
use crate::emotion::{ClassifierBackend, ClassifierContract, TensorRtClassifierBackend};
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
use crate::{Model, ModelKind, ModelParameters};
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Percentiles {
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkPhase {
    pub name: String,
    pub iterations: usize,
    pub percentiles: Percentiles,
    /// Completed iterations per second across the complete measured phase.
    pub throughput_per_second: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkReport {
    pub warmup_iterations: usize,
    pub phases: Vec<BenchmarkPhase>,
    pub peak_memory_mib: Option<u64>,
}

impl BenchmarkReport {
    pub fn to_json(&self) -> Value {
        json!({
            "warmup_iterations": self.warmup_iterations,
            "peak_memory_mib": self.peak_memory_mib,
            "phases": self.phases.iter().map(|phase| json!({
                "name": phase.name,
                "iterations": phase.iterations,
                "p50_ns": phase.percentiles.p50_ns,
                "p95_ns": phase.percentiles.p95_ns,
                "p99_ns": phase.percentiles.p99_ns,
                "throughput_per_second": phase.throughput_per_second,
            })).collect::<Vec<_>>(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkRunner {
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measured_iterations: 100,
        }
    }
}

impl BenchmarkRunner {
    pub fn run<Build, Cache, Steady, Post, EndToEnd, Memory>(
        &self,
        mut build: Build,
        mut cache: Cache,
        mut steady: Steady,
        mut post_process: Post,
        mut end_to_end: EndToEnd,
        mut memory: Memory,
    ) -> Result<BenchmarkReport>
    where
        Build: FnMut() -> Result<()>,
        Cache: FnMut() -> Result<()>,
        Steady: FnMut() -> Result<()>,
        Post: FnMut() -> Result<()>,
        EndToEnd: FnMut() -> Result<()>,
        Memory: FnMut() -> Option<u64>,
    {
        if self.warmup_iterations == 0 || self.measured_iterations == 0 {
            return Err(crate::common::Error::InvalidSchema(
                "benchmark warm-up and measured iterations must be non-zero".into(),
            ));
        }
        let mut peak = memory();
        let build = measure("build", 1, &mut build, &mut memory, &mut peak)?;
        let cache = measure("cache", 1, &mut cache, &mut memory, &mut peak)?;
        let warmup = measure(
            "warmup",
            self.warmup_iterations,
            &mut steady,
            &mut memory,
            &mut peak,
        )?;
        let steady = measure(
            "steady-state",
            self.measured_iterations,
            &mut steady,
            &mut memory,
            &mut peak,
        )?;
        let post = measure(
            "post-process",
            self.measured_iterations,
            &mut post_process,
            &mut memory,
            &mut peak,
        )?;
        let e2e = measure(
            "end-to-end",
            self.measured_iterations,
            &mut end_to_end,
            &mut memory,
            &mut peak,
        )?;
        Ok(BenchmarkReport {
            warmup_iterations: self.warmup_iterations,
            phases: vec![build, cache, warmup, steady, post, e2e],
            peak_memory_mib: peak,
        })
    }
}

/// Opaque raw TensorRT workload used by the companion CLI benchmark.
///
/// This type keeps implementation-only Backend/Postprocessor SPI inside the
/// library while preserving separate inference and post-processing timing
/// phases. It is not an executor construction or extension point.
#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
pub struct RawNetworkBenchmark {
    inner: RawNetworkBenchmarkInner,
}

#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
enum RawNetworkBenchmarkInner {
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
}

#[cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]
impl RawNetworkBenchmark {
    pub fn load(model: &Model, engine: &Path, tracks: usize) -> Result<Self> {
        let device = GpuDevice::new(0)?;
        let inner = match (model.kind(), model.network()) {
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
                RawNetworkBenchmarkInner::Regression {
                    backend: TensorRtRegressionBackend::load(device, engine, contract)?,
                    inputs,
                    outputs: Vec::new(),
                }
            }
            (ModelKind::Diffusion, NetworkDocument::Geometry(network)) => {
                let GeometryParameters::Diffusion(parameters) = &network.params else {
                    unreachable!()
                };
                let GeometryAudioParameters::Diffusion(audio) = &network.audio_params else {
                    unreachable!()
                };
                let contract = DiffusionContract::new(parameters, audio)?;
                let noise_size = contract.noise_size()?;
                let state_size = contract.state_size()?;
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
                                noise: vec![0.0; noise_size],
                                input_latents: vec![0.0; state_size],
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                RawNetworkBenchmarkInner::Diffusion {
                    backend: TensorRtDiffusionBackend::load(device, engine, contract)?,
                    inputs,
                    outputs: Vec::new(),
                }
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
                RawNetworkBenchmarkInner::Emotion {
                    backend: TensorRtClassifierBackend::load(device, engine, contract.clone())?,
                    inputs: (0..tracks)
                        .map(|track| (track, vec![0.0; contract.buffer_length]))
                        .collect(),
                    processors: (0..tracks)
                        .map(|_| PostProcessor::new(data.clone(), parameters.clone()))
                        .collect::<Result<Vec<_>>>()?,
                    outputs: Vec::new(),
                }
            }
            _ => unreachable!(),
        };
        Ok(Self { inner })
    }

    pub fn infer(&mut self) -> Result<()> {
        match &mut self.inner {
            RawNetworkBenchmarkInner::Regression {
                backend,
                inputs,
                outputs,
            } => *outputs = backend.infer_batch(inputs)?,
            RawNetworkBenchmarkInner::Diffusion {
                backend,
                inputs,
                outputs,
            } => {
                *outputs = backend
                    .infer_batch(inputs)?
                    .into_iter()
                    .map(|value| value.prediction)
                    .collect();
            }
            RawNetworkBenchmarkInner::Emotion {
                backend,
                inputs,
                outputs,
                ..
            } => *outputs = backend.infer_batch(inputs)?,
        }
        Ok(())
    }

    pub fn post_process(&mut self) -> Result<()> {
        match &mut self.inner {
            RawNetworkBenchmarkInner::Emotion {
                processors,
                outputs,
                ..
            } => {
                for (processor, output) in processors.iter_mut().zip(outputs.iter()) {
                    processor.process(output)?;
                }
            }
            RawNetworkBenchmarkInner::Regression { outputs, .. }
            | RawNetworkBenchmarkInner::Diffusion { outputs, .. } => {
                std::hint::black_box(outputs.iter().flatten().copied().sum::<f32>());
            }
        }
        Ok(())
    }
}

fn measure<F, M>(
    name: &str,
    iterations: usize,
    function: &mut F,
    memory: &mut M,
    peak: &mut Option<u64>,
) -> Result<BenchmarkPhase>
where
    F: FnMut() -> Result<()>,
    M: FnMut() -> Option<u64>,
{
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        function()?;
        samples.push(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
        if let Some(current) = memory() {
            *peak = Some(peak.unwrap_or(0).max(current));
        }
    }
    samples.sort_unstable();
    let total_ns = samples.iter().map(|value| u128::from(*value)).sum::<u128>();
    let throughput_per_second = iterations as f64 * 1_000_000_000.0 / total_ns.max(1) as f64;
    Ok(BenchmarkPhase {
        name: name.into(),
        iterations,
        percentiles: Percentiles {
            p50_ns: percentile(&samples, 50),
            p95_ns: percentile(&samples, 95),
            p99_ns: percentile(&samples, 99),
        },
        throughput_per_second,
    })
}

fn percentile(samples: &[u64], percentile: usize) -> u64 {
    let index = samples
        .len()
        .saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1)
        .min(samples.len().saturating_sub(1));
    samples[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_all_phases_and_calculates_percentiles() {
        let runner = BenchmarkRunner {
            warmup_iterations: 2,
            measured_iterations: 4,
        };
        let report = runner
            .run(
                || Ok(()),
                || Ok(()),
                || Ok(()),
                || Ok(()),
                || Ok(()),
                || Some(42),
            )
            .unwrap();
        assert_eq!(
            report
                .phases
                .iter()
                .map(|phase| phase.name.as_str())
                .collect::<Vec<_>>(),
            [
                "build",
                "cache",
                "warmup",
                "steady-state",
                "post-process",
                "end-to-end"
            ]
        );
        assert_eq!(report.peak_memory_mib, Some(42));
        assert_eq!(report.to_json()["phases"].as_array().unwrap().len(), 6);
        assert!(
            report
                .phases
                .iter()
                .all(|phase| phase.throughput_per_second.is_finite())
        );
    }
}
