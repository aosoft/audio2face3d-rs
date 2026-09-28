#[cfg(feature = "native")]
mod async_util;
#[cfg(feature = "native")]
mod benchmark_command;
mod library;
use audio2face3d::cli_logging as logging;
#[cfg(feature = "native")]
mod reference_runtime;
#[cfg(feature = "native")]
mod run_diffusion;
#[cfg(feature = "native")]
mod run_emotion;
#[cfg(feature = "native")]
mod run_regression;

use audio2face3d::model_management::cli::ModelCommand;
use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// Manage models and run the unofficial Rust Audio2Face-3D implementation.
#[derive(Debug, Parser)]
#[command(
    name = "audio2face3d",
    version,
    about = "Model, reference, benchmark, and runtime CLI",
    arg_required_else_help = true
)]
struct Cli {
    #[command(flatten)]
    logging: logging::LogArgs,
    #[command(flatten)]
    runtime: audio2face3d::runtime::cli::PlatformArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Diagnose CUDA and TensorRT runtime discovery.
    Doctor {
        /// Load native libraries and report actual versions.
        #[arg(long)]
        load: bool,
        /// Also create a CUDA device (requires --load).
        #[arg(long, requires = "load")]
        device: Option<i32>,
        /// Emit machine-readable observations and selected roots.
        #[arg(long)]
        json: bool,
    },
    /// Download models and generate TensorRT engines.
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Run an Audio2Face-3D inference pipeline.
    #[cfg(feature = "native")]
    Run {
        #[command(subcommand)]
        command: RunCommand,
    },
    /// Measure runtime inference and post-processing phases.
    #[cfg(feature = "native")]
    Benchmark(BenchmarkCommand),
    /// Prepare and compare implementation-neutral reference artifacts.
    Reference {
        #[command(subcommand)]
        command: ReferenceCommand,
    },
    /// Validate release contracts and performance baselines.
    Release {
        #[command(subcommand)]
        command: ReleaseCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ReleaseCommand {
    /// Validate an explicitly supplied release contract and its workspace artifacts.
    Audit(ReleaseAuditCommand),
    /// Compare a benchmark JSON report independently from numeric parity.
    BenchmarkCompare(BenchmarkCompareCommand),
}

#[derive(Args, Debug)]
struct ReleaseAuditCommand {
    #[arg(long)]
    baseline: PathBuf,
    #[arg(long, default_value = ".")]
    workspace: PathBuf,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct BenchmarkCompareCommand {
    baseline: PathBuf,
    candidate: PathBuf,
    #[arg(long, default_value_t = 15.0)]
    maximum_latency_regression_percent: f64,
    #[arg(long, default_value_t = 15.0)]
    maximum_throughput_regression_percent: f64,
    #[arg(long, default_value_t = 10.0)]
    maximum_memory_regression_percent: f64,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum ReferenceCommand {
    /// Decode or generate a canonical mono 16 kHz f32 fixture.
    Fixture {
        #[command(subcommand)]
        command: FixtureCommand,
    },
    /// Compare two captured artifact directories.
    Compare(CompareReferenceCommand),
    /// Capture a Rust runtime artifact from a canonical fixture.
    #[cfg(feature = "native")]
    Capture(CaptureReferenceCommand),
}

#[derive(Debug, Subcommand)]
enum FixtureCommand {
    /// Decode a mono PCM16 16 kHz WAV once and pin both source and sample hashes.
    Wav(WavFixtureCommand),
    /// Generate deterministic silence or a deterministic two-tone signal.
    Generated(GeneratedFixtureCommand),
}

#[derive(Args, Debug)]
struct WavFixtureCommand {
    input: PathBuf,
    output: PathBuf,
    #[arg(long, default_value = "speech")]
    name: String,
    /// SPDX expression or a project-local license/provenance identifier.
    #[arg(long)]
    license: String,
    /// Optional expected hash of the source WAV.
    #[arg(long)]
    sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum GeneratedFixtureKind {
    Silence,
    Synthetic,
}

#[derive(Args, Debug)]
struct GeneratedFixtureCommand {
    output: PathBuf,
    #[arg(value_enum)]
    kind: GeneratedFixtureKind,
    #[arg(long, default_value_t = 4)]
    seconds: usize,
}

#[derive(Args, Debug)]
struct CompareReferenceCommand {
    expected: PathBuf,
    actual: PathBuf,
    #[arg(long, default_value = "reference/tolerances.json")]
    tolerances: PathBuf,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ReferenceExecution {
    Standard,
    InteractiveRandom,
    InteractiveAll,
    InteractiveBlendshapeRandom,
    InteractiveBlendshapeAll,
    BlendshapeCpu,
    BlendshapeGpu,
    TeethStandalone,
}

#[cfg(feature = "native")]
#[derive(Args, Debug)]
struct CaptureReferenceCommand {
    model: PathBuf,
    fixture: PathBuf,
    output: PathBuf,
    #[arg(long, value_enum, default_value = "standard")]
    execution: ReferenceExecution,
    #[arg(long, default_value = "fp32")]
    precision: String,
    #[arg(long, default_value_t = 1)]
    tracks: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    frame: Option<usize>,
}

#[cfg(feature = "native")]
#[derive(Debug, Subcommand)]
enum RunCommand {
    /// Run the regression animation pipeline.
    Regression(PipelineCommand),
    /// Run the diffusion animation pipeline.
    Diffusion(PipelineCommand),
    /// Run the Audio2Emotion pipeline.
    Emotion(PipelineCommand),
}

#[cfg(feature = "native")]
#[derive(Args, Debug)]
struct PipelineCommand {
    /// Model descriptor JSON.
    model: PathBuf,
    /// Number of concurrent tracks.
    #[arg(default_value_t = 1)]
    tracks: usize,
    /// Number of zero-valued audio samples; defaults to one second.
    samples: Option<usize>,
}

#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BenchmarkPrecision {
    Fp32,
    Fp16,
}

#[cfg(feature = "native")]
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BenchmarkScope {
    RawNetwork,
    BlendshapeCpu,
    BlendshapeGpu,
    InteractiveGpuReplay,
}

#[cfg(feature = "native")]
impl BenchmarkScope {
    const fn as_str(self) -> &'static str {
        match self {
            Self::RawNetwork => "raw-network",
            Self::BlendshapeCpu => "blendshape-cpu",
            Self::BlendshapeGpu => "blendshape-gpu",
            Self::InteractiveGpuReplay => "interactive-gpu-replay",
        }
    }
}

#[cfg(feature = "native")]
impl BenchmarkPrecision {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
        }
    }
}

#[cfg(feature = "native")]
#[derive(Args, Debug)]
struct BenchmarkCommand {
    /// Model descriptor JSON.
    model: PathBuf,
    /// Number of concurrent tracks.
    #[arg(default_value_t = 1)]
    tracks: usize,
    /// Precision label recorded in the report.
    #[arg(value_enum, default_value = "fp32")]
    precision: BenchmarkPrecision,
    /// Timed runtime boundary.
    #[arg(long, value_enum, default_value = "raw-network")]
    scope: BenchmarkScope,
    /// Number of measured iterations.
    #[arg(default_value_t = 20)]
    iterations: usize,
    /// Optional engine override.
    engine: Option<PathBuf>,
    /// Also write the reproducible JSON result to this path.
    #[arg(long)]
    output: Option<PathBuf>,
}

pub fn run() {
    let cli = Cli::parse();
    let result = (|| {
        let logging = logging::Logging::start(&cli.logging, env!("CARGO_PKG_NAME"))
            .map_err(|e| e as Box<dyn std::error::Error>)?;
        let context = audio2face3d::Audio2Face3DContext::builder()
            .logger(logging.logger.clone())
            .native_runtime(cli.runtime.resolve()?)
            .build();
        let outcome =
            audio2face3d::logging::integration::LogScope::new(context).in_scope(|| execute(cli));
        logging.finish()?;
        outcome
    })();
    if let Err(error) = result {
        eprintln!("audio2face3d: {error}");
        std::process::exit(1);
    }
}

fn execute(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Command::Doctor { load, device, json } => {
            let scope = audio2face3d::logging::integration::LogScope::capture();
            let context = scope.context();
            let discovered = context.native_runtime().discover()?;
            let info = if load {
                #[cfg(feature = "cuda")]
                {
                    let mut info = context.initialize_native()?;
                    if let Some(ordinal) = device {
                        let _device = audio2face3d::cuda::GpuDevice::new_with_context(
                            ordinal,
                            context.clone(),
                        )?;
                        info = context.native_runtime_info().unwrap();
                    }
                    info
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = device;
                    return Err("doctor --load requires the cuda/native feature".into());
                }
            } else {
                discovered
            };
            if json {
                println!(
                    "{}",
                    serde_json::json!({"state":format!("{:?}",info.state()),"cuda_root":context.native_runtime().cuda_root(),"tensorrt_root":context.native_runtime().tensorrt_root(),"libraries":info.libraries().iter().map(|library|serde_json::json!({"name":library.name(),"path":library.path(),"build_version":library.build_version().map(|v|v.to_string()),"runtime_version":library.runtime_version().map(|v|v.to_string())})).collect::<Vec<_>>() })
                );
                return Ok(());
            }
            println!("state: {:?}", info.state());
            for library in info.libraries() {
                println!(
                    "{}: {} (build: {}, runtime: {})",
                    library.name(),
                    library.path().display(),
                    library
                        .build_version()
                        .map_or_else(|| "unknown".into(), |v| v.to_string()),
                    library
                        .runtime_version()
                        .map_or_else(|| "not loaded".into(), |v| v.to_string())
                );
            }
        }
        Command::Model { command } => audio2face3d::model_management::cli::run(command)
            .map_err(|e| e as Box<dyn std::error::Error>)?,
        #[cfg(feature = "native")]
        Command::Run { command } => run_pipeline(command)?,
        #[cfg(feature = "native")]
        Command::Benchmark(command) => benchmark_command::run(
            &command.model,
            command.tracks,
            command.precision.as_str(),
            command.scope.as_str(),
            command.iterations,
            command.engine.as_deref(),
            command.output.as_deref(),
        )?,
        Command::Reference { command } => run_reference(command)?,
        Command::Release { command } => run_release(command)?,
    }
    Ok(())
}

fn run_release(command: ReleaseCommand) -> Result<(), Box<dyn std::error::Error>> {
    use crate::cli::library::release;

    match command {
        ReleaseCommand::Audit(command) => {
            let report = release::audit_release_baseline(&command.workspace, &command.baseline)?;
            if let Some(path) = command.report {
                release::write_json(&path, &report)?;
            }
            println!("checks: {}", report.checks);
            for failure in &report.failures {
                println!("failure: {failure}");
            }
            if !report.compatible {
                return Err("release baseline audit failed".into());
            }
        }
        ReleaseCommand::BenchmarkCompare(command) => {
            let report = release::compare_benchmark(
                &command.baseline,
                &command.candidate,
                command.maximum_latency_regression_percent,
                command.maximum_throughput_regression_percent,
                command.maximum_memory_regression_percent,
            )?;
            if let Some(path) = command.report {
                release::write_json(&path, &report)?;
            }
            println!("case: {}", report.case);
            for regression in &report.regressions {
                println!("regression: {regression}");
            }
            if !report.compatible {
                return Err("benchmark regression detected".into());
            }
        }
    }
    Ok(())
}

fn run_reference(command: ReferenceCommand) -> Result<(), Box<dyn std::error::Error>> {
    use crate::cli::library::reference;

    match command {
        ReferenceCommand::Fixture { command } => {
            let manifest = match command {
                FixtureCommand::Wav(command) => reference::prepare_wav_fixture(
                    &command.input,
                    &command.output,
                    &command.name,
                    &command.license,
                    command.sha256.as_deref(),
                )?,
                FixtureCommand::Generated(command) => reference::prepare_generated_fixture(
                    &command.output,
                    match command.kind {
                        GeneratedFixtureKind::Silence => "silence",
                        GeneratedFixtureKind::Synthetic => "synthetic",
                    },
                    command.seconds,
                    command.kind == GeneratedFixtureKind::Synthetic,
                )?,
            };
            println!("samples: {}", manifest.sample_count);
            println!("samples sha256: {}", manifest.samples_sha256);
        }
        ReferenceCommand::Compare(command) => {
            let tolerances = reference::load_tolerance_profile(&command.tolerances)?;
            let report =
                reference::compare_artifacts(&command.expected, &command.actual, &tolerances)?;
            if let Some(path) = command.report {
                reference::write_comparison_report(&path, &report)?;
            }
            println!("records compared: {}", report.records_compared);
            println!("values compared: {}", report.values_compared);
            println!("max absolute error: {}", report.maximum_absolute_error);
            if let Some(difference) = &report.first_difference {
                println!(
                    "first difference: {}/{}[{}], expected {}, actual {}, allowed {}",
                    difference.layer,
                    difference.component,
                    difference.index,
                    difference.expected,
                    difference.actual,
                    difference.allowed_error
                );
            }
            for difference in &report.structural_differences {
                println!("structure: {difference}");
            }
            if !report.compatible {
                return Err("reference artifacts differ".into());
            }
        }
        #[cfg(feature = "native")]
        ReferenceCommand::Capture(command) => {
            reference_runtime::capture(reference_runtime::CaptureRequest {
                model_path: &command.model,
                fixture_root: &command.fixture,
                output: &command.output,
                execution: match command.execution {
                    ReferenceExecution::Standard => reference_runtime::Execution::Standard,
                    ReferenceExecution::InteractiveRandom => {
                        reference_runtime::Execution::InteractiveRandom
                    }
                    ReferenceExecution::InteractiveAll => {
                        reference_runtime::Execution::InteractiveAll
                    }
                    ReferenceExecution::InteractiveBlendshapeRandom => {
                        reference_runtime::Execution::InteractiveBlendshapeRandom
                    }
                    ReferenceExecution::InteractiveBlendshapeAll => {
                        reference_runtime::Execution::InteractiveBlendshapeAll
                    }
                    ReferenceExecution::BlendshapeCpu => {
                        reference_runtime::Execution::BlendshapeCpu
                    }
                    ReferenceExecution::BlendshapeGpu => {
                        reference_runtime::Execution::BlendshapeGpu
                    }
                    ReferenceExecution::TeethStandalone => {
                        reference_runtime::Execution::TeethStandalone
                    }
                },
                precision: &command.precision,
                tracks: command.tracks,
                seed: command.seed,
                selected_frame: command.frame,
            })?
        }
    }
    Ok(())
}

#[cfg(feature = "native")]
fn run_pipeline(command: RunCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        RunCommand::Regression(command) => {
            run_regression::run(&command.model, command.tracks, command.samples)
        }
        RunCommand::Diffusion(command) => {
            run_diffusion::run(&command.model, command.tracks, command.samples)
        }
        RunCommand::Emotion(command) => {
            run_emotion::run(&command.model, command.tracks, command.samples)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{CommandFactory, error::ErrorKind};

    #[test]
    fn top_level_help_describes_commands_and_standard_flags() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("Model, reference, benchmark, and runtime CLI"));
        assert!(help.contains("model"));
        assert!(help.contains("doctor"));
        #[cfg(feature = "native")]
        assert!(help.contains("run"));
        assert!(help.contains("--help"));
        assert!(help.contains("--version"));
    }

    #[test]
    fn version_flag_uses_package_version() {
        let error = Cli::try_parse_from(["audio2face3d", "--version"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayVersion);
        assert!(error.to_string().contains(env!("CARGO_PKG_VERSION")));
    }
}
