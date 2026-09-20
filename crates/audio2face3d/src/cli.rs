#[cfg(feature = "native")]
mod async_util;
#[cfg(feature = "native")]
mod benchmark_command;
mod library;
mod logging;
mod progress;
mod raw_engine;
#[cfg(feature = "native")]
mod reference_runtime;
#[cfg(feature = "native")]
mod run_diffusion;
#[cfg(feature = "native")]
mod run_emotion;
#[cfg(feature = "native")]
mod run_regression;

use crate::cli::library::{
    DownloadDisposition, DownloadOptions, DownloadReceipt, EngineBuildDisposition,
    EngineBuildReceipt, EnginePrecision, MODEL_PRESETS, ModelDownloadRequest,
    ModelEngineBuildRequest, ModelPreset, model_preset,
};

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use progress::TerminalDownloadProgress;

const DEFAULT_OUTPUT_ROOT: &str = "models";
const DEFAULT_TOKEN_ENVIRONMENT: &str = "HF_TOKEN";

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
    runtime: audio2face3d::runtime::cli::NativeRuntimeArgs,
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

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// List the built-in immutable model presets.
    List,
    /// Download and verify one preset or the complete catalog.
    Download(DownloadCommand),
    /// Download an arbitrary immutable Hugging Face model revision.
    DownloadRevision(DownloadRevisionCommand),
    /// Generate a TensorRT engine from an installed preset.
    Engine(EngineCommand),
    /// Generate a TensorRT engine from an arbitrary ONNX file.
    EngineOnnx(RawEngineCommand),
    /// Download a preset and then generate its TensorRT engine.
    Prepare(PrepareCommand),
}

#[derive(Args, Debug)]
struct RawEngineCommand {
    /// Source ONNX model.
    onnx: PathBuf,
    /// TensorRT engine output path.
    engine: PathBuf,
    /// CUDA device ordinal used by trtexec.
    #[arg(long = "device", default_value_t = 0)]
    device_id: u32,
    /// Atomically replace an existing engine after a successful build.
    #[arg(long)]
    force: bool,
    /// Additional arguments passed to trtexec after `--`.
    #[arg(last = true, allow_hyphen_values = true)]
    trtexec_options: Vec<String>,
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

#[derive(Args, Debug)]
struct DownloadCommand {
    /// Built-in model preset to download.
    #[arg(value_enum)]
    preset: PresetSelection,
    /// Directory under which preset directories are installed.
    #[arg(default_value = DEFAULT_OUTPUT_ROOT)]
    output_root: PathBuf,
    /// Name of the environment variable containing the Hugging Face token.
    #[arg(default_value = DEFAULT_TOKEN_ENVIRONMENT)]
    token_environment: String,
    /// Replace an installed snapshot after validating the new download.
    #[arg(long)]
    force: bool,
}

#[derive(Args, Debug)]
struct DownloadRevisionCommand {
    /// Hugging Face repository in owner/name form.
    repository: String,
    /// Full 40-character immutable commit revision.
    revision: String,
    /// Directory in which to install the model snapshot.
    output: PathBuf,
    /// Name of the environment variable containing the Hugging Face token.
    #[arg(default_value = DEFAULT_TOKEN_ENVIRONMENT)]
    token_environment: String,
    /// Replace an installed snapshot after validating the new download.
    #[arg(long)]
    force: bool,
}

#[derive(Args, Debug)]
struct EngineCommand {
    /// Built-in model preset whose installed ONNX should be converted.
    #[arg(value_enum)]
    preset: PresetSelection,
    /// Directory containing the installed preset directories.
    #[arg(default_value = DEFAULT_OUTPUT_ROOT)]
    output_root: PathBuf,
    #[command(flatten)]
    options: EngineOptions,
}

#[derive(Args, Debug)]
struct PrepareCommand {
    /// Built-in model preset to download and convert.
    #[arg(value_enum)]
    preset: PresetSelection,
    /// Directory under which preset directories are prepared.
    #[arg(default_value = DEFAULT_OUTPUT_ROOT)]
    output_root: PathBuf,
    /// Name of the environment variable containing the Hugging Face token.
    #[arg(default_value = DEFAULT_TOKEN_ENVIRONMENT)]
    token_environment: String,
    /// Engine options; `--force` replaces both matching snapshot and engine artifacts.
    #[command(flatten)]
    engine: EngineOptions,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum PresetSelection {
    Diffusion,
    Claire,
    James,
    Mark,
    Emotion,
    All,
}

impl PresetSelection {
    const fn name(self) -> Option<&'static str> {
        match self {
            Self::Diffusion => Some("diffusion"),
            Self::Claire => Some("claire"),
            Self::James => Some("james"),
            Self::Mark => Some("mark"),
            Self::Emotion => Some("emotion"),
            Self::All => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum PrecisionArgument {
    #[default]
    Default,
    Fp16,
    Fp32,
}

impl From<PrecisionArgument> for EnginePrecision {
    fn from(value: PrecisionArgument) -> Self {
        match value {
            PrecisionArgument::Default => Self::Default,
            PrecisionArgument::Fp16 => Self::Fp16,
            PrecisionArgument::Fp32 => Self::Fp32,
        }
    }
}

#[derive(Args, Clone, Copy, Debug)]
struct EngineOptions {
    /// TensorRT precision policy and output artifact suffix.
    #[arg(long, value_enum, default_value = "default")]
    precision: PrecisionArgument,
    /// CUDA device ordinal used by trtexec.
    #[arg(long = "device", default_value_t = 0)]
    device_id: u32,
    /// Strict maximum batch; omit to enable memory-driven fallback.
    #[arg(long = "max-batch")]
    max_batch_size: Option<NonZeroU64>,
    /// Atomically replace existing artifacts after a successful build.
    #[arg(long)]
    force: bool,
}

pub fn run() {
    let cli = Cli::parse();
    let result = (|| {
        let logger = logging::StderrLogger::from_env()?;
        let context = audio2face3d::Audio2Face3DContext::builder()
            .logger(std::sync::Arc::new(logger))
            .native_runtime(cli.runtime.resolve()?)
            .build();
        audio2face3d::logging::integration::LogScope::new(context).in_scope(|| execute(cli))
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
        Command::Model { command } => run_model(command)?,
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

fn run_model(command: ModelCommand) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        ModelCommand::List => {
            for preset in MODEL_PRESETS {
                println!(
                    "{:<9} {:<42} {}",
                    preset.name, preset.repository, preset.output_directory
                );
            }
        }
        ModelCommand::Download(command) => {
            for preset in selected_presets(command.preset) {
                download_preset(
                    preset,
                    &command.output_root,
                    &command.token_environment,
                    command.force,
                )?;
            }
        }
        ModelCommand::DownloadRevision(command) => {
            let receipt = ModelDownloadRequest {
                repository: command.repository,
                revision: command.revision,
                output: command.output,
                token_environment: command.token_environment,
            };
            let receipt = execute_download(&receipt, command.force)?;
            print_receipt(&receipt);
        }
        ModelCommand::Engine(command) => {
            for preset in selected_presets(command.preset) {
                build_preset_engine(preset, &command.output_root, command.options)?;
            }
        }
        ModelCommand::EngineOnnx(command) => raw_engine::run(
            command.onnx,
            command.engine,
            command.device_id,
            command.force,
            command.trtexec_options,
        )?,
        ModelCommand::Prepare(command) => {
            for preset in selected_presets(command.preset) {
                download_preset(
                    preset,
                    &command.output_root,
                    &command.token_environment,
                    command.engine.force,
                )?;
                build_preset_engine(preset, &command.output_root, command.engine)?;
            }
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

fn selected_presets(selection: PresetSelection) -> Vec<ModelPreset> {
    match selection.name() {
        Some(name) => vec![model_preset(name).expect("CLI presets must exist in the catalog")],
        None => MODEL_PRESETS.to_vec(),
    }
}

fn download_preset(
    preset: ModelPreset,
    output_root: &Path,
    token_environment: &str,
    force: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("checking {} ({})", preset.name, preset.repository);
    let request = preset.request(output_root, token_environment);
    let receipt = execute_download(&request, force)?;
    print_receipt(&receipt);
    Ok(())
}

fn execute_download(
    request: &ModelDownloadRequest,
    force: bool,
) -> Result<DownloadReceipt, Box<dyn std::error::Error>> {
    let progress = Arc::new(TerminalDownloadProgress::new());
    let result = request.execute_with_options(
        DownloadOptions { force },
        Some(Arc::clone(&progress).into()),
    );
    if result.is_err() {
        progress.failed();
    }
    Ok(result?)
}

fn print_receipt(receipt: &DownloadReceipt) {
    let status = match receipt.disposition {
        DownloadDisposition::Downloaded => "downloaded",
        DownloadDisposition::VerifiedExisting => "verified; skipped",
        DownloadDisposition::Replaced => "replaced",
    };
    println!("model: {} ({status})", receipt.output.display());
    println!("revision: {}", receipt.revision);
    println!("network.onnx sha256: {}", receipt.network_onnx_sha256);
}

fn build_preset_engine(
    preset: ModelPreset,
    output_root: &Path,
    options: EngineOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_directory = output_root.join(preset.output_directory);
    let precision = options.precision.into();
    println!("building {} engine ({precision})", preset.name);
    let trtexec = std::env::var_os("TRTEXEC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("trtexec"));
    let receipt = ModelEngineBuildRequest {
        model_directory,
        precision,
        device_id: options.device_id,
        max_batch_size: options.max_batch_size.map(NonZeroU64::get),
        replace: options.force,
        trtexec,
    }
    .execute()?;
    print_engine_receipt(&receipt);
    Ok(())
}

fn print_engine_receipt(receipt: &EngineBuildReceipt) {
    let status = match receipt.disposition {
        EngineBuildDisposition::Built => "built",
        EngineBuildDisposition::VerifiedExisting => "verified; skipped",
        EngineBuildDisposition::Replaced => "replaced",
    };
    println!("engine: {} ({status})", receipt.engine.display());
    println!("precision: {}", receipt.precision);
    if let Some(max_batch_size) = receipt.max_batch_size {
        let policy = if receipt.automatic_batch_size {
            "automatic"
        } else {
            "explicit"
        };
        println!("max batch size: {max_batch_size} ({policy})");
    }
    if receipt.attempted_max_batch_sizes.len() > 1 {
        println!(
            "batch attempts: {}",
            receipt
                .attempted_max_batch_sizes
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(" -> ")
        );
    }
    println!("engine sha256: {}", receipt.engine_sha256);
    println!("model descriptor: {}", receipt.model_descriptor.display());
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
    fn engine_help_describes_generation_options() {
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("model")
            .unwrap()
            .find_subcommand_mut("engine")
            .unwrap()
            .render_long_help()
            .to_string();
        assert!(help.contains("--precision"));
        assert!(help.contains("default, fp16, fp32"));
        assert!(help.contains("--device"));
        assert!(help.contains("--max-batch"));
        assert!(help.contains("--force"));
    }

    #[test]
    fn parses_existing_engine_argument_form() {
        let cli = Cli::try_parse_from([
            "audio2face3d",
            "model",
            "engine",
            "mark",
            "custom-models",
            "--precision=fp16",
            "--device=2",
            "--max-batch=16",
            "--force",
        ])
        .unwrap();

        let Command::Model {
            command: ModelCommand::Engine(command),
        } = cli.command
        else {
            panic!("expected engine command");
        };
        assert_eq!(command.preset, PresetSelection::Mark);
        assert_eq!(command.output_root, PathBuf::from("custom-models"));
        assert_eq!(command.options.precision, PrecisionArgument::Fp16);
        assert_eq!(command.options.device_id, 2);
        assert_eq!(
            command.options.max_batch_size.map(NonZeroU64::get),
            Some(16)
        );
        assert!(command.options.force);
    }

    #[test]
    fn parses_download_defaults_with_force_before_positionals() {
        let cli =
            Cli::try_parse_from(["audio2face3d", "model", "download", "--force", "all"]).unwrap();

        let Command::Model {
            command: ModelCommand::Download(command),
        } = cli.command
        else {
            panic!("expected download command");
        };
        assert_eq!(command.preset, PresetSelection::All);
        assert_eq!(command.output_root, PathBuf::from(DEFAULT_OUTPUT_ROOT));
        assert_eq!(command.token_environment, DEFAULT_TOKEN_ENVIRONMENT);
        assert!(command.force);
    }

    #[test]
    fn cli_presets_stay_in_sync_with_the_catalog() {
        let cli_presets = PresetSelection::value_variants()
            .iter()
            .filter_map(|selection| selection.name())
            .collect::<Vec<_>>();
        let catalog_presets = MODEL_PRESETS
            .iter()
            .map(|preset| preset.name)
            .collect::<Vec<_>>();
        assert_eq!(cli_presets, catalog_presets);
        assert_eq!(selected_presets(PresetSelection::All), MODEL_PRESETS);
    }

    #[test]
    fn rejects_unknown_values_and_zero_max_batch() {
        for arguments in [
            vec!["audio2face3d", "model", "download", "unknown"],
            vec![
                "audio2face3d",
                "model",
                "engine",
                "mark",
                "--precision=half",
            ],
        ] {
            let error = Cli::try_parse_from(arguments).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidValue);
        }

        let error =
            Cli::try_parse_from(["audio2face3d", "model", "engine", "mark", "--max-batch=0"])
                .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn version_flag_uses_package_version() {
        let error = Cli::try_parse_from(["audio2face3d", "--version"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayVersion);
        assert!(error.to_string().contains(env!("CARGO_PKG_VERSION")));
    }
}
