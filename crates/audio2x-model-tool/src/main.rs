mod progress;

use audio2x::RuntimeDiscovery;
use audio2x_model_tool::{
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

/// Download Audio2X models and generate environment-specific TensorRT engines.
#[derive(Debug, Parser)]
#[command(name = "audio2x-model", version, about, arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: ModelCommand,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Diagnose CUDA and TensorRT runtime discovery.
    Doctor,
    /// List the built-in immutable model presets.
    List,
    /// Download and verify one preset or the complete catalog.
    Download(DownloadCommand),
    /// Download an arbitrary immutable Hugging Face model revision.
    DownloadRevision(DownloadRevisionCommand),
    /// Generate a TensorRT engine from an installed preset.
    Engine(EngineCommand),
    /// Download a preset and then generate its TensorRT engine.
    Prepare(PrepareCommand),
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
    /// Replace an installed snapshot after validating the new download.
    #[arg(long)]
    force: bool,
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
    replace: bool,
}

fn main() {
    if let Err(error) = run(Cli::parse()) {
        eprintln!("audio2x-model: {error}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        ModelCommand::Doctor => {
            let runtime = RuntimeDiscovery::discover();
            println!("{}", runtime.diagnostic());
            if !runtime.is_ready() {
                std::process::exit(2);
            }
        }
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
        ModelCommand::Prepare(command) => {
            for preset in selected_presets(command.preset) {
                download_preset(
                    preset,
                    &command.output_root,
                    &command.token_environment,
                    command.force,
                )?;
                build_preset_engine(preset, &command.output_root, command.engine)?;
            }
        }
    }
    Ok(())
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
        replace: options.replace,
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
        assert!(help.contains("Download Audio2X models"));
        assert!(help.contains("download-revision"));
        assert!(help.contains("prepare"));
        assert!(help.contains("--help"));
        assert!(help.contains("--version"));
    }

    #[test]
    fn engine_help_describes_generation_options() {
        let mut command = Cli::command();
        let help = command
            .find_subcommand_mut("engine")
            .unwrap()
            .render_long_help()
            .to_string();
        assert!(help.contains("--precision"));
        assert!(help.contains("default, fp16, fp32"));
        assert!(help.contains("--device"));
        assert!(help.contains("--max-batch"));
        assert!(help.contains("--replace"));
    }

    #[test]
    fn parses_existing_engine_argument_form() {
        let cli = Cli::try_parse_from([
            "audio2x-model",
            "engine",
            "mark",
            "custom-models",
            "--precision=fp16",
            "--device=2",
            "--max-batch=16",
            "--replace",
        ])
        .unwrap();

        let ModelCommand::Engine(command) = cli.command else {
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
        assert!(command.options.replace);
    }

    #[test]
    fn parses_download_defaults_with_force_before_positionals() {
        let cli = Cli::try_parse_from(["audio2x-model", "download", "--force", "all"]).unwrap();

        let ModelCommand::Download(command) = cli.command else {
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
            vec!["audio2x-model", "download", "unknown"],
            vec!["audio2x-model", "engine", "mark", "--precision=half"],
        ] {
            let error = Cli::try_parse_from(arguments).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidValue);
        }

        let error =
            Cli::try_parse_from(["audio2x-model", "engine", "mark", "--max-batch=0"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn version_flag_uses_package_version() {
        let error = Cli::try_parse_from(["audio2x-model", "--version"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::DisplayVersion);
        assert!(error.to_string().contains(env!("CARGO_PKG_VERSION")));
    }
}
