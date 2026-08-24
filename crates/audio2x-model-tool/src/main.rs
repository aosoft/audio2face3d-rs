mod progress;

use audio2x::RuntimeDiscovery;
use audio2x_model_tool::{
    DownloadDisposition, DownloadOptions, DownloadReceipt, MODEL_PRESETS, ModelDownloadRequest,
    ModelPreset, model_preset,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use progress::TerminalDownloadProgress;

const DEFAULT_OUTPUT_ROOT: &str = "models";
const DEFAULT_TOKEN_ENVIRONMENT: &str = "HF_TOKEN";

fn main() {
    if let Err(error) = run() {
        eprintln!("audio2x-model: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    run_arguments(std::env::args().skip(1))
}

fn run_arguments(
    mut arguments: impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    match arguments.next().as_deref() {
        Some("doctor") => {
            reject_extra_arguments(&mut arguments)?;
            let runtime = RuntimeDiscovery::discover();
            println!("{}", runtime.diagnostic());
            if !runtime.is_ready() {
                std::process::exit(2);
            }
        }
        Some("list") => {
            reject_extra_arguments(&mut arguments)?;
            for preset in MODEL_PRESETS {
                println!(
                    "{:<9} {:<42} {}",
                    preset.name, preset.repository, preset.output_directory
                );
            }
        }
        Some("download") => {
            let mut values = arguments.collect::<Vec<_>>();
            let force = take_force(&mut values)?;
            let name = values.first().ok_or("missing model preset")?;
            let output_root = values
                .get(1)
                .map_or_else(|| PathBuf::from(DEFAULT_OUTPUT_ROOT), PathBuf::from);
            let token_environment = values
                .get(2)
                .cloned()
                .unwrap_or_else(|| DEFAULT_TOKEN_ENVIRONMENT.into());
            if values.len() > 3 {
                return Err("unexpected extra argument".into());
            }
            if name == "all" {
                for preset in MODEL_PRESETS {
                    download_preset(*preset, &output_root, &token_environment, force)?;
                }
            } else {
                let preset = model_preset(name).ok_or_else(|| {
                    format!("unknown model preset `{name}`; run `audio2x-model list`")
                })?;
                download_preset(preset, &output_root, &token_environment, force)?;
            }
        }
        Some("download-revision") => {
            let mut values = arguments.collect::<Vec<_>>();
            let force = take_force(&mut values)?;
            let repository = values.first().ok_or("missing repository")?.clone();
            let revision = values.get(1).ok_or("missing revision")?.clone();
            let output = values.get(2).ok_or("missing output directory")?.clone();
            let token_environment = values
                .get(3)
                .cloned()
                .unwrap_or_else(|| DEFAULT_TOKEN_ENVIRONMENT.into());
            if values.len() > 4 {
                return Err("unexpected extra argument".into());
            }
            let receipt = ModelDownloadRequest {
                repository,
                revision,
                output: PathBuf::from(output),
                token_environment,
            };
            let receipt = execute_download(&receipt, force)?;
            print_receipt(&receipt);
        }
        _ => return Err(usage().into()),
    }
    Ok(())
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

fn take_force(arguments: &mut Vec<String>) -> Result<bool, &'static str> {
    let count = arguments
        .iter()
        .filter(|argument| argument.as_str() == "--force")
        .count();
    if count > 1 {
        return Err("--force was specified more than once");
    }
    arguments.retain(|argument| argument != "--force");
    Ok(count == 1)
}

fn reject_extra_arguments(
    arguments: &mut impl Iterator<Item = String>,
) -> Result<(), &'static str> {
    if arguments.next().is_some() {
        Err("unexpected extra argument")
    } else {
        Ok(())
    }
}

fn usage() -> &'static str {
    "usage: audio2x-model doctor | list | download <preset|all> [output-root] [token-env] [--force] | download-revision <owner/repository> <40-character-revision> <output> [token-env] [--force]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_flag_is_removed_from_any_argument_position() {
        let mut arguments = vec!["diffusion".into(), "--force".into(), "models".into()];
        assert!(take_force(&mut arguments).unwrap());
        assert_eq!(arguments, ["diffusion", "models"]);
    }

    #[test]
    fn duplicate_force_flag_is_rejected() {
        let mut arguments = vec!["--force".into(), "diffusion".into(), "--force".into()];
        assert!(take_force(&mut arguments).is_err());
    }
}
