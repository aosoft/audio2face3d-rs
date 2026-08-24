use audio2x::RuntimeDiscovery;
use audio2x_model_tool::{
    DownloadReceipt, MODEL_PRESETS, ModelDownloadRequest, ModelPreset, model_preset,
};
use std::path::{Path, PathBuf};

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
            let name = arguments.next().ok_or("missing model preset")?;
            let output_root = arguments
                .next()
                .map_or_else(|| PathBuf::from(DEFAULT_OUTPUT_ROOT), PathBuf::from);
            let token_environment = arguments
                .next()
                .unwrap_or_else(|| DEFAULT_TOKEN_ENVIRONMENT.into());
            reject_extra_arguments(&mut arguments)?;
            if name == "all" {
                for preset in MODEL_PRESETS {
                    download_preset(*preset, &output_root, &token_environment)?;
                }
            } else {
                let preset = model_preset(&name).ok_or_else(|| {
                    format!("unknown model preset `{name}`; run `audio2x-model list`")
                })?;
                download_preset(preset, &output_root, &token_environment)?;
            }
        }
        Some("download-revision") => {
            let repository = arguments.next().ok_or("missing repository")?;
            let revision = arguments.next().ok_or("missing revision")?;
            let output = arguments.next().ok_or("missing output directory")?;
            let token_environment = arguments
                .next()
                .unwrap_or_else(|| DEFAULT_TOKEN_ENVIRONMENT.into());
            reject_extra_arguments(&mut arguments)?;
            let receipt = ModelDownloadRequest {
                repository,
                revision,
                output: PathBuf::from(output),
                token_environment,
            }
            .execute()?;
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
) -> Result<(), Box<dyn std::error::Error>> {
    println!("downloading {} ({})", preset.name, preset.repository);
    let receipt = preset.request(output_root, token_environment).execute()?;
    print_receipt(&receipt);
    Ok(())
}

fn print_receipt(receipt: &DownloadReceipt) {
    println!("model: {}", receipt.output.display());
    println!("revision: {}", receipt.revision);
    println!("network.onnx sha256: {}", receipt.network_onnx_sha256);
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
    "usage: audio2x-model doctor | list | download <preset|all> [output-root] [token-env] | download-revision <owner/repository> <40-character-revision> <output> [token-env]"
}
