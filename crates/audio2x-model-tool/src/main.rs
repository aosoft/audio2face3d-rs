use audio2x::RuntimeDiscovery;
use audio2x_model_tool::ModelDownloadRequest;
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("audio2x-model: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    match arguments.next().as_deref() {
        Some("doctor") => {
            reject_extra_arguments(&mut arguments)?;
            let runtime = RuntimeDiscovery::discover();
            println!("{}", runtime.diagnostic());
            if !runtime.is_ready() {
                std::process::exit(2);
            }
        }
        Some("download") => {
            let repository = arguments.next().ok_or("missing repository")?;
            let revision = arguments.next().ok_or("missing revision")?;
            let output = arguments.next().ok_or("missing output directory")?;
            let token_environment = arguments.next().unwrap_or_else(|| "HF_TOKEN".into());
            reject_extra_arguments(&mut arguments)?;
            let receipt = ModelDownloadRequest {
                repository,
                revision,
                output: PathBuf::from(output),
                token_environment,
            }
            .execute()?;
            println!("model: {}", receipt.output.display());
            println!("revision: {}", receipt.revision);
            println!("network.onnx sha256: {}", receipt.network_onnx_sha256);
        }
        _ => return Err(usage().into()),
    }
    Ok(())
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
    "usage: audio2x-model doctor | download <owner/repository> <40-character-revision> <output> [token-env]"
}
