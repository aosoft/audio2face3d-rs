use audio2x::{ModelDownloadRequest, RuntimeDiscovery};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    match arguments.next().as_deref() {
        Some("doctor") => {
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
            ModelDownloadRequest {
                repository,
                revision,
                output: PathBuf::from(output),
                token_environment,
            }
            .execute()?;
        }
        _ => {
            return Err(
                "usage: audio2x-model doctor | download <repository> <revision> <output> [token-env]"
                    .into(),
            );
        }
    }
    Ok(())
}
