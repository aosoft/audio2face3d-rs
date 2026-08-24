mod progress;

use audio2x::RuntimeDiscovery;
use audio2x_model_tool::{
    DownloadDisposition, DownloadOptions, DownloadReceipt, EngineBuildDisposition,
    EngineBuildReceipt, EnginePrecision, MODEL_PRESETS, ModelDownloadRequest,
    ModelEngineBuildRequest, ModelPreset, model_preset,
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
            let force = take_switch(&mut values, "--force")?;
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
            let force = take_switch(&mut values, "--force")?;
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
        Some("engine") => {
            let mut values = arguments.collect::<Vec<_>>();
            let options = take_engine_options(&mut values)?;
            let name = values.first().ok_or("missing model preset")?;
            let output_root = values
                .get(1)
                .map_or_else(|| PathBuf::from(DEFAULT_OUTPUT_ROOT), PathBuf::from);
            if values.len() > 2 {
                return Err("unexpected extra argument".into());
            }
            for preset in selected_presets(name)? {
                build_preset_engine(preset, &output_root, options)?;
            }
        }
        Some("prepare") => {
            let mut values = arguments.collect::<Vec<_>>();
            let force = take_switch(&mut values, "--force")?;
            let engine_options = take_engine_options(&mut values)?;
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
            for preset in selected_presets(name)? {
                download_preset(preset, &output_root, &token_environment, force)?;
                build_preset_engine(preset, &output_root, engine_options)?;
            }
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn selected_presets(name: &str) -> Result<Vec<ModelPreset>, Box<dyn std::error::Error>> {
    if name == "all" {
        Ok(MODEL_PRESETS.to_vec())
    } else {
        Ok(vec![model_preset(name).ok_or_else(|| {
            format!("unknown model preset `{name}`; run `audio2x-model list`")
        })?])
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

#[derive(Clone, Copy)]
struct EngineOptions {
    precision: EnginePrecision,
    device_id: u32,
    max_batch_size: Option<u64>,
    replace: bool,
}

fn take_engine_options(
    arguments: &mut Vec<String>,
) -> Result<EngineOptions, Box<dyn std::error::Error>> {
    let replace = take_switch(arguments, "--replace")?;
    let precision = take_option(arguments, "--precision=")?
        .map_or(Ok(EnginePrecision::Default), |value| value.parse())?;
    let device_id = take_option(arguments, "--device=")?.map_or(Ok(0), |value| value.parse())?;
    let max_batch_size = take_option(arguments, "--max-batch=")?
        .map(|value| value.parse())
        .transpose()?;
    Ok(EngineOptions {
        precision,
        device_id,
        max_batch_size,
        replace,
    })
}

fn build_preset_engine(
    preset: ModelPreset,
    output_root: &Path,
    options: EngineOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_directory = output_root.join(preset.output_directory);
    println!("building {} engine ({})", preset.name, options.precision);
    let trtexec = std::env::var_os("TRTEXEC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("trtexec"));
    let receipt = ModelEngineBuildRequest {
        model_directory,
        precision: options.precision,
        device_id: options.device_id,
        max_batch_size: options.max_batch_size,
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
        println!("max batch size: {max_batch_size}");
    }
    println!("engine sha256: {}", receipt.engine_sha256);
    println!("model descriptor: {}", receipt.model_descriptor.display());
}

fn take_switch(arguments: &mut Vec<String>, flag: &'static str) -> Result<bool, String> {
    let count = arguments
        .iter()
        .filter(|argument| argument.as_str() == flag)
        .count();
    if count > 1 {
        return Err(format!("{flag} was specified more than once"));
    }
    arguments.retain(|argument| argument != flag);
    Ok(count == 1)
}

fn take_option(arguments: &mut Vec<String>, prefix: &str) -> Result<Option<String>, String> {
    let values = arguments
        .iter()
        .filter_map(|argument| argument.strip_prefix(prefix).map(str::to_owned))
        .collect::<Vec<_>>();
    if values.len() > 1 {
        return Err(format!(
            "{} was specified more than once",
            prefix.trim_end_matches('=')
        ));
    }
    arguments.retain(|argument| !argument.starts_with(prefix));
    match values.into_iter().next() {
        Some(value) if value.is_empty() => Err(format!("{prefix} requires a value")),
        value => Ok(value),
    }
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
    "usage: audio2x-model doctor | list | download <preset|all> [output-root] [token-env] [--force] | download-revision <owner/repository> <40-character-revision> <output> [token-env] [--force] | engine <preset|all> [output-root] [--precision=default|fp16|fp32] [--device=N] [--max-batch=N] [--replace] | prepare <preset|all> [output-root] [token-env] [--precision=default|fp16|fp32] [--device=N] [--max-batch=N] [--force] [--replace]"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_flag_is_removed_from_any_argument_position() {
        let mut arguments = vec!["diffusion".into(), "--force".into(), "models".into()];
        assert!(take_switch(&mut arguments, "--force").unwrap());
        assert_eq!(arguments, ["diffusion", "models"]);
    }

    #[test]
    fn duplicate_force_flag_is_rejected() {
        let mut arguments = vec!["--force".into(), "diffusion".into(), "--force".into()];
        assert!(take_switch(&mut arguments, "--force").is_err());
    }

    #[test]
    fn engine_options_are_removed_and_parsed() {
        let mut arguments = vec![
            "mark".into(),
            "--precision=fp16".into(),
            "--device=2".into(),
            "--max-batch=16".into(),
            "--replace".into(),
        ];
        let options = take_engine_options(&mut arguments).unwrap();
        assert_eq!(options.precision, EnginePrecision::Fp16);
        assert_eq!(options.device_id, 2);
        assert_eq!(options.max_batch_size, Some(16));
        assert!(options.replace);
        assert_eq!(arguments, ["mark"]);
    }
}
