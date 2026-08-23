use audio2x_inference::{EngineBuildRequest, EngineBuilder};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("audio2x-engine: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let onnx = arguments.next().map(PathBuf::from).ok_or(
        "usage: audio2x-engine <model.onnx> <network.trt> [--device=N] [--replace] [trtexec options...]",
    )?;
    let engine = arguments
        .next()
        .map(PathBuf::from)
        .ok_or("missing engine output path")?;
    let mut device_id = 0;
    let mut replace_existing = false;
    let mut extra_args = Vec::new();
    for argument in arguments {
        let argument = argument.to_string_lossy();
        if argument == "--replace" {
            replace_existing = true;
        } else if let Some(value) = argument.strip_prefix("--device=") {
            device_id = value.parse()?;
        } else {
            extra_args.push(argument.into_owned());
        }
    }
    let executable = std::env::var_os("TRTEXEC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("trtexec"));
    EngineBuilder {
        executable,
        replace_existing,
    }
    .build(&EngineBuildRequest {
        onnx,
        engine,
        device_id,
        profiles: Vec::new(),
        extra_args,
    })?;
    Ok(())
}
