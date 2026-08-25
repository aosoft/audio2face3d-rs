use audio2face3d::tensorrt::{EngineBuildRequest, EngineBuilder};
use std::path::PathBuf;

pub fn run(
    onnx: PathBuf,
    engine: PathBuf,
    device_id: u32,
    force: bool,
    extra_args: Vec<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let executable = std::env::var_os("TRTEXEC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("trtexec"));
    EngineBuilder {
        executable,
        replace_existing: force,
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
