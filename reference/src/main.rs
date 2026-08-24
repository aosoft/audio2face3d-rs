use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const CASE_ROOTS: &[(&str, &[&str])] = &[
    (
        "regression",
        &[
            "audio2face-sdk/samples/data/mark",
            "audio2face-sdk/tests/data",
            "audio2x-common/tests/data",
        ],
    ),
    (
        "diffusion",
        &["audio2face-sdk/samples/data/multi-diffusion"],
    ),
    (
        "a2e",
        &[
            "audio2emotion-sdk/samples/model",
            "audio2emotion-sdk/tests/data",
        ],
    ),
];

struct Arguments {
    sdk_root: PathBuf,
    output: PathBuf,
    generate_fp16: Option<PathBuf>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments()?;
    let generated = arguments.sdk_root.join("_data/generated");
    if !generated.is_dir() {
        return Err(format!(
            "generated fixture directory not found: {}",
            generated.display()
        ));
    }
    if let Some(model_directory) = &arguments.generate_fp16 {
        generate_fp16_fixture(model_directory)?;
    }

    let mut cases = serde_json::Map::new();
    for (name, roots) in CASE_ROOTS {
        let mut files = Vec::new();
        for relative in *roots {
            let root = generated.join(relative);
            if !root.is_dir() {
                return Err(format!("fixture case not found: {}", root.display()));
            }
            for mut entry in collect_files(&root).map_err(|error| error.to_string())? {
                let path = entry
                    .get("path")
                    .and_then(Value::as_str)
                    .ok_or("generated file entry has no path")?;
                entry["path"] = json!(format!("{relative}/{path}"));
                files.push(entry);
            }
        }
        cases.insert(
            (*name).to_owned(),
            json!({ "roots": roots, "files": files }),
        );
    }

    let document = json!({
        "schema_version": 1,
        "sources": {
            "audio2face_sdk": git_revision(&arguments.sdk_root)?,
        },
        "environment": {
            "os": format!("{}-{}", env::consts::OS, env::consts::ARCH),
            "rust": command_output("rustc", &["--version"]),
            "cuda": command_output("nvcc", &["--version"]),
            "gpu": command_output(
                "nvidia-smi",
                &[
                    "--query-gpu=name,driver_version,compute_cap",
                    "--format=csv,noheader",
                ],
            ),
        },
        "cases": cases,
    });

    let payload = serde_json::to_vec_pretty(&document).map_err(|error| error.to_string())?;
    let temporary = arguments.output.with_extension(format!(
        "{}.tmp",
        arguments
            .output
            .extension()
            .and_then(OsStr::to_str)
            .unwrap_or("json")
    ));
    let mut stream = File::create(&temporary).map_err(|error| error.to_string())?;
    stream
        .write_all(&payload)
        .and_then(|()| stream.write_all(b"\n"))
        .map_err(|error| error.to_string())?;
    drop(stream);
    replace_file(&temporary, &arguments.output).map_err(|error| error.to_string())?;
    Ok(())
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut sdk_root = env::var_os("AUDIO2FACE_SDK_ROOT").map(PathBuf::from);
    let mut output = PathBuf::from("reference/artifacts.json");
    let mut generate_fp16 = None;
    let mut arguments = env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--sdk-root") => sdk_root = arguments.next().map(PathBuf::from),
            Some("--output") => {
                output = arguments
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--output requires a path")?;
            }
            Some("--generate-fp16") => {
                generate_fp16 = Some(
                    arguments
                        .next()
                        .map(PathBuf::from)
                        .ok_or("--generate-fp16 requires a model directory")?,
                );
            }
            Some("--help" | "-h") => {
                return Err(
                    "usage: audio2x-reference --sdk-root PATH [--output PATH] [--generate-fp16 MODEL-DIRECTORY]"
                        .to_owned(),
                );
            }
            _ => return Err(format!("unknown or incomplete argument: {argument:?}")),
        }
    }
    Ok(Arguments {
        sdk_root: sdk_root.ok_or("provide --sdk-root or AUDIO2FACE_SDK_ROOT")?,
        output,
        generate_fp16,
    })
}

fn generate_fp16_fixture(model_directory: &Path) -> Result<(), String> {
    let onnx_path = model_directory.join("network.onnx");
    let model_path = model_directory.join("model.json");
    let trt_info_path = model_directory.join("trt_info.json");
    for path in [&onnx_path, &model_path, &trt_info_path] {
        if !path.is_file() {
            return Err(format!("FP16 fixture input not found: {}", path.display()));
        }
    }

    let engine_path = model_directory.join("network_fp16.trt");
    let model_fp16_path = model_directory.join("model_fp16.json");
    let trt_info_fp16_path = model_directory.join("trt_info_fp16.json");
    for path in [&engine_path, &model_fp16_path, &trt_info_fp16_path] {
        if path.exists() {
            return Err(format!(
                "refusing to replace existing FP16 fixture: {}",
                path.display()
            ));
        }
    }

    let mut trt_info = read_json(&trt_info_path)?;
    trt_info["trt_build_param"]["fp16"] = json!(["--fp16"]);
    let build_arguments = trt_build_arguments(&trt_info)?;
    let temporary_engine =
        model_directory.join(format!("network_fp16.trt.partial-{}", std::process::id()));
    let trtexec = env::var_os("TRTEXEC").unwrap_or_else(|| "trtexec".into());
    let status = Command::new(trtexec)
        .arg(format!("--onnx={}", onnx_path.display()))
        .arg(format!("--saveEngine={}", temporary_engine.display()))
        .args(&build_arguments)
        .status()
        .map_err(|error| format!("unable to launch trtexec: {error}"))?;
    if !status.success() {
        let _ = fs::remove_file(&temporary_engine);
        return Err(format!(
            "trtexec failed while generating FP16 fixture: {status}"
        ));
    }
    if temporary_engine
        .metadata()
        .map_err(|error| error.to_string())?
        .len()
        == 0
    {
        let _ = fs::remove_file(&temporary_engine);
        return Err("trtexec generated an empty FP16 engine".to_owned());
    }
    fs::rename(&temporary_engine, &engine_path).map_err(|error| error.to_string())?;

    let mut model = read_json(&model_path)?;
    model["networkPath"] = json!("network_fp16.trt");
    write_json_new(&trt_info_fp16_path, &trt_info)?;
    write_json_new(&model_fp16_path, &model)?;
    Ok(())
}

fn read_json(path: &Path) -> Result<Value, String> {
    let payload = fs::read(path).map_err(|error| error.to_string())?;
    serde_json::from_slice(&payload).map_err(|error| format!("{}: {error}", path.display()))
}

fn write_json_new(path: &Path, value: &Value) -> Result<(), String> {
    let temporary = PathBuf::from(format!("{}.partial-{}", path.display(), std::process::id()));
    let mut payload = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    payload.push(b'\n');
    let mut stream = File::create(&temporary).map_err(|error| error.to_string())?;
    stream
        .write_all(&payload)
        .and_then(|()| stream.sync_all())
        .map_err(|error| error.to_string())?;
    drop(stream);
    fs::rename(&temporary, path).map_err(|error| error.to_string())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    fs::rename(source, destination)
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> io::Result<()> {
    use std::os::windows::ffi::OsStrExt;

    unsafe extern "system" {
        fn MoveFileExW(existing: *const u16, destination: *const u16, flags: u32) -> i32;
    }
    const MOVEFILE_REPLACE_EXISTING: u32 = 0x1;
    const MOVEFILE_WRITE_THROUGH: u32 = 0x8;
    let source = source
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    let destination = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect::<Vec<_>>();
    // SAFETY: both path buffers are NUL-terminated UTF-16 strings that remain
    // alive for the duration of the Win32 call.
    if unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    } == 0
    {
        Err(io::Error::last_os_error())
    } else {
        Ok(())
    }
}

fn trt_build_arguments(trt_info: &Value) -> Result<Vec<String>, String> {
    let defaults = trt_info
        .get("defaults")
        .and_then(Value::as_object)
        .ok_or("trt_info.json has no defaults object")?;
    let groups = trt_info
        .get("trt_build_param")
        .and_then(Value::as_object)
        .ok_or("trt_info.json has no trt_build_param object")?;
    let mut arguments = Vec::new();
    for values in groups.values() {
        let values = values
            .as_array()
            .ok_or("trt_build_param entries must be arrays")?;
        for value in values {
            let mut argument = value
                .as_str()
                .ok_or("trt_build_param arguments must be strings")?
                .to_owned();
            for (name, replacement) in defaults {
                let replacement = replacement
                    .as_u64()
                    .ok_or("trt_info defaults must be unsigned integers")?
                    .to_string();
                argument = argument.replace(&format!("{{{name}}}"), &replacement);
            }
            arguments.push(argument);
        }
    }
    Ok(arguments)
}

fn collect_files(root: &Path) -> io::Result<Vec<Value>> {
    let mut paths = Vec::new();
    visit_files(root, &mut paths)?;
    paths.sort();
    paths
        .into_iter()
        .filter(|path| is_fixture(path))
        .map(|path| {
            let relative = path
                .strip_prefix(root)
                .expect("visited paths are below their root")
                .to_string_lossy()
                .replace('\\', "/");
            Ok(json!({
                "path": relative,
                "bytes": path.metadata()?.len(),
                "sha256": sha256(&path)?,
            }))
        })
        .collect()
}

fn visit_files(root: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            visit_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn is_fixture(path: &Path) -> bool {
    matches!(
        path.extension().and_then(OsStr::to_str),
        Some("bin" | "json" | "npz" | "onnx" | "trt" | "wav")
    )
}

fn sha256(path: &Path) -> io::Result<String> {
    let mut stream = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let digest = digest.finalize();
    Ok(format!("{digest:x}"))
}

fn git_revision(root: &Path) -> Result<String, String> {
    let output = Command::new("git")
        .args([
            OsStr::new("-C"),
            root.as_os_str(),
            OsStr::new("rev-parse"),
            OsStr::new("HEAD"),
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    match Command::new(program).args(arguments).output() {
        Ok(output) => format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .trim()
        .to_owned(),
        Err(error) => format!("unavailable: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_sha256_matches() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let actual = sha256(&path).unwrap();
        assert_eq!(actual.len(), 64);
        assert!(actual.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    #[test]
    fn fixture_filter_is_strict() {
        assert!(is_fixture(Path::new("tensor.npz")));
        assert!(is_fixture(Path::new("network.trt")));
        assert!(!is_fixture(Path::new("notes.txt")));
        assert!(!is_fixture(Path::new("NETWORK.ONNX")));
    }

    #[test]
    fn expands_fp16_trtexec_arguments_from_sdk_metadata() {
        let info = json!({
            "trt_build_param": {
                "batch": [
                    "--minShapes=input:1x8320",
                    "--optShapes=input:{OPT_BATCH_SIZE}x8320",
                    "--maxShapes=input:{MAX_BATCH_SIZE}x8320"
                ],
                "fp16": ["--fp16"]
            },
            "defaults": { "OPT_BATCH_SIZE": 8, "MAX_BATCH_SIZE": 128 }
        });
        assert_eq!(
            trt_build_arguments(&info).unwrap(),
            [
                "--minShapes=input:1x8320",
                "--optShapes=input:8x8320",
                "--maxShapes=input:128x8320",
                "--fp16"
            ]
        );
    }
}
