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
    fs::rename(&temporary, &arguments.output).map_err(|error| error.to_string())?;
    Ok(())
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut sdk_root = env::var_os("AUDIO2FACE_SDK_ROOT").map(PathBuf::from);
    let mut output = PathBuf::from("reference/artifacts.json");
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
            Some("--help" | "-h") => {
                return Err("usage: audio2x-reference --sdk-root PATH [--output PATH]".to_owned());
            }
            _ => return Err(format!("unknown or incomplete argument: {argument:?}")),
        }
    }
    Ok(Arguments {
        sdk_root: sdk_root.ok_or("provide --sdk-root or AUDIO2FACE_SDK_ROOT")?,
        output,
    })
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
}
