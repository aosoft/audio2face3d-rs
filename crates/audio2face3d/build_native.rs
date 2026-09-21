//! Build-time SDK discovery using the shared platform.toml format.
#[path = "src/platform_config_file.rs"]
mod config_file;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct BuildConfig {
    pub cuda_root: PathBuf,
    pub tensorrt_root: Option<PathBuf>,
    pub cuda_host_compiler: Option<PathBuf>,
}

fn absolute(base: &Path, value: &str) -> PathBuf {
    let value = PathBuf::from(value);
    if value.is_absolute() {
        value
    } else {
        base.join(value)
    }
}
fn string(table: &toml::Table, name: &str) -> Result<Option<String>, String> {
    table
        .get(name)
        .map(|v| {
            v.as_str()
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .ok_or_else(|| format!("{name} must be a nonempty string"))
        })
        .transpose()
}
pub fn parse_config(text: &str, file: &Path, tensorrt: bool) -> Result<BuildConfig, String> {
    let table =
        config_file::parse(text, "build").map_err(|e| format!("{}: {e}", file.display()))?;
    let base = file.parent().ok_or("configuration file has no parent")?;
    let cuda_root = absolute(
        base,
        &string(&table, "cuda-root")?.ok_or("cuda-root is required")?,
    );
    let tensorrt_root = string(&table, "tensorrt-root")?.map(|p| absolute(base, &p));
    if tensorrt && tensorrt_root.is_none() {
        return Err("tensorrt-root is required for TensorRT".into());
    }
    Ok(BuildConfig {
        cuda_root,
        tensorrt_root,
        cuda_host_compiler: string(&table, "cuda-host-compiler")?.map(|p| absolute(base, &p)),
    })
}

/// Does not search the consumer's current working directory.
pub fn config_candidates(
    manifest: &Path,
    user: Option<PathBuf>,
    explicit: Option<PathBuf>,
) -> Vec<PathBuf> {
    if let Some(path) = explicit {
        return vec![path];
    }
    let mut candidates = Vec::new();
    if let Some(root) = manifest.parent().and_then(Path::parent)
        && root.join("Cargo.toml").is_file()
        && let Ok(actual_manifest) = manifest.canonicalize()
        && root.join("crates/audio2face3d").canonicalize().ok() == Some(actual_manifest)
        && fs::read_to_string(root.join("Cargo.toml")).is_ok_and(|s| s.contains("[workspace]"))
    {
        candidates.push(root.join("platform.toml"));
    }
    if let Some(user) = user {
        candidates.push(user);
    }
    candidates
}

pub fn first_installed_root(
    label: &str,
    mut candidates: Vec<PathBuf>,
    marker: &str,
) -> Result<PathBuf, String> {
    candidates.retain(|p| p.join(marker).is_file());
    candidates.sort();
    candidates.dedup();
    if candidates.is_empty() {
        return Err(format!(
            "specify {label} in platform.toml (found {} SDK candidates)",
            candidates.len()
        ));
    }
    Ok(candidates.remove(0))
}
fn installed_cuda() -> Result<PathBuf, String> {
    #[cfg(windows)]
    let bases = env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .map(|p| p.join("NVIDIA GPU Computing Toolkit/CUDA"))
        .into_iter()
        .collect::<Vec<_>>();
    #[cfg(not(windows))]
    let bases = vec![PathBuf::from("/usr/local")];
    let mut paths = Vec::new();
    for base in bases {
        println!("cargo:rerun-if-changed={}", base.display());
        if let Ok(entries) = fs::read_dir(base) {
            paths.extend(entries.filter_map(Result::ok).map(|e| e.path()));
        }
    }
    first_installed_root("cuda-root", paths, "include/cuda.h")
}
fn installed_tensorrt() -> Result<PathBuf, String> {
    #[cfg(windows)]
    let paths = env::var_os("ProgramFiles")
        .map(PathBuf::from)
        .and_then(|p| fs::read_dir(p).ok())
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("TensorRT"))
        })
        .collect();
    #[cfg(not(windows))]
    let paths = vec![PathBuf::from("/usr"), PathBuf::from("/usr/local/TensorRT")];
    first_installed_root("tensorrt-root", paths, "include/NvInfer.h")
}

pub fn resolve(tensorrt: bool) -> Result<BuildConfig, String> {
    if let Ok(version) = env::var("CUDARC_CUDA_VERSION")
        && version != "12090"
    {
        return Err("CUDARC_CUDA_VERSION conflicts with the required cuda-12090 bindings".into());
    }
    for name in [
        "AUDIO2FACE3D_PLATFORM_CONFIG",
        "CUDA_PATH",
        "TENSORRT_ROOT_DIR",
        "AUDIO2FACE3D_CUDA_HOST_COMPILER",
        "LOCALAPPDATA",
        "XDG_CONFIG_HOME",
        "HOME",
        "ProgramFiles",
        "CUDARC_CUDA_VERSION",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
    }
    let manifest =
        PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").ok_or("missing manifest directory")?);
    let explicit = env::var_os("AUDIO2FACE3D_PLATFORM_CONFIG")
        .map(|value| env::current_dir().map(|cwd| cwd.join(value)))
        .transpose()
        .map_err(|e| e.to_string())?;
    let candidates = config_candidates(&manifest, config_file::user_config(), explicit.clone());
    for path in &candidates {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    for path in candidates {
        if path.exists() || explicit.is_some() {
            let contents =
                fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            return parse_config(&contents, &path, tensorrt).and_then(validate);
        }
    }
    let cuda_root = match env::var_os("CUDA_PATH") {
        Some(v) => PathBuf::from(v),
        None => installed_cuda()?,
    };
    let tensorrt_root = match env::var_os("TENSORRT_ROOT_DIR") {
        Some(v) if tensorrt => Some(PathBuf::from(v)),
        _ if tensorrt => Some(installed_tensorrt()?),
        _ => None,
    };
    validate(BuildConfig {
        cuda_root,
        tensorrt_root,
        cuda_host_compiler: env::var_os("AUDIO2FACE3D_CUDA_HOST_COMPILER").map(PathBuf::from),
    })
}
fn validate(config: BuildConfig) -> Result<BuildConfig, String> {
    for (root, marker) in [
        (Some(&config.cuda_root), "include/cuda.h"),
        (config.tensorrt_root.as_ref(), "include/NvInfer.h"),
    ] {
        if let Some(root) = root {
            if !root.is_absolute() || !root.join(marker).is_file() {
                return Err(format!(
                    "invalid SDK root {}: expected {marker}",
                    root.display()
                ));
            }
            println!("cargo:rerun-if-changed={}", root.join(marker).display());
        }
    }
    if let Some(host) = &config.cuda_host_compiler
        && !host.is_file()
    {
        return Err(format!(
            "CUDA host compiler does not exist: {}",
            host.display()
        ));
    }
    Ok(config)
}

pub fn parse_define(text: &str, key: &str) -> Result<u32, String> {
    let defines = text
        .lines()
        .filter_map(|line| {
            let mut words = line.split_whitespace();
            if words.next()? != "#define" {
                return None;
            }
            Some((words.next()?, words.next()?))
        })
        .collect::<std::collections::HashMap<_, _>>();
    let mut value = key;
    for _ in 0..16 {
        value = value.trim_matches(['(', ')']);
        if let Ok(number) = value.parse() {
            return Ok(number);
        }
        value = defines
            .get(value)
            .copied()
            .ok_or_else(|| format!("missing numeric definition for {key}"))?;
    }
    Err(format!("cyclic or unsupported definition for {key}"))
}
pub fn header_version(file: &Path, key: &str) -> Result<u32, String> {
    println!("cargo:rerun-if-changed={}", file.display());
    let text = fs::read_to_string(file).map_err(|e| format!("{}: {e}", file.display()))?;
    parse_define(&text, key).map_err(|e| format!("{}: {e}", file.display()))
}
pub fn emit_versions(config: &BuildConfig) -> Result<(), String> {
    let cuda = header_version(&config.cuda_root.join("include/cuda.h"), "CUDA_VERSION")?;
    println!("cargo:rustc-env=AUDIO2FACE3D_BUILD_CUDA_VERSION={cuda}");
    if let Some(root) = &config.tensorrt_root {
        for part in ["MAJOR", "MINOR", "PATCH", "BUILD"] {
            let version = header_version(
                &root.join("include/NvInferVersion.h"),
                &format!("NV_TENSORRT_{part}"),
            )?;
            println!("cargo:rustc-env=AUDIO2FACE3D_BUILD_TENSORRT_{part}={version}");
        }
    }
    Ok(())
}
