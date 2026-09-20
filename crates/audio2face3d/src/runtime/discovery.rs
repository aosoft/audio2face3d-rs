use super::{
    NativeRuntimeConfig, NativeRuntimeError, NativeRuntimeErrorKind, NativeSearchPolicy,
    loader::LibraryFile,
};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
pub(crate) enum Sdk {
    Cuda,
    TensorRt,
}
pub(crate) fn directories(
    config: &NativeRuntimeConfig,
    sdk: Sdk,
) -> Result<Vec<PathBuf>, NativeRuntimeError> {
    let (root, explicit, variable) = match sdk {
        Sdk::Cuda => (config.cuda_root(), config.cuda_library_dirs(), "CUDA_PATH"),
        Sdk::TensorRt => (
            config.tensorrt_root(),
            config.tensorrt_library_dirs(),
            "TENSORRT_ROOT_DIR",
        ),
    };
    if let Some(root) = root {
        return root_directories(root);
    }
    if !explicit.is_empty() {
        return normalize(explicit.iter().cloned());
    }
    if config.search_policy() == NativeSearchPolicy::ExplicitOnly {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::InvalidConfig,
            format!("explicit runtime location is required for {variable}"),
        ));
    }
    if let Some(root) = std::env::var_os(variable) {
        return root_directories(&PathBuf::from(root));
    }
    let mut paths = Vec::new();
    #[cfg(windows)]
    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        let base = match sdk {
            Sdk::Cuda => PathBuf::from(program_files).join("NVIDIA GPU Computing Toolkit/CUDA"),
            Sdk::TensorRt => PathBuf::from(program_files),
        };
        if let Ok(entries) = std::fs::read_dir(base) {
            for entry in entries.flatten() {
                if matches!(sdk, Sdk::TensorRt)
                    && !entry.file_name().to_string_lossy().starts_with("TensorRT")
                {
                    continue;
                }
                paths.extend(existing_root_directories(&entry.path()));
            }
        }
    }
    #[cfg(unix)]
    paths.extend(
        [
            "/usr/local/cuda/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/local/lib",
        ]
        .into_iter()
        .map(PathBuf::from)
        .filter(|p| p.is_dir()),
    );
    let path_variable = if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    if let Some(value) = std::env::var_os(path_variable) {
        paths.extend(std::env::split_paths(&value).filter(|p| p.is_absolute() && p.is_dir()));
    }
    normalize(paths)
}
fn existing_root_directories(root: &Path) -> Vec<PathBuf> {
    ["bin", "lib", "lib64", "targets/x86_64-linux/lib"]
        .into_iter()
        .map(|suffix| root.join(suffix))
        .filter(|p| p.is_dir())
        .collect()
}
fn root_directories(root: &Path) -> Result<Vec<PathBuf>, NativeRuntimeError> {
    if !root.is_absolute() || !root.is_dir() {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::LibraryNotFound,
            "SDK root is not an existing absolute directory",
        )
        .with_path(root));
    }
    normalize(existing_root_directories(root))
}
fn normalize(paths: impl IntoIterator<Item = PathBuf>) -> Result<Vec<PathBuf>, NativeRuntimeError> {
    let mut output = Vec::new();
    for path in paths {
        if !path.is_absolute() || !path.is_dir() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::LibraryNotFound,
                "library directory does not exist or is relative",
            )
            .with_path(path));
        }
        let path = path.canonicalize().map_err(|e| {
            NativeRuntimeError::new(NativeRuntimeErrorKind::LibraryNotFound, e.to_string())
                .with_path(&path)
        })?;
        if !output.contains(&path) {
            output.push(path);
        }
    }
    Ok(output)
}
/// Resolve one component without choosing a version when multiple binaries exist.
pub(crate) fn library(
    directories: &[PathBuf],
    prefix: &str,
    suffix: &str,
) -> Result<LibraryFile, NativeRuntimeError> {
    let mut candidates: Vec<LibraryFile> = Vec::new();
    for directory in directories {
        let entries = std::fs::read_dir(directory).map_err(|e| {
            NativeRuntimeError::new(NativeRuntimeErrorKind::LibraryNotFound, e.to_string())
                .with_path(directory)
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| {
                NativeRuntimeError::new(NativeRuntimeErrorKind::LibraryNotFound, e.to_string())
                    .with_path(directory)
            })?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with(prefix) || !name.ends_with(suffix) {
                continue;
            }
            let middle = &name[prefix.len()..name.len() - suffix.len()];
            if !middle.chars().all(|c| c.is_ascii_digit() || c == '.') {
                continue;
            }
            let candidate = LibraryFile::resolve(&entry.path())?;
            if !candidates
                .iter()
                .any(|existing| existing.identity == candidate.identity)
            {
                candidates.push(candidate);
            }
        }
    }
    if candidates.len() == 1 {
        return Ok(candidates.remove(0));
    }
    let mut error = NativeRuntimeError::new(
        if candidates.is_empty() {
            NativeRuntimeErrorKind::LibraryNotFound
        } else {
            NativeRuntimeErrorKind::RuntimeConflict
        },
        format!(
            "expected one {prefix}*{suffix} library; found {}",
            candidates.len()
        ),
    );
    for directory in directories {
        error = error.with_path(directory);
    }
    Err(error)
}
pub(crate) fn driver() -> Result<LibraryFile, NativeRuntimeError> {
    #[cfg(windows)]
    {
        let mut buffer = vec![0u16; 32768];
        // SAFETY: buffer is valid and writable for the specified capacity.
        let count = unsafe {
            windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW(
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            )
        };
        if count == 0 || count as usize >= buffer.len() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::DriverUnavailable,
                "cannot resolve Windows system directory",
            ));
        }
        use std::os::windows::ffi::OsStringExt;
        LibraryFile::resolve(
            &PathBuf::from(std::ffi::OsString::from_wide(&buffer[..count as usize]))
                .join("nvcuda.dll"),
        )
    }
    #[cfg(unix)]
    {
        let dirs = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
            "/usr/lib",
            "/lib/x86_64-linux-gnu",
        ]
        .into_iter()
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .collect::<Vec<_>>();
        library(&dirs, "libcuda.so.", "")
    }
}
