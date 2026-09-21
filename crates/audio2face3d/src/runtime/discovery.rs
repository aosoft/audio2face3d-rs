#[cfg(any(feature = "cuda", feature = "runtime-cli", test))]
use super::loader::LibraryFile;
use super::{NativeRuntimeConfig, NativeRuntimeError, NativeRuntimeErrorKind, NativeSearchPolicy};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
pub(crate) enum Sdk {
    Cuda,
    TensorRt,
}
impl Sdk {
    pub(crate) fn environment_variable(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA_PATH",
            Self::TensorRt => "TENSORRT_ROOT_DIR",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CandidateSelection {
    Unique,
    First,
}
pub(crate) fn selection(config: &NativeRuntimeConfig, sdk: Sdk) -> CandidateSelection {
    let explicit = match sdk {
        Sdk::Cuda => config.cuda_root().is_some() || !config.cuda_library_dirs().is_empty(),
        Sdk::TensorRt => {
            config.tensorrt_root().is_some() || !config.tensorrt_library_dirs().is_empty()
        }
    };
    if explicit || config.search_policy() == NativeSearchPolicy::ExplicitOnly {
        CandidateSelection::Unique
    } else {
        CandidateSelection::First
    }
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
    let path_variable = if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    if let Some(value) = std::env::var_os(path_variable) {
        paths.extend(std::env::split_paths(&value).filter(|p| p.is_absolute() && p.is_dir()));
    }
    #[cfg(windows)]
    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        let base = match sdk {
            Sdk::Cuda => PathBuf::from(program_files).join("NVIDIA GPU Computing Toolkit/CUDA"),
            Sdk::TensorRt => PathBuf::from(program_files),
        };
        if let Ok(entries) = std::fs::read_dir(base) {
            let mut entries = entries.flatten().collect::<Vec<_>>();
            entries.sort_by_key(|entry| entry.path());
            for entry in entries {
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
/// Explicit locations require a unique file; discovery follows directory order.
#[cfg(any(feature = "cuda", feature = "runtime-cli", test))]
pub(crate) fn library(
    directories: &[PathBuf],
    prefix: &str,
    suffix: &str,
    selection: CandidateSelection,
) -> Result<LibraryFile, NativeRuntimeError> {
    let mut candidates: Vec<LibraryFile> = Vec::new();
    for directory in directories {
        let entries = std::fs::read_dir(directory).map_err(|e| {
            NativeRuntimeError::new(NativeRuntimeErrorKind::LibraryNotFound, e.to_string())
                .with_path(directory)
        })?;
        let mut entries = entries.collect::<Result<Vec<_>, _>>().map_err(|e| {
            NativeRuntimeError::new(NativeRuntimeErrorKind::LibraryNotFound, e.to_string())
                .with_path(directory)
        })?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
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
            if selection == CandidateSelection::First {
                return Ok(candidate);
            }
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
#[cfg(any(feature = "cuda", feature = "runtime-cli", test))]
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
        library(&dirs, "libcuda.so.", "", CandidateSelection::First)
    }
}

#[cfg(any(feature = "cuda", feature = "runtime-cli", test))]
impl NativeRuntimeConfig {
    /// Inspects candidate files without loading any DLL or calling the GPU driver.
    pub fn discover(&self) -> Result<super::NativeRuntimeInfo, NativeRuntimeError> {
        let cuda = directories(self, Sdk::Cuda)?;
        let trt = directories(self, Sdk::TensorRt)?;
        let cuda_selection = selection(self, Sdk::Cuda);
        let trt_selection = selection(self, Sdk::TensorRt);
        let mut files = vec![driver()?];
        #[cfg(windows)]
        let names = [
            (&cuda, "cublasLt64_", ".dll", cuda_selection),
            (&cuda, "cublas64_", ".dll", cuda_selection),
            (&cuda, "curand64_", ".dll", cuda_selection),
            (&cuda, "cudart64_", ".dll", cuda_selection),
            (&trt, "nvinfer_", ".dll", trt_selection),
        ];
        #[cfg(unix)]
        let names = [
            (&cuda, "libcublasLt.so", "", cuda_selection),
            (&cuda, "libcublas.so", "", cuda_selection),
            (&cuda, "libcurand.so", "", cuda_selection),
            (&cuda, "libcudart.so", "", cuda_selection),
            (&trt, "libnvinfer.so", "", trt_selection),
        ];
        for (dirs, prefix, suffix, selection) in names {
            files.push(library(dirs, prefix, suffix, selection)?);
        }
        Ok(super::NativeRuntimeInfo {
            state: super::NativeRuntimeState::Discovered,
            libraries: files
                .into_iter()
                .map(|file| super::NativeLibraryInfo {
                    name: file
                        .path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    path: file.path,
                    build_version: None,
                    runtime_version: None,
                })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_discovery_accepts_duplicates_in_search_order_but_explicit_locations_do_not() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../temp/platform-config-work/discovery")
            .join(std::process::id().to_string());
        let first = root.join("z-first");
        let second = root.join("a-second");
        for dir in [&first, &second] {
            std::fs::create_dir_all(dir).unwrap();
            std::fs::write(dir.join("fixture_12.dll"), "fixture").unwrap();
        }
        let dirs = [first, second];
        let default = NativeRuntimeConfig::default();
        let mode = selection(&default, Sdk::Cuda);
        assert_eq!(mode, CandidateSelection::First);
        let file = library(&dirs, "fixture_", ".dll", mode).unwrap();
        assert_eq!(
            file.path,
            dirs[0].join("fixture_12.dll").canonicalize().unwrap()
        );
        let reversed = [dirs[1].clone(), dirs[0].clone()];
        let file = library(&reversed, "fixture_", ".dll", mode).unwrap();
        assert_eq!(
            file.path,
            dirs[1].join("fixture_12.dll").canonicalize().unwrap()
        );
        let explicit = NativeRuntimeConfig::builder()
            .cuda_library_dirs(dirs.clone())
            .build()
            .unwrap();
        assert_eq!(
            selection(&explicit, Sdk::TensorRt),
            CandidateSelection::First
        );
        assert_eq!(
            library(&dirs, "fixture_", ".dll", selection(&explicit, Sdk::Cuda))
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::RuntimeConflict
        );
        assert_eq!(
            library(&dirs, "missing_", ".dll", mode).unwrap_err().kind(),
            NativeRuntimeErrorKind::LibraryNotFound
        );
    }
}
