//! OS handles are retained for process lifetime, independently of GPU resource lifetime.
use super::{NativeRuntimeError, NativeRuntimeErrorKind, registry::LoadAttempt};
use std::path::{Path, PathBuf};
#[cfg(unix)]
mod linux;
#[cfg(windows)]
mod windows;
#[cfg(unix)]
use linux as platform;
pub(crate) use platform::FileIdentity;
#[cfg(windows)]
use windows as platform;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LibraryFile {
    pub(crate) path: PathBuf,
    pub(crate) identity: FileIdentity,
}
impl LibraryFile {
    pub(crate) fn resolve(path: &Path) -> Result<Self, NativeRuntimeError> {
        if !path.is_absolute() {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InvalidConfig,
                "library path must be absolute",
            )
            .with_path(path));
        }
        let path = path.canonicalize().map_err(|error| {
            NativeRuntimeError::new(NativeRuntimeErrorKind::LibraryNotFound, error.to_string())
                .with_path(path)
        })?;
        let identity = platform::identity(&path)?;
        Ok(Self { path, identity })
    }
}
#[derive(Debug)]
pub(crate) struct LoadedLibrary {
    pub(crate) file: LibraryFile,
    library: &'static libloading::Library,
}
impl LoadedLibrary {
    /// The selected file must be trusted executable code.
    pub(crate) unsafe fn open(
        file: LibraryFile,
        directories: &[PathBuf],
        attempt: &mut LoadAttempt,
    ) -> Result<Self, NativeRuntimeError> {
        // SAFETY: the caller authorizes loading the selected native code.
        let library = unsafe { platform::open(&file, directories, attempt) }?;
        Ok(Self { file, library })
    }
    /// T must match the symbol's ABI and type exactly.
    pub(crate) unsafe fn symbol<T: Copy>(&self, name: &[u8]) -> Result<T, NativeRuntimeError> {
        // SAFETY: the caller supplies the exact ABI type; the module remains loaded forever.
        unsafe { self.library.get::<T>(name) }
            .map(|symbol| *symbol)
            .map_err(|error| {
                NativeRuntimeError::new(NativeRuntimeErrorKind::SymbolMissing, error.to_string())
                    .with_path(&self.file.path)
            })
    }
}

#[cfg(test)]
mod tests;

/// Reject NVIDIA dependencies resolved from a different SDK, including prior host loads.
pub(crate) fn validate_loaded(
    directories: &[PathBuf],
    include_tensorrt: bool,
) -> Result<(), NativeRuntimeError> {
    for path in platform::loaded_paths()? {
        let Some(name) = path.file_name() else {
            continue;
        };
        let lower = name.to_string_lossy().to_ascii_lowercase();
        let nvidia = [
            "cudart",
            "cublas",
            "curand",
            "nvrtc",
            "nvjitlink",
            "libcudart",
            "libcublas",
            "libcurand",
            "libnvrtc",
            "libnvjitlink",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
            || (include_tensorrt
                && (lower.starts_with("nvinfer") || lower.starts_with("libnvinfer")));
        if !nvidia {
            continue;
        }
        let actual = LibraryFile::resolve(&path)?;
        let matches = directories
            .iter()
            .map(|directory| directory.join(name))
            .filter(|candidate| candidate.is_file())
            .any(|candidate| {
                LibraryFile::resolve(&candidate)
                    .is_ok_and(|expected| expected.identity == actual.identity)
            });
        if !matches {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::RuntimeConflict,
                "NVIDIA dependency was loaded outside the selected SDK directories",
            )
            .with_path(path)
            .after_load());
        }
    }
    Ok(())
}
