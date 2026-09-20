//! Native configuration and diagnostics, usable without loading a GPU SDK.
mod config;
#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
pub(crate) mod discovery;
mod error;
mod info;
#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)]
pub(crate) mod loader;
#[cfg(any(feature = "cuda", test))]
#[allow(dead_code)] // Connected to native initialization in the following migration phase.
pub(crate) mod registry;
mod version;
pub use config::{NativeRuntimeConfig, NativeRuntimeConfigBuilder, NativeSearchPolicy};
pub use error::{NativeRuntimeError, NativeRuntimeErrorKind};
pub use info::{NativeLibraryInfo, NativeRuntimeInfo, NativeRuntimeState};
pub use version::{NativeVersion, VersionCompatibility};

use std::env;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeDiscovery {
    pub cuda_root: Option<PathBuf>,
    pub tensorrt_root: Option<PathBuf>,
    pub searched_library_directories: Vec<PathBuf>,
    pub missing_libraries: Vec<&'static str>,
}

impl RuntimeDiscovery {
    pub fn discover() -> Self {
        let cuda_root = env::var_os("CUDA_PATH").map(PathBuf::from);
        let tensorrt_root = env::var_os("TENSORRT_ROOT_DIR").map(PathBuf::from);
        let mut directories = env::var_os(path_variable())
            .map(|value| env::split_paths(&value).collect::<Vec<_>>())
            .unwrap_or_default();
        if !cfg!(windows) {
            for candidate in [
                "/usr/lib",
                "/usr/lib64",
                "/usr/local/lib",
                "/usr/local/cuda/lib64",
                "/usr/lib/x86_64-linux-gnu",
            ] {
                let candidate = PathBuf::from(candidate);
                if candidate.is_dir() && !directories.contains(&candidate) {
                    directories.push(candidate);
                }
            }
        }
        for root in [&cuda_root, &tensorrt_root].into_iter().flatten() {
            for candidate in [root.join("bin"), root.join("lib"), root.join("lib64")] {
                if candidate.is_dir() && !directories.contains(&candidate) {
                    directories.push(candidate);
                }
            }
        }
        let expected = if cfg!(windows) {
            ["nvcuda.dll", "cudart64_12.dll", "nvinfer_10.dll"]
        } else {
            ["libcuda.so", "libcudart.so.12", "libnvinfer.so.10"]
        };
        let missing_libraries = expected
            .into_iter()
            .filter(|name| !library_available(name, &directories))
            .collect();
        Self {
            cuda_root,
            tensorrt_root,
            searched_library_directories: directories,
            missing_libraries,
        }
    }

    pub fn is_ready(&self) -> bool {
        self.missing_libraries.is_empty()
    }

    pub fn diagnostic(&self) -> String {
        if self.is_ready() {
            "CUDA and TensorRT runtime libraries were found".into()
        } else {
            format!(
                "missing runtime libraries: {}; configure {} and the platform loader path",
                self.missing_libraries.join(", "),
                if self.tensorrt_root.is_none() {
                    "TENSORRT_ROOT_DIR"
                } else {
                    path_variable()
                }
            )
        }
    }
}

fn library_available(name: &str, directories: &[PathBuf]) -> bool {
    if directories
        .iter()
        .any(|directory| directory.join(name).is_file())
    {
        return true;
    }
    if cfg!(windows) && name == "nvcuda.dll" {
        return env::var_os("SystemRoot")
            .map(PathBuf::from)
            .is_some_and(|root| root.join("System32").join(name).is_file());
    }
    false
}

const fn path_variable() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_always_produces_a_diagnostic() {
        assert!(!RuntimeDiscovery::discover().diagnostic().is_empty());
    }
}
