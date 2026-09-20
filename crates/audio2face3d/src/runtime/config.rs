use super::{NativeRuntimeError, NativeRuntimeErrorKind};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeSearchPolicy {
    ExplicitOnly,
    #[default]
    Discover,
}

/// Runtime paths, independent of the SDK used to compile this crate.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeRuntimeConfig {
    cuda_root: Option<PathBuf>,
    tensorrt_root: Option<PathBuf>,
    cuda_library_dirs: Vec<PathBuf>,
    tensorrt_library_dirs: Vec<PathBuf>,
    search_policy: NativeSearchPolicy,
}

#[derive(Clone, Debug, Default)]
pub struct NativeRuntimeConfigBuilder {
    config: NativeRuntimeConfig,
    cuda_dirs_set: bool,
    tensorrt_dirs_set: bool,
}

impl NativeRuntimeConfig {
    pub fn builder() -> NativeRuntimeConfigBuilder {
        NativeRuntimeConfigBuilder::default()
    }
    pub fn cuda_root(&self) -> Option<&Path> {
        self.cuda_root.as_deref()
    }
    pub fn tensorrt_root(&self) -> Option<&Path> {
        self.tensorrt_root.as_deref()
    }
    pub fn cuda_library_dirs(&self) -> &[PathBuf] {
        &self.cuda_library_dirs
    }
    pub fn tensorrt_library_dirs(&self) -> &[PathBuf] {
        &self.tensorrt_library_dirs
    }
    pub fn search_policy(&self) -> NativeSearchPolicy {
        self.search_policy
    }
}

impl NativeRuntimeConfigBuilder {
    pub fn cuda_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.cuda_root = Some(path.into());
        self
    }
    pub fn tensorrt_root(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.tensorrt_root = Some(path.into());
        self
    }
    pub fn cuda_library_dirs(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        self.config.cuda_library_dirs = paths.into_iter().collect();
        self.cuda_dirs_set = true;
        self
    }
    pub fn tensorrt_library_dirs(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        self.config.tensorrt_library_dirs = paths.into_iter().collect();
        self.tensorrt_dirs_set = true;
        self
    }
    pub fn search_policy(mut self, policy: NativeSearchPolicy) -> Self {
        self.config.search_policy = policy;
        self
    }
    /// Validates syntax only; does not probe the filesystem or load native code.
    pub fn build(self) -> Result<NativeRuntimeConfig, NativeRuntimeError> {
        for (name, root, dirs, dirs_set) in [
            (
                "CUDA",
                &self.config.cuda_root,
                &self.config.cuda_library_dirs,
                self.cuda_dirs_set,
            ),
            (
                "TensorRT",
                &self.config.tensorrt_root,
                &self.config.tensorrt_library_dirs,
                self.tensorrt_dirs_set,
            ),
        ] {
            if dirs_set && dirs.is_empty() {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InvalidConfig,
                    format!("{name}: library directory list must not be empty"),
                ));
            }
            if root.is_some() && dirs_set {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InvalidConfig,
                    format!("{name}: root and library directories are mutually exclusive"),
                ));
            }
            for path in root.iter().chain(dirs.iter()) {
                if !path.is_absolute() || path.as_os_str().as_encoded_bytes().contains(&0) {
                    return Err(NativeRuntimeError::new(
                        NativeRuntimeErrorKind::InvalidConfig,
                        format!("{name}: expected a nonempty absolute path without NUL"),
                    )
                    .with_path(path));
                }
            }
        }
        Ok(self.config)
    }
}
