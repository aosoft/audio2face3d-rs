use super::NativeVersion;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NativeRuntimeState {
    Discovered,
    Loaded,
    DeviceReady,
}

/// Observed component identity; filesystem discovery alone is not readiness.
#[derive(Clone, Debug)]
pub struct NativeLibraryInfo {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) build_version: Option<NativeVersion>,
    pub(crate) runtime_version: Option<NativeVersion>,
}
impl NativeLibraryInfo {
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn path(&self) -> &Path {
        &self.path
    }
    pub fn build_version(&self) -> Option<NativeVersion> {
        self.build_version
    }
    pub fn runtime_version(&self) -> Option<NativeVersion> {
        self.runtime_version
    }
}
#[derive(Clone, Debug)]
pub struct NativeRuntimeInfo {
    pub(crate) state: NativeRuntimeState,
    pub(crate) libraries: Vec<NativeLibraryInfo>,
}
impl NativeRuntimeInfo {
    pub fn state(&self) -> NativeRuntimeState {
        self.state
    }
    pub fn libraries(&self) -> &[NativeLibraryInfo] {
        &self.libraries
    }
}
