use std::{
    fmt,
    path::{Path, PathBuf},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeRuntimeErrorKind {
    InvalidConfig,
    LibraryNotFound,
    DependencyLoadFailed,
    SymbolMissing,
    VersionMismatch,
    AbiMismatch,
    RuntimeConflict,
    InitializationReentered,
    InitializationFailed,
    DriverUnavailable,
    DeviceUnavailable,
}

/// A native initialization failure, retaining diagnostics across worker boundaries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeRuntimeError {
    kind: NativeRuntimeErrorKind,
    message: String,
    paths: Vec<PathBuf>,
    restart_required: bool,
}
impl NativeRuntimeError {
    pub(crate) fn new(kind: NativeRuntimeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            paths: Vec::new(),
            restart_required: false,
        }
    }
    pub(crate) fn with_path(mut self, path: impl AsRef<Path>) -> Self {
        self.paths.push(path.as_ref().to_owned());
        self
    }
    #[cfg(any(feature = "cuda", feature = "runtime-cli", test))]
    pub(crate) fn after_load(mut self) -> Self {
        self.restart_required = true;
        self
    }
    pub fn kind(&self) -> NativeRuntimeErrorKind {
        self.kind
    }
    pub fn message(&self) -> &str {
        &self.message
    }
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
    pub fn restart_required(&self) -> bool {
        self.restart_required
    }
}
impl fmt::Display for NativeRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)?;
        for path in &self.paths {
            write!(f, " [{}]", path.display())?;
        }
        if self.restart_required {
            write!(f, "; restart the process before retrying")?;
        }
        Ok(())
    }
}
impl std::error::Error for NativeRuntimeError {}
