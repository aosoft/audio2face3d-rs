use std::fmt;

/// A component version. Unknown patch/build fields remain absent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeVersion {
    major: u32,
    minor: u32,
    patch: Option<u32>,
    build: Option<u32>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VersionCompatibility {
    Compatible,
    MinorMismatch,
    MajorMismatch,
}
impl NativeVersion {
    pub const fn new(major: u32, minor: u32, patch: Option<u32>, build: Option<u32>) -> Self {
        Self {
            major,
            minor,
            patch,
            build,
        }
    }
    pub const fn major(self) -> u32 {
        self.major
    }
    pub const fn minor(self) -> u32 {
        self.minor
    }
    pub const fn patch(self) -> Option<u32> {
        self.patch
    }
    pub const fn build(self) -> Option<u32> {
        self.build
    }
    /// Compare the same component at build time and runtime (not driver vs toolkit).
    pub const fn compatibility(self, runtime: Self) -> VersionCompatibility {
        if self.major != runtime.major {
            VersionCompatibility::MajorMismatch
        } else if self.minor != runtime.minor {
            VersionCompatibility::MinorMismatch
        } else {
            VersionCompatibility::Compatible
        }
    }
}
impl fmt::Display for NativeVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)?;
        if let Some(patch) = self.patch {
            write!(f, ".{patch}")?;
        }
        if let Some(build) = self.build {
            write!(f, "+{build}")?;
        }
        Ok(())
    }
}

#[cfg(any(feature = "tensorrt", test))]
use crate::{
    Audio2Face3DContext,
    logging::{LogLevel, Logger},
    runtime::{NativeRuntimeError, NativeRuntimeErrorKind},
};
#[cfg(any(feature = "tensorrt", test))]
pub(crate) fn verify(
    context: &Audio2Face3DContext,
    name: &str,
    build: NativeVersion,
    runtime: NativeVersion,
) -> Result<(), NativeRuntimeError> {
    match build.compatibility(runtime) {
        VersionCompatibility::MajorMismatch => Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::VersionMismatch,
            format!("{name} major mismatch: built with {build}, loaded {runtime}"),
        )),
        VersionCompatibility::MinorMismatch => {
            context.logger().log(LogLevel::Warn, || {
                crate::logging::LogRecord::new(format!(
                    "{name} minor mismatch: built with {build}, loaded {runtime}; continuing"
                ))
                .field("source", module_path!())
            });
            Ok(())
        }
        VersionCompatibility::Compatible => Ok(()),
    }
}
