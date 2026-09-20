//! Absolute executable selection and child-only dependency search paths.
use super::{
    NativeRuntimeConfig, NativeRuntimeError, NativeRuntimeErrorKind, NativeSearchPolicy,
    discovery::{self, Sdk},
};
use std::{
    path::{Path, PathBuf},
    process::Command,
};
#[derive(Clone, Copy, Debug)]
pub enum NativeTool {
    Nvcc,
    Trtexec,
}
impl NativeTool {
    fn name(self) -> &'static str {
        match self {
            Self::Nvcc => "nvcc",
            Self::Trtexec => "trtexec",
        }
    }
}
fn error(message: impl Into<String>) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::LibraryNotFound, message)
}
fn unique(mut paths: Vec<PathBuf>, label: &str) -> Result<PathBuf, NativeRuntimeError> {
    paths = paths
        .into_iter()
        .filter(|path| path.is_file())
        .map(|path| {
            path.canonicalize()
                .map_err(|e| error(e.to_string()).with_path(&path))
        })
        .collect::<Result<Vec<_>, _>>()?;
    paths.sort();
    paths.dedup();
    if paths.len() != 1 {
        return Err(error(format!(
            "expected one {label}; found {} candidates",
            paths.len()
        )));
    }
    Ok(paths.remove(0))
}
fn executable_in(directory: &Path, name: &Path) -> Vec<PathBuf> {
    let path = directory.join(name);
    #[cfg(windows)]
    if path.extension().is_none() {
        return vec![
            path.with_extension("exe"),
            path.with_extension("cmd"),
            path.with_extension("bat"),
        ];
    }
    vec![path]
}
impl NativeRuntimeConfig {
    /// Creates a command without starting a process or changing the parent environment.
    /// `executable` may override the selected tool; SDK roots still govern its child dependencies.
    pub fn tool_command(
        &self,
        tool: NativeTool,
        executable: Option<&Path>,
    ) -> Result<Command, NativeRuntimeError> {
        let sdk = match tool {
            NativeTool::Nvcc => Sdk::Cuda,
            NativeTool::Trtexec => Sdk::TensorRt,
        };
        let dirs = discovery::directories(self, sdk)?;
        let explicit_sdk = match tool {
            NativeTool::Nvcc => self.cuda_root().is_some() || !self.cuda_library_dirs().is_empty(),
            NativeTool::Trtexec => {
                self.tensorrt_root().is_some() || !self.tensorrt_library_dirs().is_empty()
            }
        };
        let requested = executable.unwrap_or_else(|| Path::new(tool.name()));
        let selected = if requested.is_absolute() || requested.components().count() > 1 {
            unique(
                vec![
                    std::env::current_dir()
                        .map_err(|e| error(e.to_string()))?
                        .join(requested),
                ],
                tool.name(),
            )?
        } else {
            let mut candidates = Vec::new();
            for dir in &dirs {
                candidates.extend(executable_in(dir, requested));
                if let Some(parent) = dir.parent() {
                    candidates.extend(executable_in(&parent.join("bin"), requested));
                }
            }
            if !candidates.iter().any(|p| p.is_file())
                && !explicit_sdk
                && self.search_policy() == NativeSearchPolicy::Discover
                && let Some(path) = std::env::var_os("PATH")
            {
                for dir in std::env::split_paths(&path).filter(|p| p.is_absolute()) {
                    candidates.extend(executable_in(&dir, requested));
                }
            }
            unique(candidates, tool.name())?
        };
        let mut child_dirs = dirs;
        if matches!(tool, NativeTool::Trtexec) {
            child_dirs.extend(discovery::directories(self, Sdk::Cuda)?);
        }
        // Reject ambiguous CUDA Runtime installations before relying on child PATH ordering.
        let mut runtimes = Vec::new();
        for dir in &child_dirs {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if (name.starts_with("cudart64_") && name.ends_with(".dll"))
                        || name.starts_with("libcudart.so")
                    {
                        runtimes.push(entry.path());
                    }
                }
            }
        }
        if !runtimes.is_empty() {
            unique(runtimes, "CUDA Runtime for child process")?;
        }
        if let Some(parent) = selected.parent() {
            child_dirs.insert(0, parent.to_owned());
        }
        let variable = if cfg!(windows) {
            "PATH"
        } else {
            "LD_LIBRARY_PATH"
        };
        if let Some(existing) = std::env::var_os(variable) {
            child_dirs.extend(std::env::split_paths(&existing));
        }
        let child_path = std::env::join_paths(child_dirs).map_err(|e| error(e.to_string()))?;
        let mut command = Command::new(selected);
        command.env(variable, child_path);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        Ok(command)
    }
}
