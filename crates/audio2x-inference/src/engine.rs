//! Safe orchestration of ONNX to TensorRT engine generation.
//!
//! TensorRT itself remains behind the C ABI boundary.  This module only owns
//! the `trtexec` process and the filesystem transaction around its output.

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// One TensorRT optimization profile for a named dynamic input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeProfile {
    pub name: String,
    pub min: Vec<u64>,
    pub opt: Vec<u64>,
    pub max: Vec<u64>,
}

impl ShapeProfile {
    fn validate(&self) -> Result<(), EngineError> {
        if self.name.is_empty() || self.min.is_empty() {
            return Err(EngineError::InvalidProfile(self.name.clone()));
        }
        if self.min.len() != self.opt.len()
            || self.min.len() != self.max.len()
            || self.min.iter().zip(&self.opt).any(|(a, b)| a > b)
            || self.opt.iter().zip(&self.max).any(|(a, b)| a > b)
            || self.min.contains(&0)
            || self.opt.contains(&0)
            || self.max.contains(&0)
        {
            return Err(EngineError::InvalidProfile(self.name.clone()));
        }
        Ok(())
    }

    fn shape(&self, values: &[u64]) -> String {
        values
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join("x")
    }
}

/// Inputs needed to invoke the original SDK-compatible `trtexec` command.
#[derive(Clone, Debug)]
pub struct EngineBuildRequest {
    pub onnx: PathBuf,
    pub engine: PathBuf,
    pub device_id: u32,
    pub profiles: Vec<ShapeProfile>,
    pub extra_args: Vec<String>,
}

impl EngineBuildRequest {
    pub fn command_args(&self, output: &Path) -> Result<Vec<String>, EngineError> {
        if self
            .profiles
            .iter()
            .try_for_each(ShapeProfile::validate)
            .is_err()
        {
            return Err(EngineError::InvalidProfile("<profile>".into()));
        }
        let mut args = vec![
            format!("--onnx={}", self.onnx.display()),
            format!("--saveEngine={}", output.display()),
            format!("--device={}", self.device_id),
        ];
        if !self.profiles.is_empty() {
            for kind in ["minShapes", "optShapes", "maxShapes"] {
                let values = self
                    .profiles
                    .iter()
                    .map(|p| {
                        let dims = match kind {
                            "minShapes" => p.shape(&p.min),
                            "optShapes" => p.shape(&p.opt),
                            _ => p.shape(&p.max),
                        };
                        format!("{}:{}", p.name, dims)
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                args.push(format!("--{kind}={values}"));
            }
        }
        args.extend(self.extra_args.iter().cloned());
        Ok(args)
    }
}

#[derive(Debug)]
pub enum EngineError {
    InvalidProfile(String),
    MissingOnnx(PathBuf),
    ExistingTarget(PathBuf),
    EmptyOutput(PathBuf),
    TrtFailed(i32),
    Io(io::Error),
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidProfile(n) => write!(f, "invalid optimization profile: {n}"),
            Self::MissingOnnx(p) => write!(f, "ONNX file does not exist: {}", p.display()),
            Self::ExistingTarget(p) => write!(f, "engine target already exists: {}", p.display()),
            Self::EmptyOutput(p) => write!(f, "trtexec produced an empty engine: {}", p.display()),
            Self::TrtFailed(c) => write!(f, "trtexec failed with exit code {c}"),
            Self::Io(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for EngineError {}
impl From<io::Error> for EngineError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Runs `trtexec` and installs a newly-created engine atomically.
#[derive(Clone, Debug)]
pub struct EngineBuilder {
    pub executable: PathBuf,
}

impl Default for EngineBuilder {
    fn default() -> Self {
        Self {
            executable: PathBuf::from("trtexec"),
        }
    }
}

impl EngineBuilder {
    pub fn build(&self, request: &EngineBuildRequest) -> Result<(), EngineError> {
        if !request.onnx.is_file() {
            return Err(EngineError::MissingOnnx(request.onnx.clone()));
        }
        if request.engine.exists() {
            return Err(EngineError::ExistingTarget(request.engine.clone()));
        }
        let parent = request.engine.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let temp = temporary_path(&request.engine);
        let args = request.command_args(&temp)?;
        let status = Command::new(&self.executable).args(&args).status()?;
        if !status.success() {
            let _ = fs::remove_file(&temp);
            return Err(EngineError::TrtFailed(status.code().unwrap_or(-1)));
        }
        install_new_engine(&temp, &request.engine)
    }
}

fn temporary_path(target: &Path) -> PathBuf {
    let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let suffix = format!(".partial-{}-{}", std::process::id(), n);
    PathBuf::from(format!("{}{}", target.display(), suffix))
}

/// Verify and atomically install a freshly generated, non-empty file.
fn install_new_engine(temp: &Path, target: &Path) -> Result<(), EngineError> {
    let metadata = fs::metadata(temp)?;
    if metadata.len() == 0 {
        let _ = fs::remove_file(temp);
        return Err(EngineError::EmptyOutput(temp.to_path_buf()));
    }
    if target.exists() {
        return Err(EngineError::ExistingTarget(target.to_path_buf()));
    }
    let file = OpenOptions::new().read(true).write(true).open(temp)?;
    file.sync_all()?;
    drop(file);
    fs::rename(temp, target)?;
    if let Some(parent) = target.parent() {
        // Best effort: directory fsync is unsupported on some Windows filesystems.
        if let Ok(dir) = File::open(parent) {
            let _ = dir.sync_all();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_contains_profiles_in_trtexec_order() {
        let r = EngineBuildRequest {
            onnx: PathBuf::from("model.onnx"),
            engine: PathBuf::from("model.trt"),
            device_id: 2,
            profiles: vec![ShapeProfile {
                name: "input_values".into(),
                min: vec![1, 5000],
                opt: vec![1, 30000],
                max: vec![1, 60000],
            }],
            extra_args: vec!["--fp16".into()],
        };
        assert_eq!(
            r.command_args(Path::new("model.trt.partial")).unwrap(),
            vec![
                "--onnx=model.onnx",
                "--saveEngine=model.trt.partial",
                "--device=2",
                "--minShapes=input_values:1x5000",
                "--optShapes=input_values:1x30000",
                "--maxShapes=input_values:1x60000",
                "--fp16"
            ]
        );
    }

    #[test]
    fn install_rejects_empty_and_preserves_target() {
        let root = std::env::temp_dir().join(format!("audio2x-engine-test-{}", std::process::id()));
        let _ = fs::create_dir_all(&root);
        let temp = root.join("new.partial");
        let target = root.join("new.trt");
        fs::write(&temp, []).unwrap();
        assert!(matches!(
            install_new_engine(&temp, &target),
            Err(EngineError::EmptyOutput(_))
        ));
        assert!(!target.exists());
        let _ = fs::remove_dir_all(root);
    }
}
