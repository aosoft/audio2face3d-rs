use std::env;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDownloadRequest {
    pub repository: String,
    pub revision: String,
    pub output: PathBuf,
    pub token_environment: String,
}

impl ModelDownloadRequest {
    pub fn command(&self) -> Command {
        let mut command = Command::new("hf");
        command.args([
            "download",
            &self.repository,
            "--revision",
            &self.revision,
            "--local-dir",
        ]);
        command.arg(&self.output);
        command
    }

    /// Explicitly invokes the Hugging Face CLI. Model acquisition is never
    /// performed as a side effect of model loading or Cargo builds.
    pub fn execute(&self) -> Result<(), DownloadFailure> {
        if self.repository.trim().is_empty()
            || self.revision.trim().is_empty()
            || self.output == Path::new("")
        {
            return Err(DownloadFailure::InvalidRequest);
        }
        if env::var_os(&self.token_environment).is_none() {
            return Err(DownloadFailure::MissingToken {
                environment: self.token_environment.clone(),
            });
        }
        let output = self
            .command()
            .output()
            .map_err(|error| DownloadFailure::CliUnavailable(error.to_string()))?;
        if output.status.success() {
            return Ok(());
        }
        Err(classify_download_failure(
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ))
    }
}

fn classify_download_failure(stderr: String) -> DownloadFailure {
    let lower = stderr.to_ascii_lowercase();
    if ["401", "403", "gated", "unauthorized", "forbidden", "token"]
        .iter()
        .any(|needle| lower.contains(needle))
    {
        DownloadFailure::Authentication(stderr)
    } else {
        DownloadFailure::Command(stderr)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DownloadFailure {
    InvalidRequest,
    MissingToken { environment: String },
    CliUnavailable(String),
    Authentication(String),
    Command(String),
}

impl fmt::Display for DownloadFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => write!(formatter, "invalid model download request"),
            Self::MissingToken { environment } => write!(
                formatter,
                "model access token is missing; set {environment} after accepting the model license"
            ),
            Self::CliUnavailable(error) => {
                write!(formatter, "Hugging Face CLI `hf` is unavailable: {error}")
            }
            Self::Authentication(error) => write!(
                formatter,
                "gated model authentication failed; accept the license and verify the token: {error}"
            ),
            Self::Command(error) => write!(formatter, "model download failed: {error}"),
        }
    }
}

impl std::error::Error for DownloadFailure {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_is_explicit_and_missing_token_is_actionable() {
        let request = ModelDownloadRequest {
            repository: "nvidia/model".into(),
            revision: "revision".into(),
            output: "model".into(),
            token_environment: "AUDIO2X_TEST_TOKEN_THAT_IS_NOT_SET".into(),
        };
        let command = request.command();
        assert_eq!(command.get_program(), "hf");
        assert!(matches!(
            request.execute(),
            Err(DownloadFailure::MissingToken { .. })
        ));
    }

    #[test]
    fn discovery_always_produces_a_diagnostic() {
        assert!(!RuntimeDiscovery::discover().diagnostic().is_empty());
    }

    #[test]
    fn gated_download_failures_are_classified_as_authentication_errors() {
        for message in [
            "HTTP 401 Unauthorized",
            "HTTP 403 Forbidden",
            "Access to this gated repository is restricted",
        ] {
            assert!(matches!(
                classify_download_failure(message.into()),
                DownloadFailure::Authentication(_)
            ));
        }
        assert!(matches!(
            classify_download_failure("connection timed out".into()),
            DownloadFailure::Command(_)
        ));
    }
}
