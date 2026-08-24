//! Rust-native model acquisition for Audio2X.

use hf_hub::{HFClient, HFClientSync, HFError, split_id};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::env;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);
const REQUIRED_MODEL_FILES: &[&str] = &[
    "model.json",
    "network.onnx",
    "network_info.json",
    "trt_info.json",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelPreset {
    pub name: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub output_directory: &'static str,
}

pub const MODEL_PRESETS: &[ModelPreset] = &[
    ModelPreset {
        name: "diffusion",
        repository: "nvidia/Audio2Face-3D-v3.0",
        revision: "b74132732fd9a9d29b237bec193ded64c9745e91",
        output_directory: "diffusion",
    },
    ModelPreset {
        name: "claire",
        repository: "nvidia/Audio2Face-3D-v2.3.1-Claire",
        revision: "a46eafb067adfdd7cc8f5c7941586b309004561c",
        output_directory: "claire",
    },
    ModelPreset {
        name: "james",
        repository: "nvidia/Audio2Face-3D-v2.3.1-James",
        revision: "327d000d9f76e370014a9b7467b23ea36846b680",
        output_directory: "james",
    },
    ModelPreset {
        name: "mark",
        repository: "nvidia/Audio2Face-3D-v2.3-Mark",
        revision: "5451728e07378df93b04523279e134a9993ae71b",
        output_directory: "mark",
    },
    ModelPreset {
        name: "emotion",
        repository: "nvidia/Audio2Emotion-v2.2",
        revision: "ce1358310179ed7f6b6ea63fe4fa9de5694c1b87",
        output_directory: "emotion",
    },
];

impl ModelPreset {
    pub fn request(
        self,
        output_root: impl AsRef<Path>,
        token_environment: impl Into<String>,
    ) -> ModelDownloadRequest {
        ModelDownloadRequest {
            repository: self.repository.into(),
            revision: self.revision.into(),
            output: output_root.as_ref().join(self.output_directory),
            token_environment: token_environment.into(),
        }
    }
}

pub fn model_preset(name: &str) -> Option<ModelPreset> {
    MODEL_PRESETS
        .iter()
        .copied()
        .find(|preset| preset.name == name)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelDownloadRequest {
    pub repository: String,
    pub revision: String,
    pub output: PathBuf,
    pub token_environment: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadReceipt {
    pub repository: String,
    pub revision: String,
    pub output: PathBuf,
    pub network_onnx_sha256: String,
}

impl ModelDownloadRequest {
    /// Downloads one immutable Hub revision without invoking Python or an
    /// external Hugging Face CLI. The completed snapshot becomes visible at
    /// `output` only after its required files and ONNX digest are validated.
    pub fn execute(&self) -> Result<DownloadReceipt, DownloadFailure> {
        self.validate()?;
        let token =
            env::var(&self.token_environment).map_err(|_| DownloadFailure::MissingToken {
                environment: self.token_environment.clone(),
            })?;
        if token.trim().is_empty() {
            return Err(DownloadFailure::MissingToken {
                environment: self.token_environment.clone(),
            });
        }

        let client = HFClientSync::from_inner(
            HFClient::builder()
                .token(token)
                .user_agent(format!("audio2x-model-tool/{}", env!("CARGO_PKG_VERSION")))
                .build()
                .map_err(map_hub_error)?,
        )
        .map_err(map_hub_error)?;
        let (owner, name) = split_id(&self.repository);
        let repository = client.model(owner, name);
        self.install_snapshot(|staging| {
            repository
                .snapshot_download()
                .revision(self.revision.clone())
                .local_dir(staging.to_owned())
                .max_workers(8)
                .send()
                .map_err(map_hub_error)?;
            Ok(())
        })
    }

    fn validate(&self) -> Result<(), DownloadFailure> {
        let valid_repository = self
            .repository
            .split_once('/')
            .is_some_and(|(owner, name)| {
                !owner.is_empty() && !name.is_empty() && !name.contains('/')
            });
        if !valid_repository {
            return Err(DownloadFailure::InvalidRepository(self.repository.clone()));
        }
        if self.revision.len() != 40
            || !self.revision.bytes().all(|value| value.is_ascii_hexdigit())
        {
            return Err(DownloadFailure::InvalidRevision(self.revision.clone()));
        }
        if self.output.as_os_str().is_empty() || self.output.file_name().is_none() {
            return Err(DownloadFailure::InvalidOutput(self.output.clone()));
        }
        if self.token_environment.trim().is_empty() {
            return Err(DownloadFailure::InvalidTokenEnvironment);
        }
        Ok(())
    }

    fn install_snapshot(
        &self,
        download: impl FnOnce(&Path) -> Result<(), DownloadFailure>,
    ) -> Result<DownloadReceipt, DownloadFailure> {
        if self.output.exists() {
            return Err(DownloadFailure::OutputExists(self.output.clone()));
        }
        let parent = self.output.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent)?;
        let mut staging = StagingDirectory::create(parent, self.output.file_name().unwrap())?;
        download(staging.path())?;
        for name in REQUIRED_MODEL_FILES {
            let path = staging.path().join(name);
            if !path.is_file() {
                return Err(DownloadFailure::MissingModelFile((*name).to_owned()));
            }
        }
        let network_onnx_sha256 = sha256(&staging.path().join("network.onnx"))?;
        let provenance = json!({
            "schema_version": 1,
            "repository": self.repository,
            "revision": self.revision,
            "network_onnx_sha256": network_onnx_sha256,
        });
        let mut payload = serde_json::to_vec_pretty(&provenance)
            .map_err(|error| DownloadFailure::Metadata(error.to_string()))?;
        payload.push(b'\n');
        let mut stream = File::create(staging.path().join(".audio2x-source.json"))?;
        stream.write_all(&payload)?;
        stream.sync_all()?;
        drop(stream);
        staging.install(&self.output)?;
        Ok(DownloadReceipt {
            repository: self.repository.clone(),
            revision: self.revision.clone(),
            output: self.output.clone(),
            network_onnx_sha256,
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DownloadFailure {
    #[error("model repository must use owner/name form: {0}")]
    InvalidRepository(String),
    #[error("model revision must be a full 40-character commit SHA: {0}")]
    InvalidRevision(String),
    #[error("model output must name a new directory: {}", .0.display())]
    InvalidOutput(PathBuf),
    #[error("token environment variable name must not be empty")]
    InvalidTokenEnvironment,
    #[error("model access token is missing; set {environment} after accepting the model license")]
    MissingToken { environment: String },
    #[error("model output already exists: {}", .0.display())]
    OutputExists(PathBuf),
    #[error("downloaded model is missing required file: {0}")]
    MissingModelFile(String),
    #[error("gated model authentication failed; accept the license and verify the token: {0}")]
    Authentication(String),
    #[error("model repository is unavailable: {0}")]
    Repository(String),
    #[error("model revision is unavailable: {0}")]
    Revision(String),
    #[error("Hugging Face rate limit reached: {0}")]
    RateLimited(String),
    #[error("Hugging Face download failed: {0}")]
    Hub(String),
    #[error("unable to write model provenance: {0}")]
    Metadata(String),
    #[error(transparent)]
    Io(#[from] io::Error),
}

fn map_hub_error(error: HFError) -> DownloadFailure {
    match error {
        value @ (HFError::AuthRequired { .. } | HFError::Forbidden { .. }) => {
            DownloadFailure::Authentication(value.to_string())
        }
        value @ HFError::RepoNotFound { .. } => DownloadFailure::Repository(value.to_string()),
        value @ HFError::RevisionNotFound { .. } => DownloadFailure::Revision(value.to_string()),
        value @ HFError::RateLimited { .. } => DownloadFailure::RateLimited(value.to_string()),
        value => DownloadFailure::Hub(value.to_string()),
    }
}

fn sha256(path: &Path) -> io::Result<String> {
    let mut stream = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = stream.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let digest = digest.finalize();
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to a string cannot fail");
    }
    Ok(encoded)
}

struct StagingDirectory {
    path: PathBuf,
    installed: bool,
}

impl StagingDirectory {
    fn create(parent: &Path, output_name: &std::ffi::OsStr) -> io::Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(
            ".{}.partial-{}-{nonce}-{counter}",
            output_name.to_string_lossy(),
            std::process::id()
        ));
        fs::create_dir(&path)?;
        Ok(Self {
            path,
            installed: false,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn install(&mut self, output: &Path) -> io::Result<()> {
        fs::rename(&self.path, output)?;
        self.installed = true;
        Ok(())
    }
}

impl Drop for StagingDirectory {
    fn drop(&mut self) {
        if !self.installed {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(output: PathBuf) -> ModelDownloadRequest {
        ModelDownloadRequest {
            repository: "nvidia/model".into(),
            revision: "0123456789abcdef0123456789abcdef01234567".into(),
            output,
            token_environment: "AUDIO2X_TEST_TOKEN_THAT_IS_NOT_SET".into(),
        }
    }

    #[test]
    fn presets_pin_all_original_model_repositories() {
        assert_eq!(MODEL_PRESETS.len(), 5);
        for preset in MODEL_PRESETS {
            assert_eq!(preset.revision.len(), 40);
            assert!(
                preset
                    .revision
                    .bytes()
                    .all(|value| value.is_ascii_hexdigit())
            );
            let request = preset.request("models", "TOKEN");
            assert_eq!(
                request.output,
                Path::new("models").join(preset.output_directory)
            );
            assert_eq!(model_preset(preset.name), Some(*preset));
        }
        assert_eq!(model_preset("unknown"), None);
    }

    fn test_root(name: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "audio2x-model-tool-{name}-{}-{}",
            std::process::id(),
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    #[test]
    fn rejects_non_immutable_revisions_and_missing_tokens_before_network() {
        let mut value = request(PathBuf::from("model"));
        value.revision = "main".into();
        assert!(matches!(
            value.execute(),
            Err(DownloadFailure::InvalidRevision(_))
        ));
        value.revision = "0123456789abcdef0123456789abcdef01234567".into();
        assert!(matches!(
            value.execute(),
            Err(DownloadFailure::MissingToken { .. })
        ));
    }

    #[test]
    fn installs_validated_snapshot_with_provenance_atomically() {
        let root = test_root("install");
        let output = root.join("model");
        let value = request(output.clone());
        let receipt = value
            .install_snapshot(|staging| {
                for name in REQUIRED_MODEL_FILES {
                    fs::write(staging.join(name), name.as_bytes())?;
                }
                Ok(())
            })
            .unwrap();
        assert_eq!(receipt.output, output);
        assert_eq!(receipt.network_onnx_sha256.len(), 64);
        assert!(output.join(".audio2x-source.json").is_file());
        assert!(!root.read_dir().unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("partial")
        }));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn incomplete_snapshot_is_not_installed() {
        let root = test_root("incomplete");
        let output = root.join("model");
        let value = request(output.clone());
        assert!(matches!(
            value.install_snapshot(|staging| {
                fs::write(staging.join("network.onnx"), b"onnx")?;
                Ok(())
            }),
            Err(DownloadFailure::MissingModelFile(_))
        ));
        assert!(!output.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
