#![allow(dead_code, unused_imports)]
//! Rust-native model acquisition for Audio2Face-3D.

mod engine;
pub mod reference;
pub mod release;

pub use engine::{
    EngineBuildDisposition, EngineBuildReceipt, EnginePrecision, ModelEngineBuildRequest,
    ModelEngineFailure,
};

use hf_hub::{HFClient, HFClientSync, HFError, progress::Progress, split_id};
use serde::Deserialize;
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
    pub disposition: DownloadDisposition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DownloadDisposition {
    Downloaded,
    VerifiedExisting,
    Replaced,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DownloadOptions {
    pub force: bool,
}

#[derive(Deserialize)]
struct SourceMetadata {
    schema_version: u64,
    repository: String,
    revision: String,
    network_onnx_sha256: String,
}

impl ModelDownloadRequest {
    /// Downloads one immutable Hub revision without invoking Python or an
    /// external Hugging Face CLI. The completed snapshot becomes visible at
    /// `output` only after its required files and ONNX digest are validated.
    pub fn execute(&self) -> Result<DownloadReceipt, DownloadFailure> {
        self.execute_with_options(DownloadOptions::default(), None)
    }

    pub fn execute_with_progress(
        &self,
        progress: impl Into<Progress>,
    ) -> Result<DownloadReceipt, DownloadFailure> {
        self.execute_with_options(DownloadOptions::default(), Some(progress.into()))
    }

    pub fn execute_with_options(
        &self,
        options: DownloadOptions,
        progress: Option<Progress>,
    ) -> Result<DownloadReceipt, DownloadFailure> {
        self.validate()?;
        if self.output.exists() {
            if !options.force {
                return self.verify_existing();
            }
            if !self.output.is_dir() {
                return Err(DownloadFailure::ExistingOutputMismatch {
                    output: self.output.clone(),
                    reason: "the output is not a directory".into(),
                });
            }
        }
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
                .user_agent(format!("audio2face3d-cli/{}", env!("CARGO_PKG_VERSION")))
                .build()
                .map_err(map_hub_error)?,
        )
        .map_err(map_hub_error)?;
        let (owner, name) = split_id(&self.repository);
        let repository = client.model(owner, name);
        self.install_snapshot(options.force, |staging| {
            let download = repository
                .snapshot_download()
                .revision(self.revision.clone())
                .local_dir(staging.to_owned())
                .max_workers(8);
            match progress {
                Some(progress) => download.progress(progress).send(),
                None => download.send(),
            }
            .map_err(map_hub_error)?;
            Ok(())
        })
    }

    fn verify_existing(&self) -> Result<DownloadReceipt, DownloadFailure> {
        if !self.output.is_dir() {
            return Err(self.existing_mismatch("the output is not a directory"));
        }
        for name in REQUIRED_MODEL_FILES {
            if !self.output.join(name).is_file() {
                return Err(self.existing_mismatch(format!("required file `{name}` is missing")));
            }
        }
        // Preserve the established on-disk name so existing model snapshots remain verifiable.
        let metadata_path = self.output.join(".audio2x-source.json");
        let payload = fs::read(&metadata_path).map_err(|error| {
            self.existing_mismatch(format!(
                "cannot read `{}`: {error}",
                metadata_path.display()
            ))
        })?;
        let metadata: SourceMetadata = serde_json::from_slice(&payload).map_err(|error| {
            self.existing_mismatch(format!(
                "cannot parse `{}`: {error}",
                metadata_path.display()
            ))
        })?;
        if metadata.schema_version != 1 {
            return Err(self.existing_mismatch(format!(
                "unsupported provenance schema {}",
                metadata.schema_version
            )));
        }
        if metadata.repository != self.repository {
            return Err(self.existing_mismatch(format!(
                "repository is `{}`, expected `{}`",
                metadata.repository, self.repository
            )));
        }
        if metadata.revision != self.revision {
            return Err(self.existing_mismatch(format!(
                "revision is `{}`, expected `{}`",
                metadata.revision, self.revision
            )));
        }
        let actual_hash = sha256(&self.output.join("network.onnx"))?;
        if metadata.network_onnx_sha256.len() != 64
            || !metadata
                .network_onnx_sha256
                .eq_ignore_ascii_case(&actual_hash)
        {
            return Err(self.existing_mismatch(format!(
                "network.onnx SHA-256 is `{actual_hash}`, expected `{}`",
                metadata.network_onnx_sha256
            )));
        }
        Ok(DownloadReceipt {
            repository: self.repository.clone(),
            revision: self.revision.clone(),
            output: self.output.clone(),
            network_onnx_sha256: actual_hash,
            disposition: DownloadDisposition::VerifiedExisting,
        })
    }

    fn existing_mismatch(&self, reason: impl Into<String>) -> DownloadFailure {
        DownloadFailure::ExistingOutputMismatch {
            output: self.output.clone(),
            reason: reason.into(),
        }
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
        replace_existing: bool,
        download: impl FnOnce(&Path) -> Result<(), DownloadFailure>,
    ) -> Result<DownloadReceipt, DownloadFailure> {
        if self.output.exists() && !replace_existing {
            return Err(self.existing_mismatch("the output appeared while downloading"));
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
        // This filename is part of the existing persistent provenance format.
        let mut stream = File::create(staging.path().join(".audio2x-source.json"))?;
        stream.write_all(&payload)?;
        stream.sync_all()?;
        drop(stream);
        let replaced = staging.install(&self.output, replace_existing)?;
        Ok(DownloadReceipt {
            repository: self.repository.clone(),
            revision: self.revision.clone(),
            output: self.output.clone(),
            network_onnx_sha256,
            disposition: if replaced {
                DownloadDisposition::Replaced
            } else {
                DownloadDisposition::Downloaded
            },
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
    #[error(
        "existing model output does not match the requested snapshot: {} ({reason}); use --force to replace it",
        output.display()
    )]
    ExistingOutputMismatch { output: PathBuf, reason: String },
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
    #[error(
        "model replacement failed for {}: {replace_error}; rollback from {} also failed: {rollback_error}",
        output.display(),
        backup.display()
    )]
    ReplacementRollback {
        output: PathBuf,
        backup: PathBuf,
        replace_error: io::Error,
        rollback_error: io::Error,
    },
    #[error(
        "model was replaced, but the old backup could not be removed: {} ({source})",
        backup.display()
    )]
    BackupCleanup {
        backup: PathBuf,
        #[source]
        source: io::Error,
    },
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

    fn install(&mut self, output: &Path, replace_existing: bool) -> Result<bool, DownloadFailure> {
        if !output.exists() {
            fs::rename(&self.path, output)?;
            self.installed = true;
            return Ok(false);
        }
        if !replace_existing {
            return Err(DownloadFailure::ExistingOutputMismatch {
                output: output.to_owned(),
                reason: "the output appeared while downloading".into(),
            });
        }

        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        let backup = unique_sibling(parent, output.file_name().unwrap(), "backup");
        fs::rename(output, &backup)?;
        if let Err(replace_error) = fs::rename(&self.path, output) {
            if let Err(rollback_error) = fs::rename(&backup, output) {
                return Err(DownloadFailure::ReplacementRollback {
                    output: output.to_owned(),
                    backup,
                    replace_error,
                    rollback_error,
                });
            }
            return Err(DownloadFailure::Io(replace_error));
        }
        self.installed = true;
        fs::remove_dir_all(&backup)
            .map_err(|source| DownloadFailure::BackupCleanup { backup, source })?;
        Ok(true)
    }
}

fn unique_sibling(parent: &Path, output_name: &std::ffi::OsStr, kind: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let counter = STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
    parent.join(format!(
        ".{}.{kind}-{}-{nonce}-{counter}",
        output_name.to_string_lossy(),
        std::process::id()
    ))
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
            token_environment: "AUDIO2FACE3D_TEST_TOKEN_THAT_IS_NOT_SET".into(),
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
            "audio2face3d-cli-{name}-{}-{}",
            std::process::id(),
            STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn write_model_files(directory: &Path, contents: &[u8]) -> Result<(), DownloadFailure> {
        for name in REQUIRED_MODEL_FILES {
            fs::write(directory.join(name), contents)?;
        }
        Ok(())
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
            .install_snapshot(false, |staging| write_model_files(staging, b"model"))
            .unwrap();
        assert_eq!(receipt.output, output);
        assert_eq!(receipt.disposition, DownloadDisposition::Downloaded);
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
            value.install_snapshot(false, |staging| {
                fs::write(staging.join("network.onnx"), b"onnx")?;
                Ok(())
            }),
            Err(DownloadFailure::MissingModelFile(_))
        ));
        assert!(!output.exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn matching_existing_snapshot_is_verified_and_skipped_without_a_token() {
        let root = test_root("skip");
        let output = root.join("model");
        let value = request(output.clone());
        value
            .install_snapshot(false, |staging| write_model_files(staging, b"model"))
            .unwrap();

        let receipt = value.execute().unwrap();
        assert_eq!(receipt.disposition, DownloadDisposition::VerifiedExisting);
        assert_eq!(receipt.output, output);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn changed_existing_onnx_requires_force() {
        let root = test_root("changed");
        let output = root.join("model");
        let value = request(output.clone());
        value
            .install_snapshot(false, |staging| write_model_files(staging, b"model"))
            .unwrap();
        fs::write(output.join("network.onnx"), b"changed").unwrap();

        assert!(matches!(
            value.execute(),
            Err(DownloadFailure::ExistingOutputMismatch { .. })
        ));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn forced_install_validates_before_replacing_existing_directory() {
        let root = test_root("replace");
        let output = root.join("model");
        fs::create_dir_all(&output).unwrap();
        fs::write(output.join("old-marker"), b"old").unwrap();
        let value = request(output.clone());

        let receipt = value
            .install_snapshot(true, |staging| {
                assert!(output.join("old-marker").is_file());
                write_model_files(staging, b"new")
            })
            .unwrap();
        assert_eq!(receipt.disposition, DownloadDisposition::Replaced);
        assert_eq!(fs::read(output.join("network.onnx")).unwrap(), b"new");
        assert!(!output.join("old-marker").exists());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn failed_forced_download_preserves_existing_directory() {
        let root = test_root("replace-failure");
        let output = root.join("model");
        fs::create_dir_all(&output).unwrap();
        fs::write(output.join("old-marker"), b"old").unwrap();
        let value = request(output.clone());

        assert!(matches!(
            value.install_snapshot(true, |staging| {
                fs::write(staging.join("network.onnx"), b"incomplete")?;
                Ok(())
            }),
            Err(DownloadFailure::MissingModelFile(_))
        ));
        assert_eq!(fs::read(output.join("old-marker")).unwrap(), b"old");
        fs::remove_dir_all(root).unwrap();
    }
}
