use crate::sha256;
use audio2x_inference::{EngineBuildRequest, EngineBuilder, TrtBuildInfo, TrtBuildInfoError};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static ENGINE_STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

/// TensorRT precision policy and its original-SDK-compatible artifact suffix.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EnginePrecision {
    #[default]
    Default,
    Fp16,
    Fp32,
}

impl EnginePrecision {
    fn suffix(self) -> &'static str {
        match self {
            Self::Default => "",
            Self::Fp16 => "_fp16",
            Self::Fp32 => "_fp32",
        }
    }

    fn argument_group(self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Default => None,
            Self::Fp16 => Some(("fp16", "--fp16")),
            Self::Fp32 => Some(("fp32", "--noTF32")),
        }
    }

    fn engine_name(self) -> String {
        format!("network{}.trt", self.suffix())
    }

    fn model_name(self) -> String {
        format!("model{}.json", self.suffix())
    }

    fn trt_info_name(self) -> String {
        format!("trt_info{}.json", self.suffix())
    }

    fn provenance_name(self) -> String {
        match self {
            Self::Default => ".audio2x-engine.json".into(),
            Self::Fp16 => ".audio2x-engine-fp16.json".into(),
            Self::Fp32 => ".audio2x-engine-fp32.json".into(),
        }
    }
}

impl fmt::Display for EnginePrecision {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Default => "default",
            Self::Fp16 => "fp16",
            Self::Fp32 => "fp32",
        })
    }
}

impl FromStr for EnginePrecision {
    type Err = ModelEngineFailure;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "default" => Ok(Self::Default),
            "fp16" => Ok(Self::Fp16),
            "fp32" => Ok(Self::Fp32),
            _ => Err(ModelEngineFailure::InvalidPrecision(value.into())),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelEngineBuildRequest {
    pub model_directory: PathBuf,
    pub precision: EnginePrecision,
    pub device_id: u32,
    /// Overrides `MAX_BATCH_SIZE`; `OPT_BATCH_SIZE` is clamped when needed.
    pub max_batch_size: Option<u64>,
    pub replace: bool,
    pub trtexec: PathBuf,
}

impl ModelEngineBuildRequest {
    pub fn execute(&self) -> Result<EngineBuildReceipt, ModelEngineFailure> {
        let plan = BuildPlan::load(self)?;
        let executable = resolve_executable(&self.trtexec).ok_or_else(|| {
            ModelEngineFailure::ExecutableNotFound(self.trtexec.to_string_lossy().into_owned())
        })?;
        let identity = BuildIdentity::discover(&plan, self.device_id, &executable)?;
        self.execute_with(&plan, identity, |request| {
            EngineBuilder {
                executable,
                replace_existing: false,
            }
            .build(request)
            .map_err(ModelEngineFailure::Build)
        })
    }

    fn execute_with(
        &self,
        plan: &BuildPlan,
        identity: BuildIdentity,
        build: impl FnOnce(&EngineBuildRequest) -> Result<(), ModelEngineFailure>,
    ) -> Result<EngineBuildReceipt, ModelEngineFailure> {
        let generated_names = plan.generated_names();
        let had_existing = generated_names
            .iter()
            .any(|name| self.model_directory.join(name).exists());
        if had_existing && !self.replace {
            return verify_existing(plan, identity).map_err(|reason| {
                ModelEngineFailure::ExistingArtifactsMismatch {
                    directory: self.model_directory.clone(),
                    precision: self.precision,
                    reason,
                }
            });
        }

        let mut staging = EngineStagingDirectory::create(&self.model_directory, self.precision)?;
        let staged_engine = staging.path().join(&plan.engine_name);
        let mut build_arguments = plan.arguments.clone();
        if !build_arguments
            .iter()
            .any(|value| value == "--skipInference")
        {
            build_arguments.push("--skipInference".into());
        }
        build(&EngineBuildRequest {
            onnx: plan.onnx.clone(),
            engine: staged_engine.clone(),
            device_id: self.device_id,
            profiles: Vec::new(),
            extra_args: build_arguments,
        })?;
        let engine_metadata = fs::metadata(&staged_engine)?;
        if engine_metadata.len() == 0 {
            return Err(ModelEngineFailure::EmptyEngine(staged_engine));
        }

        if self.precision != EnginePrecision::Default {
            write_json(&staging.path().join(&plan.model_name), &plan.model_document)?;
            write_json(&staging.path().join(&plan.trt_info_name), &plan.trt_info)?;
        }
        let engine_sha256 = sha256(&staged_engine)?;
        let provenance = EngineProvenance {
            schema_version: 1,
            build: identity,
            engine_sha256: engine_sha256.clone(),
        };
        write_json(&staging.path().join(&plan.provenance_name), &provenance)?;
        staging.install(&generated_names, self.replace)?;

        Ok(EngineBuildReceipt {
            model_directory: self.model_directory.clone(),
            engine: self.model_directory.join(&plan.engine_name),
            model_descriptor: self.model_directory.join(&plan.model_name),
            trt_info: self.model_directory.join(&plan.trt_info_name),
            provenance: self.model_directory.join(&plan.provenance_name),
            precision: self.precision,
            max_batch_size: plan.max_batch_size,
            engine_sha256,
            disposition: if had_existing {
                EngineBuildDisposition::Replaced
            } else {
                EngineBuildDisposition::Built
            },
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EngineBuildDisposition {
    Built,
    VerifiedExisting,
    Replaced,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineBuildReceipt {
    pub model_directory: PathBuf,
    pub engine: PathBuf,
    pub model_descriptor: PathBuf,
    pub trt_info: PathBuf,
    pub provenance: PathBuf,
    pub precision: EnginePrecision,
    pub max_batch_size: Option<u64>,
    pub engine_sha256: String,
    pub disposition: EngineBuildDisposition,
}

#[derive(Clone, Debug)]
struct BuildPlan {
    model_directory: PathBuf,
    onnx: PathBuf,
    precision: EnginePrecision,
    max_batch_size: Option<u64>,
    engine_name: String,
    model_name: String,
    trt_info_name: String,
    provenance_name: String,
    model_document: Value,
    trt_info: TrtBuildInfo,
    arguments: Vec<String>,
}

impl BuildPlan {
    fn load(request: &ModelEngineBuildRequest) -> Result<Self, ModelEngineFailure> {
        if !request.model_directory.is_dir() {
            return Err(ModelEngineFailure::MissingModelDirectory(
                request.model_directory.clone(),
            ));
        }
        let onnx = request.model_directory.join("network.onnx");
        let model_path = request.model_directory.join("model.json");
        let trt_info_path = request.model_directory.join("trt_info.json");
        for path in [&onnx, &model_path, &trt_info_path] {
            if !path.is_file() {
                return Err(ModelEngineFailure::MissingInput(path.clone()));
            }
        }

        let mut model_document: Value = read_json(&model_path)?;
        let network_path = model_document
            .get_mut("networkPath")
            .ok_or_else(|| ModelEngineFailure::MissingNetworkPath(model_path.clone()))?;
        if !network_path.is_string() {
            return Err(ModelEngineFailure::MissingNetworkPath(model_path));
        }

        let mut trt_info = TrtBuildInfo::load(&trt_info_path)?;
        if let Some(max_batch_size) = request.max_batch_size {
            if max_batch_size == 0 {
                return Err(ModelEngineFailure::InvalidMaxBatchSize(max_batch_size));
            }
            let maximum = trt_info
                .defaults
                .get_mut("MAX_BATCH_SIZE")
                .ok_or(ModelEngineFailure::MissingMaxBatchDefault)?;
            *maximum = max_batch_size;
            if let Some(optimum) = trt_info.defaults.get_mut("OPT_BATCH_SIZE") {
                *optimum = (*optimum).min(max_batch_size);
            }
        }
        let base_arguments = trt_info.arguments()?;
        if let Some(argument) = base_arguments
            .iter()
            .find(|argument| is_precision_argument(argument))
        {
            return Err(ModelEngineFailure::SourcePrecisionArgument(
                argument.clone(),
            ));
        }
        if let Some((group, argument)) = request.precision.argument_group() {
            trt_info.insert_group(group, vec![argument.into()])?;
        }
        let arguments = trt_info.arguments()?;
        let engine_name = request.precision.engine_name();
        *network_path = Value::String(engine_name.clone());

        Ok(Self {
            model_directory: request.model_directory.clone(),
            onnx,
            precision: request.precision,
            max_batch_size: trt_info.defaults.get("MAX_BATCH_SIZE").copied(),
            engine_name,
            model_name: request.precision.model_name(),
            trt_info_name: request.precision.trt_info_name(),
            provenance_name: request.precision.provenance_name(),
            model_document,
            trt_info,
            arguments,
        })
    }

    fn generated_names(&self) -> Vec<String> {
        let mut names = vec![self.engine_name.clone(), self.provenance_name.clone()];
        if self.precision != EnginePrecision::Default {
            names.push(self.model_name.clone());
            names.push(self.trt_info_name.clone());
        }
        names
    }
}

fn is_precision_argument(argument: &str) -> bool {
    ["--fp16", "--bf16", "--fp8", "--int8", "--int4", "--noTF32"]
        .iter()
        .any(|flag| argument == *flag || argument.starts_with(&format!("{flag}=")))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct BuildIdentity {
    network_onnx_sha256: String,
    precision: EnginePrecision,
    device_id: u32,
    arguments: Vec<String>,
    trtexec_sha256: String,
    trtexec_version: String,
    cuda_toolkit: String,
    gpu: String,
}

impl BuildIdentity {
    fn discover(
        plan: &BuildPlan,
        device_id: u32,
        trtexec: &Path,
    ) -> Result<Self, ModelEngineFailure> {
        Ok(Self {
            network_onnx_sha256: sha256(&plan.onnx)?,
            precision: plan.precision,
            device_id,
            arguments: plan.arguments.clone(),
            trtexec_sha256: sha256(trtexec)?,
            trtexec_version: trtexec_version(trtexec),
            cuda_toolkit: cuda_toolkit_version(),
            gpu: gpu_identity(device_id),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct EngineProvenance {
    schema_version: u64,
    build: BuildIdentity,
    engine_sha256: String,
}

fn verify_existing(
    plan: &BuildPlan,
    identity: BuildIdentity,
) -> Result<EngineBuildReceipt, String> {
    for name in plan.generated_names() {
        let path = plan.model_directory.join(name);
        if !path.is_file() {
            return Err(format!(
                "required generated artifact is missing: {}",
                path.display()
            ));
        }
    }
    let engine = plan.model_directory.join(&plan.engine_name);
    if engine.metadata().map_err(|error| error.to_string())?.len() == 0 {
        return Err(format!("engine is empty: {}", engine.display()));
    }
    if plan.precision != EnginePrecision::Default {
        let actual_model: Value = read_json(&plan.model_directory.join(&plan.model_name))
            .map_err(|error| error.to_string())?;
        if actual_model != plan.model_document {
            return Err(format!("{} does not match model.json", plan.model_name));
        }
        let actual_trt_info = TrtBuildInfo::load(plan.model_directory.join(&plan.trt_info_name))
            .map_err(|error| error.to_string())?;
        if actual_trt_info != plan.trt_info {
            return Err(format!(
                "{} does not match the requested precision",
                plan.trt_info_name
            ));
        }
    }

    let provenance_path = plan.model_directory.join(&plan.provenance_name);
    let provenance: EngineProvenance =
        read_json(&provenance_path).map_err(|error| error.to_string())?;
    if provenance.schema_version != 1 {
        return Err(format!(
            "unsupported engine provenance schema {}",
            provenance.schema_version
        ));
    }
    if provenance.build != identity {
        return Err("ONNX, precision, arguments, device, or build environment changed".into());
    }
    let engine_sha256 = sha256(&engine).map_err(|error| error.to_string())?;
    if !provenance
        .engine_sha256
        .eq_ignore_ascii_case(&engine_sha256)
    {
        return Err(format!(
            "engine SHA-256 is {engine_sha256}, expected {}",
            provenance.engine_sha256
        ));
    }
    Ok(EngineBuildReceipt {
        model_directory: plan.model_directory.clone(),
        engine,
        model_descriptor: plan.model_directory.join(&plan.model_name),
        trt_info: plan.model_directory.join(&plan.trt_info_name),
        provenance: provenance_path,
        precision: plan.precision,
        max_batch_size: plan.max_batch_size,
        engine_sha256,
        disposition: EngineBuildDisposition::VerifiedExisting,
    })
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, ModelEngineFailure> {
    let payload = fs::read(path)?;
    serde_json::from_slice(&payload).map_err(|source| ModelEngineFailure::Json {
        path: path.to_owned(),
        source,
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), ModelEngineFailure> {
    let mut payload =
        serde_json::to_vec_pretty(value).map_err(|source| ModelEngineFailure::Json {
            path: path.to_owned(),
            source,
        })?;
    payload.push(b'\n');
    let mut file = File::create(path)?;
    file.write_all(&payload)?;
    file.sync_all()?;
    Ok(())
}

fn resolve_executable(executable: &Path) -> Option<PathBuf> {
    if executable.components().count() > 1 || executable.is_absolute() {
        return executable.is_file().then(|| executable.to_owned());
    }
    let path = env::var_os("PATH")?;
    let extensions = executable_extensions();
    for directory in env::split_paths(&path) {
        for extension in &extensions {
            let candidate = if extension.is_empty() {
                directory.join(executable)
            } else {
                let mut name = executable.as_os_str().to_owned();
                name.push(extension);
                directory.join(name)
            };
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn executable_extensions() -> Vec<OsString> {
    #[cfg(windows)]
    {
        let mut extensions = vec![OsString::new()];
        extensions.extend(
            env::var_os("PATHEXT")
                .map(|value| {
                    value
                        .to_string_lossy()
                        .split(';')
                        .filter(|value| !value.is_empty())
                        .map(OsString::from)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| vec![OsString::from(".EXE")]),
        );
        extensions
    }
    #[cfg(not(windows))]
    {
        vec![OsString::new()]
    }
}

fn trtexec_version(executable: &Path) -> String {
    let output = command_text(executable.as_os_str(), &[OsStr::new("--help")]);
    output
        .lines()
        .find_map(|line| {
            let (_, value) = line.split_once("[TensorRT v")?;
            let (version, _) = value.split_once(']')?;
            Some(format!("TensorRT v{version}"))
        })
        .unwrap_or_else(|| summarize_command_output(output))
}

fn cuda_toolkit_version() -> String {
    let executable = env::var_os("CUDA_PATH")
        .map(PathBuf::from)
        .map(|root| {
            root.join("bin")
                .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" })
        })
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    summarize_command_output(command_text(
        executable.as_os_str(),
        &[OsStr::new("--version")],
    ))
}

fn gpu_identity(device_id: u32) -> String {
    let query = OsString::from("--query-gpu=name,driver_version,compute_cap");
    let format = OsString::from("--format=csv,noheader");
    let id = OsString::from(format!("--id={device_id}"));
    summarize_command_output(command_text(
        OsStr::new("nvidia-smi"),
        &[query.as_os_str(), format.as_os_str(), id.as_os_str()],
    ))
}

fn command_text(executable: &OsStr, arguments: &[&OsStr]) -> String {
    match Command::new(executable).args(arguments).output() {
        Ok(output) => format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ),
        Err(error) => format!("unavailable: {error}"),
    }
}

fn summarize_command_output(output: String) -> String {
    let lines = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        "unavailable: command produced no output".into()
    } else {
        lines.join(" | ")
    }
}

struct EngineStagingDirectory {
    path: PathBuf,
    installed: bool,
    preserve: bool,
}

impl EngineStagingDirectory {
    fn create(model_directory: &Path, precision: EnginePrecision) -> io::Result<Self> {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let counter = ENGINE_STAGING_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = model_directory.join(format!(
            ".audio2x-engine-{}.partial-{}-{nonce}-{counter}",
            precision,
            std::process::id()
        ));
        fs::create_dir(&path)?;
        Ok(Self {
            path,
            installed: false,
            preserve: false,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn install(&mut self, names: &[String], replace: bool) -> Result<(), ModelEngineFailure> {
        for name in names {
            if !self.path.join(name).is_file() {
                return Err(ModelEngineFailure::MissingStagedArtifact(
                    self.path.join(name),
                ));
            }
        }
        let target_directory = self
            .path
            .parent()
            .expect("engine staging directory always has a parent")
            .to_owned();
        let backup = self.path.join("backup");
        fs::create_dir(&backup)?;
        let mut backed_up = Vec::new();
        for name in names {
            let target = target_directory.join(name);
            if target.exists() {
                if !replace {
                    return Err(ModelEngineFailure::ExistingTarget(target));
                }
                let saved = backup.join(name);
                if let Err(error) = fs::rename(&target, &saved) {
                    return self.rollback_install(
                        &target_directory,
                        &backup,
                        &backed_up,
                        &[],
                        error,
                    );
                }
                backed_up.push(name.clone());
            }
        }

        let mut published = Vec::new();
        for name in names {
            let source = self.path.join(name);
            let target = target_directory.join(name);
            if let Err(error) = fs::rename(&source, &target) {
                return self.rollback_install(
                    &target_directory,
                    &backup,
                    &backed_up,
                    &published,
                    error,
                );
            }
            published.push(name.clone());
        }
        self.installed = true;
        if let Err(source) = fs::remove_dir_all(&backup) {
            return Err(ModelEngineFailure::BackupCleanup { backup, source });
        }
        fs::remove_dir(&self.path)?;
        Ok(())
    }

    fn rollback_install<T>(
        &mut self,
        target_directory: &Path,
        backup: &Path,
        backed_up: &[String],
        published: &[String],
        install_error: io::Error,
    ) -> Result<T, ModelEngineFailure> {
        let mut rollback_errors = Vec::new();
        for name in published.iter().rev() {
            if let Err(error) = fs::remove_file(target_directory.join(name)) {
                rollback_errors.push(format!("remove {name}: {error}"));
            }
        }
        for name in backed_up.iter().rev() {
            if let Err(error) = fs::rename(backup.join(name), target_directory.join(name)) {
                rollback_errors.push(format!("restore {name}: {error}"));
            }
        }
        if rollback_errors.is_empty() {
            Err(ModelEngineFailure::Install(install_error))
        } else {
            self.preserve = true;
            Err(ModelEngineFailure::InstallRollback {
                staging: self.path.clone(),
                install_error,
                rollback_errors: rollback_errors.join("; "),
            })
        }
    }
}

impl Drop for EngineStagingDirectory {
    fn drop(&mut self) {
        if !self.installed && !self.preserve {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModelEngineFailure {
    #[error("engine precision must be default, fp16, or fp32: {0}")]
    InvalidPrecision(String),
    #[error("model directory does not exist: {}", .0.display())]
    MissingModelDirectory(PathBuf),
    #[error("engine build input does not exist: {}", .0.display())]
    MissingInput(PathBuf),
    #[error("model descriptor has no string networkPath: {}", .0.display())]
    MissingNetworkPath(PathBuf),
    #[error("base trt_info.json already selects precision with `{0}`")]
    SourcePrecisionArgument(String),
    #[error("maximum engine batch size must be greater than zero: {0}")]
    InvalidMaxBatchSize(u64),
    #[error("trt_info.json has no MAX_BATCH_SIZE default to override")]
    MissingMaxBatchDefault,
    #[error("TensorRT executable was not found: {0}; set TRTEXEC or add trtexec to PATH")]
    ExecutableNotFound(String),
    #[error(
        "existing {precision} engine artifacts in {} do not match: {reason}; use --replace to rebuild them",
        directory.display()
    )]
    ExistingArtifactsMismatch {
        directory: PathBuf,
        precision: EnginePrecision,
        reason: String,
    },
    #[error("engine target already exists: {}", .0.display())]
    ExistingTarget(PathBuf),
    #[error("generated engine is empty: {}", .0.display())]
    EmptyEngine(PathBuf),
    #[error("staged engine artifact is missing: {}", .0.display())]
    MissingStagedArtifact(PathBuf),
    #[error("unable to install generated engine artifacts: {0}")]
    Install(io::Error),
    #[error(
        "engine artifacts were installed, but the old backup could not be removed: {} ({source})",
        backup.display()
    )]
    BackupCleanup {
        backup: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "engine artifact install failed: {install_error}; rollback also failed ({rollback_errors}); recovery files are preserved at {}",
        staging.display()
    )]
    InstallRollback {
        staging: PathBuf,
        install_error: io::Error,
        rollback_errors: String,
    },
    #[error("invalid JSON {}: {source}", path.display())]
    Json {
        path: PathBuf,
        source: serde_json::Error,
    },
    #[error(transparent)]
    TrtInfo(#[from] TrtBuildInfoError),
    #[error(transparent)]
    Build(#[from] audio2x_inference::EngineError),
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root(name: &str) -> PathBuf {
        env::temp_dir().join(format!(
            "audio2x-engine-model-tool-{name}-{}-{}",
            std::process::id(),
            ENGINE_STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn write_source(directory: &Path) {
        fs::create_dir_all(directory).unwrap();
        fs::write(directory.join("network.onnx"), b"onnx").unwrap();
        fs::write(
            directory.join("model.json"),
            br#"{"networkInfoPath":"network_info.json","networkPath":"network.trt","modelConfigPath":"model_config.json"}"#,
        )
        .unwrap();
        fs::write(
            directory.join("trt_info.json"),
            br#"{"trt_build_param":{"batch":["--minShapes=input:1x2","--optShapes=input:{OPT_BATCH_SIZE}x2","--maxShapes=input:{MAX_BATCH_SIZE}x2"]},"defaults":{"OPT_BATCH_SIZE":8,"MAX_BATCH_SIZE":32}}"#,
        )
        .unwrap();
    }

    fn request(
        directory: &Path,
        precision: EnginePrecision,
        replace: bool,
    ) -> ModelEngineBuildRequest {
        ModelEngineBuildRequest {
            model_directory: directory.to_owned(),
            precision,
            device_id: 0,
            max_batch_size: None,
            replace,
            trtexec: PathBuf::from("unused-by-test"),
        }
    }

    fn identity(plan: &BuildPlan) -> BuildIdentity {
        BuildIdentity {
            network_onnx_sha256: sha256(&plan.onnx).unwrap(),
            precision: plan.precision,
            device_id: 0,
            arguments: plan.arguments.clone(),
            trtexec_sha256: "trtexec-hash".into(),
            trtexec_version: "TensorRT test".into(),
            cuda_toolkit: "CUDA test".into(),
            gpu: "GPU test".into(),
        }
    }

    #[test]
    fn fp16_build_writes_original_compatible_artifacts_and_then_skips() {
        let root = test_root("fp16");
        write_source(&root);
        let request = request(&root, EnginePrecision::Fp16, false);
        let plan = BuildPlan::load(&request).unwrap();
        let identity = identity(&plan);
        let receipt = request
            .execute_with(&plan, identity.clone(), |build| {
                assert!(build.extra_args.contains(&"--fp16".into()));
                assert!(build.extra_args.contains(&"--skipInference".into()));
                fs::write(&build.engine, b"engine")?;
                Ok(())
            })
            .unwrap();
        assert_eq!(receipt.disposition, EngineBuildDisposition::Built);
        assert!(root.join("network_fp16.trt").is_file());
        assert!(root.join("trt_info_fp16.json").is_file());
        let model: Value = read_json(&root.join("model_fp16.json")).unwrap();
        assert_eq!(model["networkPath"], "network_fp16.trt");

        let receipt = request
            .execute_with(&plan, identity, |_| {
                panic!("matching engine must be skipped")
            })
            .unwrap();
        assert_eq!(
            receipt.disposition,
            EngineBuildDisposition::VerifiedExisting
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn fp32_uses_no_tf32_and_replaces_all_precision_artifacts_together() {
        let root = test_root("fp32-replace");
        write_source(&root);
        let first = request(&root, EnginePrecision::Fp32, false);
        let first_plan = BuildPlan::load(&first).unwrap();
        let identity = identity(&first_plan);
        first
            .execute_with(&first_plan, identity.clone(), |build| {
                assert!(build.extra_args.contains(&"--noTF32".into()));
                fs::write(&build.engine, b"old-engine")?;
                Ok(())
            })
            .unwrap();
        fs::write(root.join("model_fp32.json"), b"changed").unwrap();
        assert!(matches!(
            first.execute_with(&first_plan, identity.clone(), |_| unreachable!()),
            Err(ModelEngineFailure::ExistingArtifactsMismatch { .. })
        ));

        let replacement = request(&root, EnginePrecision::Fp32, true);
        let receipt = replacement
            .execute_with(&first_plan, identity, |build| {
                fs::write(&build.engine, b"new-engine")?;
                Ok(())
            })
            .unwrap();
        assert_eq!(receipt.disposition, EngineBuildDisposition::Replaced);
        assert_eq!(
            fs::read(root.join("network_fp32.trt")).unwrap(),
            b"new-engine"
        );
        let model: Value = read_json(&root.join("model_fp32.json")).unwrap();
        assert_eq!(model["networkPath"], "network_fp32.trt");
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn default_build_preserves_downloaded_descriptors() {
        let root = test_root("default");
        write_source(&root);
        let original_model = fs::read(root.join("model.json")).unwrap();
        let original_info = fs::read(root.join("trt_info.json")).unwrap();
        let request = request(&root, EnginePrecision::Default, false);
        let plan = BuildPlan::load(&request).unwrap();
        request
            .execute_with(&plan, identity(&plan), |build| {
                fs::write(&build.engine, b"engine")?;
                Ok(())
            })
            .unwrap();
        assert_eq!(fs::read(root.join("model.json")).unwrap(), original_model);
        assert_eq!(fs::read(root.join("trt_info.json")).unwrap(), original_info);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parses_configured_installed_models_for_every_precision() {
        let Some(directories) = env::var_os("AUDIO2X_TEST_MODEL_DIRS") else {
            return;
        };
        for directory in env::split_paths(&directories) {
            for precision in [
                EnginePrecision::Default,
                EnginePrecision::Fp16,
                EnginePrecision::Fp32,
            ] {
                BuildPlan::load(&request(&directory, precision, false)).unwrap();
            }
        }
    }

    #[test]
    fn max_batch_override_clamps_optimum_and_expands_profiles() {
        let root = test_root("max-batch");
        write_source(&root);
        let mut request = request(&root, EnginePrecision::Default, false);
        request.max_batch_size = Some(4);
        let plan = BuildPlan::load(&request).unwrap();
        assert_eq!(plan.max_batch_size, Some(4));
        assert!(plan.arguments.contains(&"--optShapes=input:4x2".into()));
        assert!(plan.arguments.contains(&"--maxShapes=input:4x2".into()));
        fs::remove_dir_all(root).unwrap();
    }
}
