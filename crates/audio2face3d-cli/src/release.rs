//! Release-baseline and performance-regression validation.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::Path;

const SCHEMA_VERSION: u32 = 1;
const REQUIRED_FEATURES: &[&str] = &["animation", "emotion", "cuda", "tensorrt"];
const REQUIRED_WORKLOADS: &[&str] = &[
    "regression",
    "diffusion",
    "audio2emotion",
    "blendshape-cpu",
    "blendshape-gpu",
    "interactive-gpu-replay",
];
const REQUIRED_TIERS: &[&str] = &[
    "portable",
    "cuda-lifetime",
    "tensorrt-model",
    "reference-parity",
    "release",
];

#[derive(Debug, Deserialize)]
struct ReleaseBaseline {
    schema_version: u32,
    release_version: String,
    minimum_rust_version: String,
    supported_environments: Vec<SupportedEnvironment>,
    features: BTreeMap<String, FeatureContract>,
    ci_tiers: Vec<CiTier>,
    provenance: ProvenanceContract,
    benchmark: BenchmarkContract,
    api_policy: ApiPolicy,
    safety_invariants: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SupportedEnvironment {
    target: String,
    toolchain: String,
    cuda: String,
    tensorrt: String,
    compute_capabilities: Vec<String>,
    precisions: Vec<String>,
    status: String,
}

#[derive(Debug, Deserialize)]
struct FeatureContract {
    requires: Vec<String>,
    tier: String,
}

#[derive(Debug, Deserialize)]
struct CiTier {
    name: String,
    required: bool,
    command: Vec<String>,
    environment: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ProvenanceContract {
    sdk: SourceContract,
    models: Vec<ModelContract>,
    artifact_manifest: String,
    fixture_license_required: bool,
    generated_sidecars: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SourceContract {
    repository: String,
    revision: String,
    license: String,
}

#[derive(Debug, Deserialize)]
struct ModelContract {
    name: String,
    repository: String,
    revision: String,
    license: String,
}

#[derive(Debug, Deserialize)]
struct BenchmarkContract {
    baseline: String,
    required_workloads: Vec<String>,
    maximum_latency_regression_percent: f64,
    maximum_throughput_regression_percent: f64,
    maximum_memory_regression_percent: f64,
}

#[derive(Debug, Deserialize)]
struct ApiPolicy {
    semver: String,
    breaking_changes_before_1_0: String,
    public_api_review_required: bool,
    feature_removal_is_breaking: bool,
}

#[derive(Debug, Deserialize)]
struct SdkArtifactManifest {
    schema_version: u32,
    sources: SdkSources,
    cases: BTreeMap<String, SdkArtifactCase>,
}

#[derive(Debug, Deserialize)]
struct SdkSources {
    audio2face_sdk: String,
}

#[derive(Debug, Deserialize)]
struct SdkArtifactCase {
    roots: Vec<String>,
    files: Vec<SdkArtifactFile>,
}

#[derive(Debug, Deserialize)]
struct SdkArtifactFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ReleaseAuditReport {
    pub schema_version: u32,
    pub compatible: bool,
    pub baseline: String,
    pub checks: usize,
    pub failures: Vec<String>,
}

pub fn audit_release_baseline(
    workspace: &Path,
    baseline_path: &Path,
) -> io::Result<ReleaseAuditReport> {
    let baseline: ReleaseBaseline = read_json(baseline_path)?;
    let mut audit = Audit::default();
    audit.require(
        baseline.schema_version == SCHEMA_VERSION,
        "unsupported release baseline schema",
    );
    audit.require(
        baseline.release_version == env!("CARGO_PKG_VERSION"),
        "release baseline version differs from the crate version",
    );
    audit.require(
        baseline.minimum_rust_version == env!("CARGO_PKG_RUST_VERSION"),
        "minimum Rust version differs from the crate manifest",
    );
    audit.require(
        baseline.supported_environments.iter().any(|environment| {
            environment.target == "x86_64-pc-windows-msvc"
                && environment.status == "required"
                && environment.precisions.iter().any(|value| value == "fp32")
                && environment.precisions.iter().any(|value| value == "fp16")
                && !environment.toolchain.is_empty()
                && !environment.cuda.is_empty()
                && !environment.tensorrt.is_empty()
                && !environment.compute_capabilities.is_empty()
        }),
        "required Windows/MSVC CUDA/TensorRT FP32/FP16 environment is missing",
    );
    for feature in REQUIRED_FEATURES {
        audit.require(
            baseline.features.contains_key(*feature),
            format!("feature `{feature}` is missing from the release contract"),
        );
    }
    for (name, feature) in &baseline.features {
        audit.require(
            !feature.tier.trim().is_empty(),
            format!("feature `{name}` has no CI tier"),
        );
        audit.require(
            feature
                .requires
                .iter()
                .all(|value| !value.trim().is_empty()),
            format!("feature `{name}` contains an empty requirement"),
        );
    }
    let tiers = baseline
        .ci_tiers
        .iter()
        .map(|tier| tier.name.as_str())
        .collect::<BTreeSet<_>>();
    for tier in REQUIRED_TIERS {
        audit.require(tiers.contains(tier), format!("CI tier `{tier}` is missing"));
    }
    for tier in &baseline.ci_tiers {
        audit.require(
            !tier.command.is_empty(),
            format!("CI tier `{}` has no command", tier.name),
        );
        if tier.required && tier.name != "portable" {
            audit.require(
                !tier.environment.is_empty(),
                format!(
                    "required native CI tier `{}` has no environment contract",
                    tier.name
                ),
            );
        }
    }
    validate_source(&mut audit, "SDK", &baseline.provenance.sdk);
    audit.require(
        baseline.provenance.fixture_license_required,
        "fixture license is not required by the provenance contract",
    );
    audit.require(
        baseline
            .provenance
            .generated_sidecars
            .iter()
            .any(|value| value == ".audio2x-source.json")
            && baseline
                .provenance
                .generated_sidecars
                .iter()
                .any(|value| value == ".audio2x-engine*.json"),
        "model and engine provenance sidecars are not both required",
    );
    let mut model_names = BTreeSet::new();
    for model in &baseline.provenance.models {
        audit.require(
            model_names.insert(model.name.as_str()),
            format!("duplicate model provenance `{}`", model.name),
        );
        validate_revision_license(
            &mut audit,
            &format!("model `{}`", model.name),
            &model.repository,
            &model.revision,
            &model.license,
        );
    }
    for workload in REQUIRED_WORKLOADS {
        audit.require(
            baseline
                .benchmark
                .required_workloads
                .iter()
                .any(|value| value == workload),
            format!("benchmark workload `{workload}` is missing"),
        );
    }
    audit.require(
        baseline.benchmark.maximum_latency_regression_percent > 0.0
            && baseline.benchmark.maximum_throughput_regression_percent > 0.0
            && baseline.benchmark.maximum_memory_regression_percent > 0.0,
        "benchmark regression thresholds must be positive",
    );
    audit.require(
        baseline.api_policy.semver == "SemVer 2.0.0"
            && !baseline.api_policy.breaking_changes_before_1_0.is_empty()
            && baseline.api_policy.public_api_review_required
            && baseline.api_policy.feature_removal_is_breaking,
        "public API and versioning policy is incomplete",
    );
    audit.require(
        baseline.safety_invariants.len() >= 6,
        "fewer than six release safety invariants are documented",
    );

    let artifact_path = workspace.join(&baseline.provenance.artifact_manifest);
    let artifacts: SdkArtifactManifest = read_json(&artifact_path)?;
    audit.require(
        artifacts.schema_version == SCHEMA_VERSION,
        "unsupported SDK artifact manifest schema",
    );
    audit.require(
        artifacts
            .sources
            .audio2face_sdk
            .eq_ignore_ascii_case(&baseline.provenance.sdk.revision),
        "SDK artifact revision differs from the release baseline",
    );
    for required in ["regression", "diffusion", "a2e"] {
        audit.require(
            artifacts.cases.contains_key(required),
            format!("SDK artifact case `{required}` is missing"),
        );
    }
    for (case, value) in artifacts.cases {
        audit.require(
            !value.roots.is_empty() && !value.files.is_empty(),
            format!("SDK artifact case `{case}` is empty"),
        );
        let mut paths = BTreeSet::new();
        for file in value.files {
            audit.require(
                paths.insert(file.path.clone()),
                format!("duplicate SDK artifact `{}`", file.path),
            );
            audit.require(
                is_sha256(&file.sha256),
                format!("SDK artifact `{}` has an invalid SHA-256", file.path),
            );
            if file.bytes == 0 {
                audit.require(
                    file.sha256.eq_ignore_ascii_case(
                        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    ),
                    format!("zero-byte SDK artifact `{}` has the wrong hash", file.path),
                );
            }
        }
    }
    let benchmark_path = workspace.join(&baseline.benchmark.baseline);
    let benchmarks: BenchmarkBaseline = read_json(&benchmark_path)?;
    audit.require(
        benchmarks.schema_version == SCHEMA_VERSION,
        "unsupported benchmark baseline schema",
    );
    audit.require(
        [
            &benchmarks.environment.target,
            &benchmarks.environment.gpu,
            &benchmarks.environment.driver,
            &benchmarks.environment.cuda,
            &benchmarks.environment.tensorrt,
        ]
        .into_iter()
        .all(|value| !value.trim().is_empty()),
        "benchmark environment is incomplete",
    );
    let mut workload_names = BTreeSet::new();
    for workload in &benchmarks.workloads {
        audit.require(
            workload_names.insert(workload.name.as_str()),
            format!("duplicate benchmark workload `{}`", workload.name),
        );
        audit.require(
            !workload.scope.trim().is_empty() && !workload.capture_command.is_empty(),
            format!("benchmark workload `{}` cannot be captured", workload.name),
        );
    }
    for required in REQUIRED_WORKLOADS {
        audit.require(
            workload_names.contains(required),
            format!("benchmark workload definition `{required}` is missing"),
        );
    }
    let mut benchmark_cases = BTreeSet::new();
    let mut measured_workloads = BTreeSet::new();
    for case in &benchmarks.cases {
        audit.require(
            benchmark_cases.insert(case.name.as_str()),
            format!("duplicate benchmark case `{}`", case.name),
        );
        let workload = match (case.scope.as_str(), case.pipeline.as_str()) {
            ("raw-network", "emotion") => "audio2emotion",
            ("raw-network", pipeline) => pipeline,
            (scope, _) => scope,
        };
        measured_workloads.insert(workload);
        audit.require(
            !case.pipeline.trim().is_empty()
                && !case.precision.trim().is_empty()
                && case.tracks > 0
                && !case.scope.trim().is_empty()
                && is_sha256(&case.engine_sha256)
                && !case.phases.is_empty(),
            format!("benchmark case `{}` is incomplete", case.name),
        );
        for (phase, values) in &case.phases {
            audit.require(
                values.p50_ns > 0
                    && values.p95_ns >= values.p50_ns
                    && values.p99_ns >= values.p95_ns
                    && values
                        .throughput_per_second
                        .is_none_or(|value| value.is_finite() && value > 0.0),
                format!("benchmark case `{}` phase `{phase}` is invalid", case.name),
            );
        }
    }
    for required in REQUIRED_WORKLOADS {
        audit.require(
            measured_workloads.contains(required),
            format!("benchmark workload `{required}` has no measured baseline"),
        );
    }
    audit.require(
        workspace.join("LICENSE").is_file(),
        "workspace LICENSE is missing",
    );
    audit.require(
        workspace.join("README.md").is_file(),
        "workspace README is missing",
    );

    Ok(ReleaseAuditReport {
        schema_version: SCHEMA_VERSION,
        compatible: audit.failures.is_empty(),
        baseline: baseline_path.display().to_string(),
        checks: audit.checks,
        failures: audit.failures,
    })
}

fn validate_source(audit: &mut Audit, label: &str, source: &SourceContract) {
    validate_revision_license(
        audit,
        label,
        &source.repository,
        &source.revision,
        &source.license,
    );
}

fn validate_revision_license(
    audit: &mut Audit,
    label: &str,
    repository: &str,
    revision: &str,
    license: &str,
) {
    audit.require(
        !repository.trim().is_empty(),
        format!("{label} repository is empty"),
    );
    audit.require(
        revision.len() == 40 && revision.bytes().all(|value| value.is_ascii_hexdigit()),
        format!("{label} revision is not an immutable commit"),
    );
    audit.require(
        !license.trim().is_empty(),
        format!("{label} license is empty"),
    );
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkBaseline {
    pub schema_version: u32,
    pub environment: BenchmarkEnvironment,
    pub workloads: Vec<BenchmarkWorkload>,
    pub cases: Vec<BenchmarkCase>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkWorkload {
    pub name: String,
    pub scope: String,
    pub capture_command: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkEnvironment {
    pub target: String,
    pub gpu: String,
    pub driver: String,
    pub cuda: String,
    pub tensorrt: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkCase {
    pub name: String,
    pub pipeline: String,
    pub precision: String,
    pub tracks: usize,
    pub scope: String,
    pub engine_sha256: String,
    pub phases: BTreeMap<String, BenchmarkPhaseBaseline>,
    pub peak_memory_mib: Option<u64>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct BenchmarkPhaseBaseline {
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub throughput_per_second: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct BenchmarkCandidate {
    schema_version: u32,
    pipeline: String,
    precision: String,
    tracks: usize,
    scope: String,
    engine_sha256: String,
    report: CandidateReport,
}

#[derive(Debug, Deserialize)]
struct CandidateReport {
    peak_memory_mib: Option<u64>,
    phases: Vec<CandidatePhase>,
}

#[derive(Debug, Deserialize)]
struct CandidatePhase {
    name: String,
    p50_ns: u64,
    p95_ns: u64,
    p99_ns: u64,
    throughput_per_second: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct BenchmarkComparisonReport {
    pub schema_version: u32,
    pub compatible: bool,
    pub case: String,
    pub maximum_latency_regression_percent: f64,
    pub maximum_throughput_regression_percent: f64,
    pub maximum_memory_regression_percent: f64,
    pub regressions: Vec<String>,
}

pub fn compare_benchmark(
    baseline_path: &Path,
    candidate_path: &Path,
    maximum_latency_regression_percent: f64,
    maximum_throughput_regression_percent: f64,
    maximum_memory_regression_percent: f64,
) -> io::Result<BenchmarkComparisonReport> {
    if maximum_latency_regression_percent < 0.0
        || maximum_throughput_regression_percent < 0.0
        || maximum_memory_regression_percent < 0.0
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "benchmark thresholds must not be negative",
        ));
    }
    let baseline: BenchmarkBaseline = read_json(baseline_path)?;
    let candidate: BenchmarkCandidate = read_json(candidate_path)?;
    if baseline.schema_version != SCHEMA_VERSION || candidate.schema_version != SCHEMA_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported benchmark schema",
        ));
    }
    let expected = baseline
        .cases
        .iter()
        .find(|case| {
            case.pipeline == candidate.pipeline
                && case.precision == candidate.precision
                && case.tracks == candidate.tracks
                && case.scope == candidate.scope
        })
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "benchmark case is not baselined")
        })?;
    let mut regressions = Vec::new();
    if !expected
        .engine_sha256
        .eq_ignore_ascii_case(&candidate.engine_sha256)
    {
        regressions.push("engine SHA-256 differs from the baseline".into());
    }
    let candidate_phases = candidate
        .report
        .phases
        .iter()
        .map(|phase| (phase.name.as_str(), phase))
        .collect::<BTreeMap<_, _>>();
    for (name, expected_phase) in &expected.phases {
        let Some(actual) = candidate_phases.get(name.as_str()) else {
            regressions.push(format!("phase `{name}` is missing"));
            continue;
        };
        for (percentile, expected_value, actual_value) in [
            ("p50", expected_phase.p50_ns, actual.p50_ns),
            ("p95", expected_phase.p95_ns, actual.p95_ns),
            ("p99", expected_phase.p99_ns, actual.p99_ns),
        ] {
            if exceeds(
                expected_value,
                actual_value,
                maximum_latency_regression_percent,
            ) {
                regressions.push(format!(
                    "phase `{name}` {percentile} regressed from {expected_value} ns to {actual_value} ns"
                ));
            }
        }
        if let Some(expected_throughput) = expected_phase.throughput_per_second
            && actual.throughput_per_second
                < expected_throughput * (1.0 - maximum_throughput_regression_percent / 100.0)
        {
            regressions.push(format!(
                "phase `{name}` throughput regressed from {expected_throughput:.3}/s to {:.3}/s",
                actual.throughput_per_second
            ));
        }
    }
    if let (Some(expected_memory), Some(actual_memory)) =
        (expected.peak_memory_mib, candidate.report.peak_memory_mib)
        && exceeds(
            expected_memory,
            actual_memory,
            maximum_memory_regression_percent,
        )
    {
        regressions.push(format!(
            "peak memory regressed from {expected_memory} MiB to {actual_memory} MiB"
        ));
    }
    Ok(BenchmarkComparisonReport {
        schema_version: SCHEMA_VERSION,
        compatible: regressions.is_empty(),
        case: expected.name.clone(),
        maximum_latency_regression_percent,
        maximum_throughput_regression_percent,
        maximum_memory_regression_percent,
        regressions,
    })
}

pub fn write_json(path: &Path, value: &impl Serialize) -> io::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    bytes.push(b'\n');
    fs::write(path, bytes)
}

fn exceeds(expected: u64, actual: u64, percent: f64) -> bool {
    actual as f64 > expected as f64 * (1.0 + percent / 100.0)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> io::Result<T> {
    serde_json::from_slice(&fs::read(path)?)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(&mut result, "{byte:02X}").expect("writing to a string cannot fail");
    }
    result
}

#[derive(Default)]
struct Audit {
    checks: usize,
    failures: Vec<String>,
}

impl Audit {
    fn require(&mut self, condition: bool, message: impl Into<String>) {
        self.checks += 1;
        if !condition {
            self.failures.push(message.into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracked_release_baseline_and_artifacts_are_consistent() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let baseline = workspace.join("release/release-baseline.json");
        let report = audit_release_baseline(&workspace, &baseline).unwrap();
        assert!(report.compatible, "{:?}", report.failures);
        assert!(report.checks > 100);
    }

    #[test]
    fn benchmark_thresholds_are_independent_from_numeric_parity() {
        assert!(!exceeds(100, 110, 10.0));
        assert!(exceeds(100, 111, 10.0));
    }

    #[test]
    fn comparison_detects_throughput_and_memory_independently() {
        let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let baseline = workspace.join("reference/benchmark-baseline.json");
        let candidate = std::env::temp_dir().join(format!(
            "audio2face3d-benchmark-candidate-{}.json",
            std::process::id()
        ));
        let document = serde_json::json!({
            "schema_version": 1,
            "pipeline": "regression",
            "precision": "fp32",
            "tracks": 1,
            "scope": "raw-network",
            "engine_sha256": "81d12167c8219125637aec55ec66cad50539d3e4c78a99626657b4eff4285cd9",
            "report": {
                "peak_memory_mib": 2000,
                "phases": [
                    { "name": "steady-state", "p50_ns": 3462400, "p95_ns": 3804700, "p99_ns": 3804700, "throughput_per_second": 100.0 },
                    { "name": "post-process", "p50_ns": 100, "p95_ns": 100, "p99_ns": 100, "throughput_per_second": 10000000.0 },
                    { "name": "end-to-end", "p50_ns": 3366600, "p95_ns": 3553900, "p99_ns": 3553900, "throughput_per_second": 100.0 }
                ]
            }
        });
        write_json(&candidate, &document).unwrap();
        let report = compare_benchmark(&baseline, &candidate, 15.0, 15.0, 10.0).unwrap();
        let _ = fs::remove_file(candidate);
        assert!(!report.compatible);
        assert!(
            report
                .regressions
                .iter()
                .any(|value| value.contains("throughput"))
        );
        assert!(
            report
                .regressions
                .iter()
                .any(|value| value.contains("memory"))
        );
        assert!(!report.regressions.iter().any(|value| value.contains("p50")));
    }
}
