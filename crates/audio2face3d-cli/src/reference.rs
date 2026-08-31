//! Stable, implementation-neutral artifacts used to compare the Rust and C++ SDKs.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const FIXTURE_SCHEMA_VERSION: u32 = 1;
pub const SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FileProvenance {
    pub path: String,
    pub sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FixtureManifest {
    pub schema_version: u32,
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_count: usize,
    pub encoding: String,
    pub samples_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<FileProvenance>,
    pub license: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Producer {
    pub implementation: String,
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Case {
    pub name: String,
    pub pipeline: String,
    pub execution: String,
    pub precision: String,
    pub seed: u64,
    pub track_count: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Record {
    pub sequence: usize,
    pub layer: String,
    pub component: String,
    pub track: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_timestamp: Option<i64>,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub offset_bytes: u64,
    pub byte_length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub producer: Producer,
    pub case: Case,
    pub fixture: FileProvenance,
    #[serde(default)]
    pub model_files: BTreeMap<String, FileProvenance>,
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    #[serde(default)]
    pub counters: BTreeMap<String, u64>,
    pub records: Vec<Record>,
    pub data_sha256: String,
}

#[derive(Clone, Debug)]
pub struct RecordMetadata {
    pub layer: String,
    pub component: String,
    pub track: usize,
    pub frame: Option<usize>,
    pub inference: Option<usize>,
    pub timestamp: Option<i64>,
    pub next_timestamp: Option<i64>,
    pub shape: Vec<usize>,
}

pub struct ArtifactWriter {
    root: PathBuf,
    manifest: ArtifactManifest,
    data: File,
    data_digest: Sha256,
    offset: u64,
}

impl ArtifactWriter {
    pub fn create(
        root: impl AsRef<Path>,
        producer: Producer,
        case: Case,
        fixture: FileProvenance,
    ) -> io::Result<Self> {
        let root = root.as_ref().to_owned();
        fs::create_dir_all(&root)?;
        let data = File::create(root.join("values.f32le"))?;
        Ok(Self {
            root,
            manifest: ArtifactManifest {
                schema_version: ARTIFACT_SCHEMA_VERSION,
                producer,
                case,
                fixture,
                model_files: BTreeMap::new(),
                environment: BTreeMap::new(),
                counters: BTreeMap::new(),
                records: Vec::new(),
                data_sha256: String::new(),
            },
            data,
            data_digest: Sha256::new(),
            offset: 0,
        })
    }

    pub fn manifest_mut(&mut self) -> &mut ArtifactManifest {
        &mut self.manifest
    }

    pub fn push_f32(&mut self, metadata: RecordMetadata, values: &[f32]) -> io::Result<()> {
        let expected = metadata
            .shape
            .iter()
            .try_fold(1_usize, |total, &dimension| total.checked_mul(dimension));
        if expected != Some(values.len()) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "record shape does not match its f32 value count",
            ));
        }
        let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.data.write_all(&bytes)?;
        self.data_digest.update(&bytes);
        let byte_length = u64::try_from(bytes.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "record is too large"))?;
        self.manifest.records.push(Record {
            sequence: self.manifest.records.len(),
            layer: metadata.layer,
            component: metadata.component,
            track: metadata.track,
            frame: metadata.frame,
            inference: metadata.inference,
            timestamp: metadata.timestamp,
            next_timestamp: metadata.next_timestamp,
            dtype: "f32le".into(),
            shape: metadata.shape,
            offset_bytes: self.offset,
            byte_length,
            sha256: encode_digest(Sha256::digest(&bytes)),
        });
        self.offset = self
            .offset
            .checked_add(byte_length)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "artifact is too large"))?;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<ArtifactManifest> {
        self.data.flush()?;
        self.data.sync_all()?;
        self.manifest.data_sha256 = encode_digest(self.data_digest.finalize());
        write_json(&self.root.join("artifact.json"), &self.manifest)?;
        Ok(self.manifest)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct Tolerance {
    pub absolute: f64,
    pub relative: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToleranceProfile {
    pub schema_version: u32,
    pub exact_metadata: bool,
    pub default: BTreeMap<String, Tolerance>,
    #[serde(default)]
    pub components: BTreeMap<String, BTreeMap<String, Tolerance>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Difference {
    pub record: usize,
    pub layer: String,
    pub component: String,
    pub index: usize,
    pub expected: f32,
    pub actual: f32,
    pub absolute_error: f64,
    pub allowed_error: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ComparisonReport {
    pub schema_version: u32,
    pub compatible: bool,
    pub expected: String,
    pub actual: String,
    pub records_compared: usize,
    pub values_compared: usize,
    pub maximum_absolute_error: f64,
    pub maximum_relative_error: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_difference: Option<Difference>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub structural_differences: Vec<String>,
}

pub fn compare_artifacts(
    expected_root: &Path,
    actual_root: &Path,
    tolerances: &ToleranceProfile,
) -> io::Result<ComparisonReport> {
    let expected = read_artifact(expected_root)?;
    let actual = read_artifact(actual_root)?;
    let mut report = ComparisonReport {
        schema_version: ARTIFACT_SCHEMA_VERSION,
        compatible: true,
        expected: expected_root.display().to_string(),
        actual: actual_root.display().to_string(),
        records_compared: 0,
        values_compared: 0,
        maximum_absolute_error: 0.0,
        maximum_relative_error: 0.0,
        first_difference: None,
        structural_differences: Vec::new(),
    };
    if expected.manifest.case != actual.manifest.case {
        report
            .structural_differences
            .push("case metadata differs".into());
    }
    if expected.manifest.fixture.sha256 != actual.manifest.fixture.sha256 {
        report
            .structural_differences
            .push("decoded fixture SHA-256 differs".into());
    }
    if expected.manifest.records.len() != actual.manifest.records.len() {
        report.structural_differences.push(format!(
            "record count differs: expected {}, actual {}",
            expected.manifest.records.len(),
            actual.manifest.records.len()
        ));
    }
    for (index, (expected_record, actual_record)) in expected
        .manifest
        .records
        .iter()
        .zip(&actual.manifest.records)
        .enumerate()
    {
        if !same_record_identity(expected_record, actual_record) {
            report
                .structural_differences
                .push(format!("record {index} metadata differs"));
            if tolerances.exact_metadata {
                break;
            }
        }
        let expected_values = expected.values(expected_record)?;
        let actual_values = actual.values(actual_record)?;
        if expected_values.len() != actual_values.len() {
            report.structural_differences.push(format!(
                "record {index} value count differs: expected {}, actual {}",
                expected_values.len(),
                actual_values.len()
            ));
            break;
        }
        let tolerance = tolerance_for(
            tolerances,
            &expected.manifest.case.precision,
            expected_record,
        )
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "no tolerance for precision `{}` and component `{}`",
                    expected.manifest.case.precision, expected_record.component
                ),
            )
        })?;
        report.records_compared += 1;
        for (value_index, (&expected_value, &actual_value)) in
            expected_values.iter().zip(&actual_values).enumerate()
        {
            report.values_compared += 1;
            let absolute_error = f64::from((actual_value - expected_value).abs());
            let scale = f64::from(expected_value.abs().max(actual_value.abs()));
            let relative_error = if scale == 0.0 {
                absolute_error
            } else {
                absolute_error / scale
            };
            report.maximum_absolute_error = report.maximum_absolute_error.max(absolute_error);
            report.maximum_relative_error = report.maximum_relative_error.max(relative_error);
            let allowed_error = tolerance.absolute + tolerance.relative * scale;
            if (!expected_value.is_finite()
                || !actual_value.is_finite()
                || absolute_error > allowed_error)
                && report.first_difference.is_none()
            {
                report.first_difference = Some(Difference {
                    record: index,
                    layer: expected_record.layer.clone(),
                    component: expected_record.component.clone(),
                    index: value_index,
                    expected: expected_value,
                    actual: actual_value,
                    absolute_error,
                    allowed_error,
                });
            }
        }
        if report.first_difference.is_some() {
            break;
        }
    }
    report.compatible =
        report.structural_differences.is_empty() && report.first_difference.is_none();
    Ok(report)
}

pub fn write_comparison_report(path: &Path, report: &ComparisonReport) -> io::Result<()> {
    write_json(path, report)
}

pub fn load_tolerance_profile(path: &Path) -> io::Result<ToleranceProfile> {
    let profile: ToleranceProfile = read_json(path)?;
    if profile.schema_version != ARTIFACT_SCHEMA_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported tolerance profile schema",
        ));
    }
    Ok(profile)
}

pub fn prepare_wav_fixture(
    wav: &Path,
    output: &Path,
    name: &str,
    license: &str,
    expected_source_sha256: Option<&str>,
) -> io::Result<FixtureManifest> {
    let source_bytes = fs::read(wav)?;
    let source_hash = encode_digest(Sha256::digest(&source_bytes));
    if expected_source_sha256.is_some_and(|expected| !expected.eq_ignore_ascii_case(&source_hash)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("source SHA-256 is {source_hash}"),
        ));
    }
    let samples = decode_pcm16_mono_wav(&source_bytes)?;
    prepare_samples_fixture(
        output,
        name,
        license,
        &samples,
        Some(FileProvenance {
            path: wav.display().to_string(),
            sha256: source_hash,
            bytes: Some(source_bytes.len() as u64),
            license: Some(license.into()),
            revision: None,
        }),
    )
}

pub fn prepare_generated_fixture(
    output: &Path,
    name: &str,
    seconds: usize,
    synthetic: bool,
) -> io::Result<FixtureManifest> {
    let sample_count = usize::try_from(SAMPLE_RATE)
        .ok()
        .and_then(|rate| rate.checked_mul(seconds))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "fixture is too long"))?;
    let samples = if synthetic {
        (0..sample_count)
            .map(|index| {
                let t = index as f32 / SAMPLE_RATE as f32;
                let envelope = (1.0 - t / seconds.max(1) as f32).max(0.0);
                envelope
                    * (0.20 * (std::f32::consts::TAU * 220.0 * t).sin()
                        + 0.05 * (std::f32::consts::TAU * 997.0 * t).sin())
            })
            .collect::<Vec<_>>()
    } else {
        vec![0.0; sample_count]
    };
    prepare_samples_fixture(output, name, "CC0-1.0", &samples, None)
}

pub fn load_fixture(root: &Path) -> io::Result<(FixtureManifest, Vec<f32>)> {
    let manifest: FixtureManifest = read_json(&root.join("fixture.json"))?;
    if manifest.schema_version != FIXTURE_SCHEMA_VERSION
        || manifest.sample_rate != SAMPLE_RATE
        || manifest.channels != 1
        || manifest.encoding != "f32le"
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported fixture contract",
        ));
    }
    let bytes = fs::read(root.join("samples.f32le"))?;
    if encode_digest(Sha256::digest(&bytes)) != manifest.samples_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "decoded sample SHA-256 differs from fixture manifest",
        ));
    }
    let samples = decode_f32le(&bytes)?;
    if samples.len() != manifest.sample_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "decoded sample count differs from fixture manifest",
        ));
    }
    Ok((manifest, samples))
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(encode_digest(digest.finalize()))
}

fn prepare_samples_fixture(
    output: &Path,
    name: &str,
    license: &str,
    samples: &[f32],
    source: Option<FileProvenance>,
) -> io::Result<FixtureManifest> {
    if samples.iter().any(|value| !value.is_finite()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fixture samples must be finite",
        ));
    }
    fs::create_dir_all(output)?;
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(samples));
    for sample in samples {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    fs::write(output.join("samples.f32le"), &bytes)?;
    let manifest = FixtureManifest {
        schema_version: FIXTURE_SCHEMA_VERSION,
        name: name.into(),
        sample_rate: SAMPLE_RATE,
        channels: 1,
        sample_count: samples.len(),
        encoding: "f32le".into(),
        samples_sha256: encode_digest(Sha256::digest(&bytes)),
        source,
        license: license.into(),
    };
    write_json(&output.join("fixture.json"), &manifest)?;
    Ok(manifest)
}

fn decode_pcm16_mono_wav(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a RIFF/WAVE file",
        ));
    }
    let mut format = None;
    let mut data = None;
    let mut cursor = 12_usize;
    while cursor.checked_add(8).is_some_and(|end| end <= bytes.len()) {
        let id = &bytes[cursor..cursor + 4];
        let length = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let start = cursor + 8;
        let end = start.checked_add(length).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "WAV chunk length overflows")
        })?;
        if end > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "truncated WAV chunk",
            ));
        }
        match id {
            b"fmt " => format = Some(&bytes[start..end]),
            b"data" => data = Some(&bytes[start..end]),
            _ => {}
        }
        cursor = end + (length & 1);
    }
    let format =
        format.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing fmt chunk"))?;
    let data =
        data.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing data chunk"))?;
    if format.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "short fmt chunk",
        ));
    }
    let encoding = u16::from_le_bytes(format[0..2].try_into().unwrap());
    let channels = u16::from_le_bytes(format[2..4].try_into().unwrap());
    let sample_rate = u32::from_le_bytes(format[4..8].try_into().unwrap());
    let bits = u16::from_le_bytes(format[14..16].try_into().unwrap());
    if encoding != 1 || channels != 1 || sample_rate != SAMPLE_RATE || bits != 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "fixture WAV must be mono PCM16 at 16000 Hz",
        ));
    }
    if data.len() % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "odd PCM16 data length",
        ));
    }
    Ok(data
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32768.0)
        .collect())
}

struct ArtifactData {
    manifest: ArtifactManifest,
    bytes: Vec<u8>,
}

impl ArtifactData {
    fn values(&self, record: &Record) -> io::Result<Vec<f32>> {
        let start = usize::try_from(record.offset_bytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "record offset is too large")
        })?;
        let length = usize::try_from(record.byte_length).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "record length is too large")
        })?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "record range overflows"))?;
        let bytes = self.bytes.get(start..end).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "record range is outside values.f32le",
            )
        })?;
        if encode_digest(Sha256::digest(bytes)) != record.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("record {} SHA-256 differs", record.sequence),
            ));
        }
        decode_f32le(bytes)
    }
}

fn read_artifact(root: &Path) -> io::Result<ArtifactData> {
    let manifest: ArtifactManifest = read_json(&root.join("artifact.json"))?;
    if manifest.schema_version != ARTIFACT_SCHEMA_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unsupported artifact schema",
        ));
    }
    let bytes = fs::read(root.join("values.f32le"))?;
    if encode_digest(Sha256::digest(&bytes)) != manifest.data_sha256 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "values.f32le SHA-256 differs from artifact manifest",
        ));
    }
    Ok(ArtifactData { manifest, bytes })
}

fn decode_f32le(bytes: &[u8]) -> io::Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(size_of::<f32>()) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "f32le byte length is invalid",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn same_record_identity(expected: &Record, actual: &Record) -> bool {
    expected.layer == actual.layer
        && expected.component == actual.component
        && expected.track == actual.track
        && expected.frame == actual.frame
        && expected.inference == actual.inference
        && expected.timestamp == actual.timestamp
        && expected.next_timestamp == actual.next_timestamp
        && expected.dtype == actual.dtype
        && expected.shape == actual.shape
}

fn tolerance_for(
    profile: &ToleranceProfile,
    precision: &str,
    record: &Record,
) -> Option<Tolerance> {
    profile
        .components
        .get(&record.component)
        .and_then(|values| values.get(precision))
        .or_else(|| profile.default.get(precision))
        .copied()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> io::Result<T> {
    let bytes = fs::read(path)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))
}

fn write_json(path: &Path, value: &impl Serialize) -> io::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    bytes.push(b'\n');
    fs::write(path, bytes)
}

fn encode_digest(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02X}").expect("writing to a string cannot fail");
    }
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temporary(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "audio2face3d-reference-{name}-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn case() -> Case {
        Case {
            name: "test".into(),
            pipeline: "regression".into(),
            execution: "standard".into(),
            precision: "fp32".into(),
            seed: 0,
            track_count: 1,
        }
    }

    fn producer(name: &str) -> Producer {
        Producer {
            implementation: name.into(),
            version: "test".into(),
            revision: None,
        }
    }

    #[test]
    fn generated_fixture_round_trips_and_is_deterministic() {
        let first = temporary("fixture-a");
        let second = temporary("fixture-b");
        let a = prepare_generated_fixture(&first, "synthetic", 1, true).unwrap();
        let b = prepare_generated_fixture(&second, "synthetic", 1, true).unwrap();
        assert_eq!(a.samples_sha256, b.samples_sha256);
        let (loaded, samples) = load_fixture(&first).unwrap();
        assert_eq!(loaded, a);
        assert_eq!(samples.len(), SAMPLE_RATE as usize);
        fs::remove_dir_all(first).unwrap();
        fs::remove_dir_all(second).unwrap();
    }

    #[test]
    fn comparator_reports_first_layer_and_value_index() {
        let expected = temporary("expected");
        let actual = temporary("actual");
        let fixture = FileProvenance {
            path: "samples.f32le".into(),
            sha256: "fixture".into(),
            bytes: Some(8),
            license: Some("CC0-1.0".into()),
            revision: None,
        };
        for (root, values) in [(&expected, [0.0, 1.0]), (&actual, [0.0, 1.1])] {
            let mut writer = ArtifactWriter::create(
                root,
                producer(if root == &expected { "cpp" } else { "rust" }),
                case(),
                fixture.clone(),
            )
            .unwrap();
            writer
                .push_f32(
                    RecordMetadata {
                        layer: "postprocess".into(),
                        component: "skin".into(),
                        track: 0,
                        frame: Some(0),
                        inference: None,
                        timestamp: Some(0),
                        next_timestamp: Some(533),
                        shape: vec![2],
                    },
                    &values,
                )
                .unwrap();
            writer.finish().unwrap();
        }
        let profile = ToleranceProfile {
            schema_version: 1,
            exact_metadata: true,
            default: BTreeMap::from([(
                "fp32".into(),
                Tolerance {
                    absolute: 1.0e-5,
                    relative: 1.0e-5,
                },
            )]),
            components: BTreeMap::new(),
        };
        let report = compare_artifacts(&expected, &actual, &profile).unwrap();
        assert!(!report.compatible);
        let first = report.first_difference.unwrap();
        assert_eq!(first.layer, "postprocess");
        assert_eq!(first.component, "skin");
        assert_eq!(first.index, 1);
        fs::remove_dir_all(expected).unwrap();
        fs::remove_dir_all(actual).unwrap();
    }

    #[test]
    fn pcm16_decoder_uses_the_fixed_negative_full_scale_mapping() {
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&40_u32.to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16_u32.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&1_u16.to_le_bytes());
        wav.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
        wav.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
        wav.extend_from_slice(&2_u16.to_le_bytes());
        wav.extend_from_slice(&16_u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&4_u32.to_le_bytes());
        wav.extend_from_slice(&i16::MIN.to_le_bytes());
        wav.extend_from_slice(&i16::MAX.to_le_bytes());
        assert_eq!(
            decode_pcm16_mono_wav(&wav).unwrap(),
            [-1.0, 32767.0 / 32768.0]
        );
    }

    #[test]
    fn malformed_wav_inputs_return_errors_without_panicking() {
        let malformed = [
            Vec::new(),
            b"RIFF\0\0\0\0WAVE".to_vec(),
            b"RIFF\0\0\0\0WAVEfmt \x10\0\0".to_vec(),
            b"RIFF\0\0\0\0WAVEdata\xFF\xFF\xFF\xFF".to_vec(),
        ];
        for bytes in malformed {
            let result = std::panic::catch_unwind(|| decode_pcm16_mono_wav(&bytes));
            assert!(result.is_ok(), "malformed WAV caused a panic");
            assert!(result.unwrap().is_err());
        }
    }
}
