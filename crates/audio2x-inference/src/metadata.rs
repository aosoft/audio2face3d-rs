//! Machine-readable TensorRT engine provenance and compatibility checks.

use audio2x_core::{BindingSchema, Dimension, ElementType, IoMode};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct IoTensorMetadata {
    pub name: String,
    pub mode: String,
    pub dtype: String,
    pub shape: Vec<DimensionMetadata>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind")]
pub enum DimensionMetadata {
    Fixed { value: usize },
    Batch,
    Dynamic { min: usize, max: usize },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct ProfileMetadata {
    pub name: String,
    pub min: Vec<u64>,
    pub opt: Vec<u64>,
    pub max: Vec<u64>,
}

/// Provenance and ABI contract for a serialized TensorRT engine.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct EngineMetadata {
    pub schema_version: u32,
    pub onnx_sha256: String,
    pub engine_sha256: String,
    pub tensorrt_version: String,
    pub cuda_version: String,
    pub driver_version: String,
    pub gpu_name: String,
    pub compute_capability: String,
    pub precision: String,
    pub version_compatible: bool,
    pub hardware_compatibility_level: Option<String>,
    pub trtexec_args: Vec<String>,
    pub io: Vec<IoTensorMetadata>,
    pub profiles: Vec<ProfileMetadata>,
}

impl EngineMetadata {
    pub const CURRENT_SCHEMA_VERSION: u32 = 1;

    pub fn from_schema(
        schema: &BindingSchema,
        onnx_sha256: impl Into<String>,
        engine_sha256: impl Into<String>,
    ) -> Self {
        let io = schema
            .bindings()
            .iter()
            .map(|b| IoTensorMetadata {
                name: b.name.clone(),
                mode: match b.mode {
                    IoMode::Input => "input",
                    IoMode::Output => "output",
                }
                .into(),
                dtype: match b.element_type {
                    ElementType::F32 => "f32",
                    ElementType::F16 => "f16",
                    ElementType::I64 => "i64",
                    ElementType::U64 => "u64",
                    ElementType::Bool => "bool",
                    ElementType::Raw => "raw",
                }
                .into(),
                shape: b
                    .shape
                    .dimensions()
                    .iter()
                    .map(|d| match *d {
                        Dimension::Fixed(value) => DimensionMetadata::Fixed { value },
                        Dimension::Batch => DimensionMetadata::Batch,
                        Dimension::Dynamic { min, max } => DimensionMetadata::Dynamic { min, max },
                    })
                    .collect(),
            })
            .collect();
        Self {
            schema_version: Self::CURRENT_SCHEMA_VERSION,
            onnx_sha256: onnx_sha256.into(),
            engine_sha256: engine_sha256.into(),
            tensorrt_version: String::new(),
            cuda_version: String::new(),
            driver_version: String::new(),
            gpu_name: String::new(),
            compute_capability: String::new(),
            precision: "fp32".into(),
            version_compatible: false,
            hardware_compatibility_level: None,
            trtexec_args: Vec::new(),
            io,
            profiles: Vec::new(),
        }
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn compatibility(&self, actual: &Self) -> Compatibility {
        let mut issues = Vec::new();
        if self.schema_version != actual.schema_version {
            issues.push(CompatibilityIssue::SchemaVersion);
        }
        if self.onnx_sha256 != actual.onnx_sha256 {
            issues.push(CompatibilityIssue::OnnxHash);
        }
        if self.io != actual.io {
            issues.push(CompatibilityIssue::IoSchema);
        }
        if self.profiles != actual.profiles {
            issues.push(CompatibilityIssue::Profiles);
        }
        if major(&self.tensorrt_version) != major(&actual.tensorrt_version) {
            issues.push(CompatibilityIssue::TensorRtMajor);
        }
        if major(&self.cuda_version) != major(&actual.cuda_version) {
            issues.push(CompatibilityIssue::CudaMajor);
        }
        if self.compute_capability != actual.compute_capability {
            issues.push(CompatibilityIssue::ComputeCapability);
        }
        if self.precision != actual.precision {
            issues.push(CompatibilityIssue::Precision);
        }
        if self.version_compatible != actual.version_compatible {
            issues.push(CompatibilityIssue::VersionCompatibility);
        }
        Compatibility {
            compatible: issues.is_empty(),
            issues,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Compatibility {
    pub compatible: bool,
    pub issues: Vec<CompatibilityIssue>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum CompatibilityIssue {
    SchemaVersion,
    OnnxHash,
    IoSchema,
    Profiles,
    TensorRtMajor,
    CudaMajor,
    ComputeCapability,
    Precision,
    VersionCompatibility,
}

fn major(version: &str) -> &str {
    version.split('.').next().unwrap_or(version)
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio2x_core::{Binding, Shape};

    #[test]
    fn metadata_round_trips_and_maps_io() {
        let schema = BindingSchema::new(vec![Binding {
            name: "input_values".into(),
            mode: IoMode::Input,
            element_type: ElementType::F32,
            shape: Shape::new(vec![
                Dimension::Batch,
                Dimension::Dynamic { min: 2, max: 8 },
            ])
            .unwrap(),
        }])
        .unwrap();
        let metadata = EngineMetadata::from_schema(&schema, "onnx", "engine");
        let decoded = EngineMetadata::from_json(&metadata.to_json().unwrap()).unwrap();
        assert_eq!(metadata, decoded);
        assert!(metadata.to_json().unwrap().contains("input_values"));
    }

    #[test]
    fn compatibility_rejects_hash_schema_and_runtime_major_mismatch() {
        let schema = BindingSchema::new(Vec::<Binding>::new()).unwrap();
        let mut expected = EngineMetadata::from_schema(&schema, "a", "e");
        let mut actual = expected.clone();
        actual.onnx_sha256 = "b".into();
        actual.tensorrt_version = "11.0".into();
        actual.cuda_version = "13.0".into();
        actual.compute_capability = "9.0".into();
        let result = expected.compatibility(&actual);
        assert!(!result.compatible);
        assert!(result.issues.contains(&CompatibilityIssue::OnnxHash));
        assert!(result.issues.contains(&CompatibilityIssue::TensorRtMajor));
        assert!(result.issues.contains(&CompatibilityIssue::CudaMajor));
        assert!(
            result
                .issues
                .contains(&CompatibilityIssue::ComputeCapability)
        );
        expected.engine_sha256 = "different-is-allowed-for-regeneration".into();
        assert!(expected.compatibility(&expected).compatible);
    }
}
