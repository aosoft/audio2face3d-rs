//! Parsing and expansion of the SDK's `trt_info.json` build metadata.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// TensorRT build metadata distributed with an Audio2X ONNX model.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrtBuildInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_trt_builder_time: Option<u64>,
    pub trt_build_param: BTreeMap<String, Vec<String>>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub defaults: BTreeMap<String, u64>,
}

impl TrtBuildInfo {
    /// Loads and strictly parses one SDK `trt_info.json` document.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TrtBuildInfoError> {
        let path = path.as_ref();
        let payload = fs::read(path).map_err(|source| TrtBuildInfoError::Read {
            path: path.to_owned(),
            source,
        })?;
        serde_json::from_slice(&payload).map_err(|source| TrtBuildInfoError::Parse {
            path: path.to_owned(),
            source,
        })
    }

    /// Expands `{NAME}` placeholders using `defaults` and flattens the named
    /// argument groups in deterministic key order.
    pub fn arguments(&self) -> Result<Vec<String>, TrtBuildInfoError> {
        let mut arguments = Vec::new();
        for values in self.trt_build_param.values() {
            for value in values {
                arguments.push(expand_argument(value, &self.defaults)?);
            }
        }
        Ok(arguments)
    }

    /// Adds a named argument group while rejecting an accidental overwrite.
    pub fn insert_group(
        &mut self,
        name: impl Into<String>,
        arguments: Vec<String>,
    ) -> Result<(), TrtBuildInfoError> {
        let name = name.into();
        if name.is_empty() || arguments.is_empty() || arguments.iter().any(String::is_empty) {
            return Err(TrtBuildInfoError::InvalidGroup(name));
        }
        if self.trt_build_param.contains_key(&name) {
            return Err(TrtBuildInfoError::DuplicateGroup(name));
        }
        self.trt_build_param.insert(name, arguments);
        Ok(())
    }
}

fn expand_argument(
    argument: &str,
    defaults: &BTreeMap<String, u64>,
) -> Result<String, TrtBuildInfoError> {
    let mut expanded = String::with_capacity(argument.len());
    let mut remaining = argument;
    while let Some(open) = remaining.find('{') {
        expanded.push_str(&remaining[..open]);
        let placeholder = &remaining[open + 1..];
        let close = placeholder
            .find('}')
            .ok_or_else(|| TrtBuildInfoError::InvalidPlaceholder(argument.to_owned()))?;
        let name = &placeholder[..close];
        if name.is_empty() || name.contains('{') {
            return Err(TrtBuildInfoError::InvalidPlaceholder(argument.to_owned()));
        }
        let value = defaults
            .get(name)
            .ok_or_else(|| TrtBuildInfoError::MissingDefault(name.to_owned()))?;
        expanded.push_str(&value.to_string());
        remaining = &placeholder[close + 1..];
    }
    if remaining.contains('}') {
        return Err(TrtBuildInfoError::InvalidPlaceholder(argument.to_owned()));
    }
    expanded.push_str(remaining);
    Ok(expanded)
}

#[derive(Debug)]
pub enum TrtBuildInfoError {
    Read {
        path: PathBuf,
        source: io::Error,
    },
    Parse {
        path: PathBuf,
        source: serde_json::Error,
    },
    MissingDefault(String),
    InvalidPlaceholder(String),
    InvalidGroup(String),
    DuplicateGroup(String),
}

impl fmt::Display for TrtBuildInfoError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => {
                write!(formatter, "cannot read {}: {source}", path.display())
            }
            Self::Parse { path, source } => {
                write!(formatter, "cannot parse {}: {source}", path.display())
            }
            Self::MissingDefault(name) => {
                write!(formatter, "trt_info.json has no default for {{{name}}}")
            }
            Self::InvalidPlaceholder(argument) => {
                write!(
                    formatter,
                    "invalid trt_info.json placeholder in `{argument}`"
                )
            }
            Self::InvalidGroup(name) => {
                write!(formatter, "invalid TensorRT build argument group `{name}`")
            }
            Self::DuplicateGroup(name) => {
                write!(
                    formatter,
                    "TensorRT build argument group `{name}` already exists"
                )
            }
        }
    }
}

impl std::error::Error for TrtBuildInfoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Parse { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_defaults_in_deterministic_group_order() {
        let info: TrtBuildInfo = serde_json::from_str(
            r#"{
                "trt_build_param": {
                    "cuda_in_graphics": ["--memPoolSize=tacticSharedMem:0.046875"],
                    "batch": ["--minShapes=input:1x2", "--maxShapes=input:{MAX_BATCH}x2"]
                },
                "defaults": {"MAX_BATCH": 128}
            }"#,
        )
        .unwrap();
        assert_eq!(
            info.arguments().unwrap(),
            [
                "--minShapes=input:1x2",
                "--maxShapes=input:128x2",
                "--memPoolSize=tacticSharedMem:0.046875"
            ]
        );
    }

    #[test]
    fn rejects_missing_and_malformed_placeholders() {
        let defaults = BTreeMap::new();
        assert!(matches!(
            expand_argument("--maxShapes=input:{MAX}x2", &defaults),
            Err(TrtBuildInfoError::MissingDefault(name)) if name == "MAX"
        ));
        assert!(matches!(
            expand_argument("--maxShapes=input:{MAX", &defaults),
            Err(TrtBuildInfoError::InvalidPlaceholder(_))
        ));
    }

    #[test]
    fn precision_group_does_not_overwrite_source_metadata() {
        let mut info = TrtBuildInfo {
            estimated_trt_builder_time: Some(150),
            trt_build_param: BTreeMap::new(),
            defaults: BTreeMap::new(),
        };
        info.insert_group("fp16", vec!["--fp16".into()]).unwrap();
        assert!(matches!(
            info.insert_group("fp16", vec!["--fp16".into()]),
            Err(TrtBuildInfoError::DuplicateGroup(name)) if name == "fp16"
        ));
    }
}
