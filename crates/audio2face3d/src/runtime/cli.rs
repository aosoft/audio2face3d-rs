//! Shared command-line parsing for the two executables. Context never reads these implicitly.
use super::{NativeRuntimeConfig, NativeRuntimeError, NativeRuntimeErrorKind, NativeSearchPolicy};
use std::path::{Path, PathBuf};
#[derive(Clone, Debug, Default, clap::Args)]
pub struct NativeRuntimeArgs {
    #[arg(long, global = true)]
    cuda_root: Option<PathBuf>,
    #[arg(long, global = true)]
    tensorrt_root: Option<PathBuf>,
    #[arg(long, global = true, conflicts_with = "cuda_root")]
    cuda_library_dir: Vec<PathBuf>,
    #[arg(long, global = true, conflicts_with = "tensorrt_root")]
    tensorrt_library_dir: Vec<PathBuf>,
    #[arg(long, global = true)]
    runtime_config: Option<PathBuf>,
    #[arg(long, global = true, value_parser = ["explicit", "discover"])]
    runtime_search: Option<String>,
}
fn error(message: impl Into<String>) -> NativeRuntimeError {
    NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidConfig, message)
}
fn absolute(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_owned()
    } else {
        base.join(path)
    }
}
fn parse(text: &str, base: &Path) -> Result<NativeRuntimeConfig, NativeRuntimeError> {
    let table = text
        .parse::<toml::Table>()
        .map_err(|e| error(e.to_string()))?;
    for key in table.keys() {
        if ![
            "schema_version",
            "search_policy",
            "cuda_root",
            "tensorrt_root",
            "cuda_library_dirs",
            "tensorrt_library_dirs",
        ]
        .contains(&key.as_str())
        {
            return Err(error(format!("unknown runtime setting: {key}")));
        }
    }
    if table
        .get("schema_version")
        .and_then(toml::Value::as_integer)
        != Some(1)
    {
        return Err(error("runtime schema_version must be 1"));
    }
    let mut builder = NativeRuntimeConfig::builder();
    let string = |key: &str| -> Result<Option<&str>, NativeRuntimeError> {
        table
            .get(key)
            .map(|value| {
                value
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| error(format!("{key} must be a nonempty string")))
            })
            .transpose()
    };
    if let Some(value) = string("cuda_root")? {
        builder = builder.cuda_root(absolute(base, Path::new(value)));
    }
    if let Some(value) = string("tensorrt_root")? {
        builder = builder.tensorrt_root(absolute(base, Path::new(value)));
    }
    for key in ["cuda_library_dirs", "tensorrt_library_dirs"] {
        if let Some(value) = table.get(key) {
            let dirs = value
                .as_array()
                .ok_or_else(|| error(format!("{key} must be an array")))?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .map(|s| absolute(base, Path::new(s)))
                        .ok_or_else(|| error(format!("{key} must contain nonempty paths")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            builder = if key == "cuda_library_dirs" {
                builder.cuda_library_dirs(dirs)
            } else {
                builder.tensorrt_library_dirs(dirs)
            };
        }
    }
    if let Some(policy) = string("search_policy")? {
        builder = builder.search_policy(policy_value(policy)?);
    }
    builder.build()
}
fn policy_value(value: &str) -> Result<NativeSearchPolicy, NativeRuntimeError> {
    match value {
        "explicit" => Ok(NativeSearchPolicy::ExplicitOnly),
        "discover" => Ok(NativeSearchPolicy::Discover),
        _ => Err(error("search_policy must be explicit or discover")),
    }
}
impl NativeRuntimeArgs {
    pub fn resolve(&self) -> Result<NativeRuntimeConfig, NativeRuntimeError> {
        let cwd = std::env::current_dir().map_err(|e| error(e.to_string()))?;
        let base = if let Some(file) = &self.runtime_config {
            let file = absolute(&cwd, file);
            let text = std::fs::read_to_string(&file)
                .map_err(|e| error(e.to_string()).with_path(&file))?;
            parse(&text, file.parent().unwrap())?
        } else {
            NativeRuntimeConfig::default()
        };
        self.overlay(base, &cwd)
    }
    fn overlay(
        &self,
        base: NativeRuntimeConfig,
        cwd: &Path,
    ) -> Result<NativeRuntimeConfig, NativeRuntimeError> {
        if self.cuda_root.is_some() && !self.cuda_library_dir.is_empty()
            || self.tensorrt_root.is_some() && !self.tensorrt_library_dir.is_empty()
        {
            return Err(error("root and library directories are mutually exclusive"));
        }
        let mut builder =
            NativeRuntimeConfig::builder().search_policy(match &self.runtime_search {
                Some(value) => policy_value(value)?,
                None => base.search_policy(),
            });
        for (cuda, root, dirs, inherited_root, inherited_dirs) in [
            (
                true,
                &self.cuda_root,
                &self.cuda_library_dir,
                base.cuda_root(),
                base.cuda_library_dirs(),
            ),
            (
                false,
                &self.tensorrt_root,
                &self.tensorrt_library_dir,
                base.tensorrt_root(),
                base.tensorrt_library_dirs(),
            ),
        ] {
            let overrides = root.is_some() || !dirs.is_empty();
            let selected_root = if overrides {
                root.as_deref()
            } else {
                inherited_root
            };
            let selected_dirs = if overrides {
                dirs.as_slice()
            } else {
                inherited_dirs
            };
            if let Some(root) = selected_root {
                builder = if cuda {
                    builder.cuda_root(absolute(cwd, root))
                } else {
                    builder.tensorrt_root(absolute(cwd, root))
                };
            }
            if !selected_dirs.is_empty() {
                let dirs = selected_dirs
                    .iter()
                    .map(|p| absolute(cwd, p))
                    .collect::<Vec<_>>();
                builder = if cuda {
                    builder.cuda_library_dirs(dirs)
                } else {
                    builder.tensorrt_library_dirs(dirs)
                };
            }
        }
        builder.build()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn file_paths_and_cli_overrides_use_their_own_bases_and_replace_groups() {
        let cwd = std::env::current_dir().unwrap();
        let file_base = cwd.join("temp/config");
        let base=parse("schema_version=1\nsearch_policy='explicit'\ncuda_root='cuda'\ntensorrt_library_dirs=['trt']", &file_base).unwrap();
        assert_eq!(base.cuda_root(), Some(file_base.join("cuda").as_path()));
        let args = NativeRuntimeArgs {
            cuda_library_dir: vec!["custom".into()],
            tensorrt_root: Some("sdk".into()),
            ..Default::default()
        };
        let result = args.overlay(base, &cwd).unwrap();
        assert!(result.cuda_root().is_none());
        assert_eq!(result.cuda_library_dirs(), &[cwd.join("custom")]);
        assert_eq!(result.tensorrt_root(), Some(cwd.join("sdk").as_path()));
        assert!(result.tensorrt_library_dirs().is_empty());
        assert_eq!(result.search_policy(), NativeSearchPolicy::ExplicitOnly);
    }
    #[test]
    fn malformed_files_are_not_supplemented_from_other_configuration() {
        let cwd = std::env::current_dir().unwrap();
        for text in [
            "schema_version=2",
            "schema_version=1\nunknown=true",
            "schema_version=1\ncuda_root=''",
            "schema_version=1\ncuda_library_dirs=[]",
            "schema_version=1\ncuda_root='a'\ncuda_library_dirs=['b']",
            "schema_version=1\nsearch_policy='latest'",
        ] {
            assert!(parse(text, &cwd).is_err(), "{text}");
        }
    }
}
