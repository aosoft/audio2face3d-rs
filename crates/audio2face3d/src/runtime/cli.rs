//! Shared command-line parsing for the two executables. Context never reads these implicitly.
use super::{NativeRuntimeConfig, NativeRuntimeError, NativeRuntimeErrorKind, NativeSearchPolicy};
use std::path::{Path, PathBuf};
#[derive(Clone, Debug, Default, clap::Args)]
pub struct PlatformArgs {
    #[arg(long, global = true)]
    cuda_root: Option<PathBuf>,
    #[arg(long, global = true)]
    tensorrt_root: Option<PathBuf>,
    #[arg(long, global = true, conflicts_with = "cuda_root")]
    cuda_library_dir: Vec<PathBuf>,
    #[arg(long, global = true, conflicts_with = "tensorrt_root")]
    tensorrt_library_dir: Vec<PathBuf>,
    /// Platform configuration file shared by build and runtime.
    #[arg(long, global = true)]
    platform_config: Option<PathBuf>,
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
    let table = crate::platform_config_file::parse(text, "runtime").map_err(error)?;
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
    if let Some(value) = string("cuda-root")? {
        builder = builder.cuda_root(absolute(base, Path::new(value)));
    }
    if let Some(value) = string("tensorrt-root")? {
        builder = builder.tensorrt_root(absolute(base, Path::new(value)));
    }
    for key in ["cuda-library-dirs", "tensorrt-library-dirs"] {
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
            builder = if key == "cuda-library-dirs" {
                builder.cuda_library_dirs(dirs)
            } else {
                builder.tensorrt_library_dirs(dirs)
            };
        }
    }
    if let Some(policy) = string("search-policy")? {
        builder = builder.search_policy(policy_value(policy)?);
    }
    builder.build()
}
fn policy_value(value: &str) -> Result<NativeSearchPolicy, NativeRuntimeError> {
    match value {
        "explicit" => Ok(NativeSearchPolicy::ExplicitOnly),
        "discover" => Ok(NativeSearchPolicy::Discover),
        _ => Err(error("search-policy must be explicit or discover")),
    }
}
fn select_file(cwd: &Path, explicit: Option<PathBuf>, user: Option<PathBuf>) -> Option<PathBuf> {
    explicit.map(|path| absolute(cwd, &path)).or_else(|| {
        std::iter::once(cwd.join("platform.toml"))
            .chain(user)
            .find(|path| path.exists())
    })
}

impl PlatformArgs {
    pub fn resolve(&self) -> Result<NativeRuntimeConfig, NativeRuntimeError> {
        let cwd = std::env::current_dir().map_err(|e| error(e.to_string()))?;
        let selected = select_file(
            &cwd,
            self.platform_config
                .clone()
                .or_else(|| std::env::var_os("AUDIO2FACE3D_PLATFORM_CONFIG").map(PathBuf::from)),
            crate::platform_config_file::user_config(),
        );
        let base = if let Some(file) = &selected {
            let file = absolute(&cwd, file);
            let text = std::fs::read_to_string(&file)
                .map_err(|e| error(e.to_string()).with_path(&file))?;
            parse(&text, file.parent().unwrap()).map_err(|e| e.with_path(&file))?
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
        let base = parse(
            "cuda-root='cuda'\n[runtime]\nsearch-policy='explicit'\ntensorrt-library-dirs=['trt']",
            &file_base,
        )
        .unwrap();
        assert_eq!(base.cuda_root(), Some(file_base.join("cuda").as_path()));
        let args = PlatformArgs {
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
            "schema_version=1",
            "schema-version=1",
            "cuda_root='cuda'",
            "unknown=true",
            "cuda-root=''",
            "[runtime]\ncuda-library-dirs=[]",
            "[runtime]\ncuda-root='a'\ncuda-library-dirs=['b']",
            "[runtime]\nsearch-policy='latest'",
            "[build]\ncuda-archs=false",
        ] {
            assert!(parse(text, &cwd).is_err(), "{text}");
        }
    }
    #[test]
    fn shared_file_separates_build_settings_and_runtime_overrides() {
        let base = std::env::current_dir().unwrap().join("temp/config");
        let config = parse(
            include_str!("../../tests/fixtures/platform-config.toml"),
            &base,
        )
        .unwrap();
        assert!(config.cuda_root().is_none());
        assert_eq!(config.cuda_library_dirs(), &[base.join("deploy/cuda")]);
        assert_eq!(
            config.tensorrt_root(),
            Some(base.join("deploy/trt").as_path())
        );
        assert_eq!(config.search_policy(), NativeSearchPolicy::ExplicitOnly);
        let common = parse(
            "cuda-root='sdk'\n[build]\ncuda-host-compiler='missing/compiler'",
            &base,
        )
        .unwrap();
        assert_eq!(common.cuda_root(), Some(base.join("sdk").as_path()));
    }
    #[test]
    fn file_selection_prefers_explicit_then_local_then_user_without_merging() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../temp/platform-config-work/tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir_all(&root).unwrap();
        let user = root.join("user.toml");
        let local = root.join("platform.toml");
        assert_eq!(select_file(&root, None, Some(user.clone())), None);
        std::fs::write(&user, "").unwrap();
        assert_eq!(
            select_file(&root, None, Some(user.clone())),
            Some(user.clone())
        );
        std::fs::write(&local, "invalid").unwrap();
        assert_eq!(select_file(&root, None, Some(user.clone())), Some(local));
        assert_eq!(
            select_file(&root, Some("missing.toml".into()), Some(user)),
            Some(root.join("missing.toml"))
        );
    }
    #[test]
    fn both_cli_and_file_option_names_are_kebab_case() {
        use clap::Parser;
        #[derive(Parser)]
        struct Args {
            #[command(flatten)]
            platform: PlatformArgs,
        }
        let args = Args::try_parse_from([
            "test",
            "--platform-config",
            "config.toml",
            "--cuda-root",
            "cuda",
            "--runtime-search",
            "explicit",
        ])
        .unwrap();
        assert_eq!(
            args.platform.platform_config.as_deref(),
            Some(Path::new("config.toml"))
        );
        assert!(Args::try_parse_from(["test", "--runtime-config", "config.toml"]).is_err());
        assert!(
            Args::try_parse_from(["test", "--cuda-root", "cuda", "--cuda-library-dir", "lib"])
                .is_err()
        );
    }
}
