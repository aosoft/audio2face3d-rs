//! Shared platform.toml syntax used by build.rs and the CLI. No SDK access.
use std::{env, path::PathBuf};
use toml::{Table, Value};

/// Validate the whole document, then return the selected section with common roots inherited.
pub fn parse(text: &str, phase: &str) -> Result<Table, String> {
    let mut common = text.parse::<Table>().map_err(|e| e.to_string())?;
    let mut build = section(&mut common, "build")?;
    let mut runtime = section(&mut common, "runtime")?;
    validate(&common, &["cuda-root", "tensorrt-root"])?;
    validate(
        &build,
        &["cuda-root", "tensorrt-root", "cuda-host-compiler"],
    )?;
    validate(
        &runtime,
        &[
            "cuda-root",
            "tensorrt-root",
            "cuda-library-dirs",
            "tensorrt-library-dirs",
            "search-policy",
        ],
    )?;
    for sdk in ["cuda", "tensorrt"] {
        let root = format!("{sdk}-root");
        let dirs = format!("{sdk}-library-dirs");
        if runtime.contains_key(&root) && runtime.contains_key(&dirs) {
            return Err(format!(
                "runtime.{root} and runtime.{dirs} are mutually exclusive"
            ));
        }
        if let Some(value) = common.get(&root) {
            build.entry(root.clone()).or_insert_with(|| value.clone());
            if !runtime.contains_key(&dirs) {
                runtime.entry(root).or_insert_with(|| value.clone());
            }
        }
    }
    match phase {
        "build" => Ok(build),
        "runtime" => Ok(runtime),
        _ => Err(format!("unknown platform configuration section: {phase}")),
    }
}
fn section(table: &mut Table, name: &str) -> Result<Table, String> {
    match table.remove(name) {
        None => Ok(Table::new()),
        Some(Value::Table(table)) => Ok(table),
        Some(_) => Err(format!("{name} must be a table")),
    }
}
fn validate(table: &Table, keys: &[&str]) -> Result<(), String> {
    for (key, value) in table {
        if !keys.contains(&key.as_str()) {
            return Err(format!("unknown platform setting: {key}"));
        }
        if matches!(key.as_str(), "cuda-library-dirs" | "tensorrt-library-dirs") {
            let values = value
                .as_array()
                .ok_or_else(|| format!("{key} must be an array of nonempty strings"))?;
            if values.is_empty()
                || values
                    .iter()
                    .any(|v| v.as_str().is_none_or(|s| s.trim().is_empty()))
            {
                return Err(format!(
                    "{key} must be a nonempty array of nonempty strings"
                ));
            }
        } else {
            let text = value
                .as_str()
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| format!("{key} must be a nonempty string"))?;
            if key == "search-policy" && !matches!(text, "explicit" | "discover") {
                return Err("search-policy must be explicit or discover".into());
            }
        }
    }
    Ok(())
}
pub fn user_config() -> Option<PathBuf> {
    #[cfg(windows)]
    let root = env::var_os("LOCALAPPDATA").map(PathBuf::from);
    #[cfg(not(windows))]
    let root = env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|v| PathBuf::from(v).join(".config")));
    root.map(|p| p.join("audio2face3d/platform.toml"))
}
