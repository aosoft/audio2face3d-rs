//! Resolve one explicitly selected MSVC installation without changing the parent environment.
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    process::Command,
};

type Environment = Vec<(OsString, OsString)>;

pub fn architectures(
    host: &str,
    target: &str,
) -> Result<(&'static str, &'static str, &'static str), String> {
    if !host.ends_with("windows-msvc") || !target.ends_with("windows-msvc") {
        return Err("build-cuda.windows requires Windows MSVC host and target".into());
    }
    match (host.split('-').next(), target.split('-').next()) {
        (Some("x86_64"), Some("x86_64")) => Ok(("Hostx64", "x64", "amd64")),
        (Some("x86_64"), Some("aarch64")) => Ok(("Hostx64", "arm64", "amd64_arm64")),
        (Some("x86_64"), Some("i686")) => Ok(("Hostx64", "x86", "amd64_x86")),
        (Some("aarch64"), Some("aarch64")) => Ok(("Hostarm64", "arm64", "arm64")),
        _ => Err(format!(
            "unsupported MSVC host/target pair: {host} -> {target}"
        )),
    }
}

pub fn parse_environment(bytes: &[u8]) -> Result<Environment, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err("invalid UTF-16 MSVC environment output".into());
    }
    let words: Vec<_> = bytes
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect();
    let text = String::from_utf16(&words).map_err(|e| e.to_string())?;
    let (_, environment) = text
        .split_once("A2F_ENV_BEGIN\r\n")
        .ok_or("MSVC environment marker missing")?;
    Ok(environment
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once('=')?;
            // Skip cmd's hidden drive variables and our temporary launcher variables.
            if key.is_empty() || key.starts_with("A2F_VCVARS") {
                return None;
            }
            Some((key.into(), value.into()))
        })
        .collect())
}

pub fn resolve(
    root: &Path,
    version: &str,
    host: &str,
    target: &str,
) -> Result<(PathBuf, Environment), String> {
    let (host_dir, target_dir, architecture) = architectures(host, target)?;
    let tools = root.join("VC/Tools/MSVC").join(version);
    let compiler = tools
        .join("bin")
        .join(host_dir)
        .join(target_dir)
        .join("cl.exe");
    let batch = root.join("VC/Auxiliary/Build/vcvarsall.bat");
    for path in [
        &compiler,
        &compiler.with_file_name("lib.exe"),
        &tools.join("include/yvals_core.h"),
        &batch,
    ] {
        if !path.is_file() {
            return Err(format!(
                "requested MSVC {version}: missing {}",
                path.display()
            ));
        }
        println!("cargo:rerun-if-changed={}", path.display());
    }
    // The only interpolated shell argument is a validated numeric version.
    if version.split('.').count() != 3
        || version
            .split('.')
            .any(|s| s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err("invalid MSVC version".into());
    }
    if batch.to_string_lossy().contains(['%', '"', '\r', '\n']) {
        return Err("Visual Studio root contains unsupported command-shell characters".into());
    }
    let mut cmd = Command::new("cmd.exe");
    cmd.args(["/d", "/u", "/c"]);
    let script = format!(
        "call \"%A2F_VCVARS%\" {architecture} -vcvars_ver={version} >nul && echo A2F_ENV_BEGIN&& set"
    );
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.raw_arg(&script);
    }
    #[cfg(not(windows))]
    cmd.arg(&script);
    cmd.env("A2F_VCVARS", &batch);
    // A previous Developer Prompt must not contaminate the selected toolset.
    for (key, _) in std::env::vars_os() {
        let upper = key.to_string_lossy().to_ascii_uppercase();
        if matches!(
            upper.as_str(),
            "INCLUDE" | "LIB" | "LIBPATH" | "CL" | "_CL_" | "LINK" | "__VSCMD_PREINIT_PATH"
        ) || [
            "VS",
            "VC",
            "WINDOWSSDK",
            "WINDOWSLIB",
            "UNIVERSALCRT",
            "UCRT",
            "EXTENSIONSDK",
            "FRAMEWORK",
            "NETFXSDK",
        ]
        .iter()
        .any(|prefix| upper.starts_with(prefix))
        {
            cmd.env_remove(&key);
        }
    }
    let system = std::env::var_os("SystemRoot").ok_or("SystemRoot is missing")?;
    let system = PathBuf::from(system);
    cmd.env(
        "PATH",
        std::env::join_paths([
            system.join("System32"),
            system.clone(),
            system.join("System32/Wbem"),
        ])
        .map_err(|e| e.to_string())?,
    );
    let output = cmd
        .output()
        .map_err(|e| format!("cannot initialize MSVC environment: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "vcvarsall.bat failed for MSVC {version} ({})",
            output.status
        ));
    }
    let mut environment = parse_environment(&output.stdout)?;
    let get = |name: &str| {
        environment
            .iter()
            .find(|(k, _)| k.to_string_lossy().eq_ignore_ascii_case(name))
            .map(|(_, v)| v.to_string_lossy().into_owned())
    };
    let selected = get("VCToolsInstallDir").ok_or("vcvarsall did not set VCToolsInstallDir")?;
    if Path::new(&selected)
        .canonicalize()
        .map_err(|e| e.to_string())?
        != tools.canonicalize().map_err(|e| e.to_string())?
    {
        return Err(format!(
            "vcvarsall selected a different toolset than requested MSVC {version}"
        ));
    }
    for (name, expected) in [
        ("INCLUDE", tools.join("include")),
        ("LIB", tools.join("lib").join(target_dir)),
    ] {
        let value = get(name).ok_or_else(|| format!("MSVC environment missing {name}"))?;
        let expected = expected.canonicalize().map_err(|e| e.to_string())?;
        if !std::env::split_paths(&value).any(|p| p.canonicalize().ok().as_ref() == Some(&expected))
        {
            return Err(format!(
                "MSVC {version} environment has inconsistent {name}"
            ));
        }
    }
    // Command::envs overlays the parent: explicitly clear compiler injection variables.
    for key in ["CL", "_CL_", "LINK"] {
        environment.push((key.into(), "".into()));
    }
    Ok((compiler, environment))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn architecture_selection_rejects_non_msvc_targets() {
        assert_eq!(
            architectures("x86_64-pc-windows-msvc", "x86_64-pc-windows-msvc").unwrap(),
            ("Hostx64", "x64", "amd64")
        );
        assert!(architectures("x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc").is_err());
    }
    #[test]
    fn environment_parsing_preserves_unicode_and_equals() {
        let text = "noise\r\nA2F_ENV_BEGIN\r\nINCLUDE=C:\\日本語\r\nVALUE=a=b\r\n=C:=ignored\r\n";
        let bytes: Vec<_> = text.encode_utf16().flat_map(u16::to_le_bytes).collect();
        let values = parse_environment(&bytes).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].1, OsString::from("C:\\日本語"));
        assert_eq!(values[1].1, OsString::from("a=b"));
        assert!(parse_environment(&[1]).is_err());
    }
}
