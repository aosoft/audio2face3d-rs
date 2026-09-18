#![cfg(feature = "cli")]
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsStr;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Deserialize)]
struct ArtifactManifest {
    schema_version: u64,
    sources: Sources,
    cases: BTreeMap<String, ArtifactCase>,
}

#[derive(Deserialize)]
struct Sources {
    audio2face_sdk: String,
}

#[derive(Deserialize)]
struct ArtifactCase {
    roots: Vec<String>,
    files: Vec<ArtifactFile>,
}

#[derive(Deserialize)]
struct ArtifactFile {
    path: String,
    bytes: u64,
    sha256: String,
}

#[test]
#[ignore = "hashes the complete local NVIDIA SDK reference fixture set"]
fn reference_artifact_manifest_matches_sdk_checkout() {
    let sdk_root = env::var_os("AUDIO2FACE_SDK_ROOT")
        .map(PathBuf::from)
        .expect("AUDIO2FACE_SDK_ROOT must name the NVIDIA SDK checkout");
    let generated = sdk_root.join("_data/generated");
    assert!(generated.is_dir(), "missing {}", generated.display());

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let manifest_path = workspace.join("reference/artifacts.json");
    let manifest: ArtifactManifest =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    assert_eq!(manifest.schema_version, 1);
    assert_eq!(
        git_revision(&sdk_root),
        manifest.sources.audio2face_sdk,
        "reference SDK revision changed"
    );

    for (case_name, case) in manifest.cases {
        let expected = case
            .files
            .into_iter()
            .map(|file| (file.path.clone(), file))
            .collect::<BTreeMap<_, _>>();
        let actual = collect_case_paths(&generated, &case.roots);
        let expected_paths = expected.keys().cloned().collect::<BTreeSet<_>>();
        assert_eq!(
            actual, expected_paths,
            "fixture set changed for {case_name}"
        );

        for (relative, expected) in expected {
            let path = generated.join(&relative);
            let metadata = path.metadata().unwrap();
            assert_eq!(metadata.len(), expected.bytes, "size changed: {relative}");
            assert_eq!(sha256(&path), expected.sha256, "hash changed: {relative}");
        }
    }
}

fn collect_case_paths(generated: &Path, roots: &[String]) -> BTreeSet<String> {
    let mut result = BTreeSet::new();
    for relative_root in roots {
        let root = generated.join(relative_root);
        assert!(root.is_dir(), "missing fixture root {}", root.display());
        let mut files = Vec::new();
        visit_files(&root, &mut files).unwrap();
        for path in files.into_iter().filter(|path| is_fixture(path)) {
            let suffix = path
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            assert!(result.insert(format!("{relative_root}/{suffix}")));
        }
    }
    result
}

fn visit_files(root: &Path, files: &mut Vec<PathBuf>) -> io::Result<()> {
    for entry in fs::read_dir(root)? {
        let path = entry?.path();
        if path.is_dir() {
            visit_files(&path, files)?;
        } else if path.is_file() {
            files.push(path);
        }
    }
    Ok(())
}

fn is_fixture(path: &Path) -> bool {
    matches!(
        path.extension().and_then(OsStr::to_str),
        Some("bin" | "json" | "npz" | "onnx" | "trt" | "wav")
    )
}

fn sha256(path: &Path) -> String {
    let mut stream = File::open(path)
        .unwrap_or_else(|error| panic!("failed to open {}: {error}", path.display()));
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let count = stream
            .read(&mut buffer)
            .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
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
    encoded
}

fn git_revision(root: &Path) -> String {
    let output = Command::new("git")
        .args([
            OsStr::new("-C"),
            root.as_os_str(),
            OsStr::new("rev-parse"),
            OsStr::new("HEAD"),
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}
