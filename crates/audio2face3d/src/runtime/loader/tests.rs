use super::*;
use crate::runtime::{NativeRuntimeConfig, NativeSearchPolicy, discovery, registry::Registry};
use std::{fs, process::Command};
fn compile(source: &Path, name: &str, directory: &Path, libraries: Option<&Path>) {
    let mut command = Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()));
    command
        .args([
            "--edition=2024",
            "--crate-type=cdylib",
            "--crate-name",
            name,
        ])
        .arg(source)
        .arg("--out-dir")
        .arg(directory);
    if let Some(libraries) = libraries {
        command.arg("-L").arg(libraries);
    }
    #[cfg(unix)]
    command.arg("-C").arg("link-arg=-Wl,-rpath,$ORIGIN");
    let output = command
        .output()
        .expect("run rustc for native loader fixture");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn filename(name: &str) -> String {
    format!(
        "{}{name}{}",
        std::env::consts::DLL_PREFIX,
        std::env::consts::DLL_SUFFIX
    )
}
#[test]
fn native_fixture_absolute_path_dependencies_symbols_identity_and_conflicts() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .join("../../temp/native-runtime-work/fixtures")
        .join(format!("loader-{}", std::process::id()));
    let valid = root.join("空白 path");
    let missing = root.join("missing");
    let other = root.join("other");
    for path in [&valid, &missing, &other] {
        fs::create_dir_all(path).unwrap();
    }
    let sources = manifest.join("tests/fixtures/native_loader");
    compile(
        &sources.join("dependency.rs"),
        "a2f_fixture_dependency",
        &valid,
        None,
    );
    #[cfg(windows)]
    fs::copy(
        valid.join("a2f_fixture_dependency.dll.lib"),
        valid.join("a2f_fixture_dependency.lib"),
    )
    .unwrap();
    compile(
        &sources.join("library.rs"),
        "a2f_fixture_main",
        &valid,
        Some(&valid),
    );
    compile(
        &sources.join("library.rs"),
        "a2f_fixture_missing",
        &missing,
        Some(&valid),
    );
    let absent = LibraryFile::resolve(&missing.join(filename("a2f_fixture_missing"))).unwrap();
    let failed = Registry::new();
    // SAFETY: the freshly compiled fixture is trusted test code.
    let error = failed
        .initialize((), |attempt| {
            // SAFETY: freshly compiled trusted fixture.
            unsafe { LoadedLibrary::open(absent, &[], attempt) }.map(|_| ())
        })
        .unwrap_err();
    assert_eq!(error.kind(), NativeRuntimeErrorKind::DependencyLoadFailed);
    assert!(error.restart_required());
    let file = LibraryFile::resolve(&valid.join(filename("a2f_fixture_main"))).unwrap();
    let registry = Registry::new();
    // SAFETY: the freshly compiled fixture is trusted test code.
    let loaded = registry
        .initialize(file.identity.clone(), |attempt| unsafe {
            LoadedLibrary::open(file.clone(), std::slice::from_ref(&valid), attempt)
        })
        .unwrap();
    // SAFETY: fixture_version was compiled with this signature above.
    let version =
        unsafe { loaded.symbol::<unsafe extern "C" fn() -> u32>(b"fixture_version\0") }.unwrap();
    // SAFETY: the fixture's module and dependency are retained permanently.
    assert_eq!(unsafe { version() }, 12090);
    // SAFETY: missing symbol cannot produce a callable value.
    assert_eq!(
        // SAFETY: a missing export returns an error without dereferencing.
        unsafe { loaded.symbol::<unsafe extern "C" fn()>(b"missing_symbol\0") }
            .unwrap_err()
            .kind(),
        NativeRuntimeErrorKind::SymbolMissing
    );
    let alias = other.join(filename("alias"));
    fs::hard_link(&file.path, &alias).unwrap();
    assert_eq!(
        LibraryFile::resolve(&alias).unwrap().identity,
        file.identity
    );
    let duplicate = other.join(filename("a2f_fixture_main"));
    fs::copy(&file.path, &duplicate).unwrap();
    let duplicate = LibraryFile::resolve(&duplicate).unwrap();
    assert_ne!(duplicate.identity, file.identity);
    #[cfg(windows)]
    {
        let conflict = Registry::new();
        // SAFETY: fixture copy is trusted; loader must reject its conflict with the loaded original.
        let result = conflict.initialize((), |attempt| {
            // SAFETY: trusted fixture copy; only loading is attempted.
            unsafe { LoadedLibrary::open(duplicate, &[], attempt) }.map(|_| ())
        });
        assert_eq!(
            result.unwrap_err().kind(),
            NativeRuntimeErrorKind::RuntimeConflict
        );
    }
    assert!(LibraryFile::resolve(Path::new("relative.dll")).is_err());
    let config = NativeRuntimeConfig::builder()
        .cuda_library_dirs([missing.clone()])
        .build()
        .unwrap();
    let dirs = discovery::directories(&config, discovery::Sdk::Cuda).unwrap();
    assert_eq!(
        discovery::library(&dirs, "does_not_exist", ".dll")
            .unwrap_err()
            .kind(),
        NativeRuntimeErrorKind::LibraryNotFound
    );
    assert!(
        discovery::directories(
            &NativeRuntimeConfig::builder()
                .search_policy(NativeSearchPolicy::ExplicitOnly)
                .build()
                .unwrap(),
            discovery::Sdk::Cuda
        )
        .is_err()
    );
    assert_eq!(
        discovery::library(
            &[valid, other],
            &format!("{}a2f_fixture_main", std::env::consts::DLL_PREFIX),
            std::env::consts::DLL_SUFFIX
        )
        .unwrap_err()
        .kind(),
        NativeRuntimeErrorKind::RuntimeConflict
    );
}
