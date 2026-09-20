#[allow(dead_code)]
#[path = "../build_native.rs"]
mod native_build;
use native_build::{config_candidates, parse_config};
use std::path::PathBuf;
fn file() -> PathBuf {
    std::env::temp_dir().join("a2f-build-test/native-build.toml")
}
#[test]
fn file_configuration_is_complete_and_relative_to_itself() {
    let file = file();
    let config=parse_config("schema_version=1\ncuda_root='cuda'\ntensorrt_root='trt'\ncuda_archs=['86','89']\ncuda_host_compiler='compiler'", &file, true).unwrap();
    assert_eq!(config.cuda_root, file.parent().unwrap().join("cuda"));
    assert_eq!(
        config.tensorrt_root,
        Some(file.parent().unwrap().join("trt"))
    );
    assert_eq!(config.cuda_archs, "86,89");
    assert_eq!(
        config.cuda_host_compiler,
        Some(file.parent().unwrap().join("compiler"))
    );
}
#[test]
fn incomplete_or_unknown_configuration_does_not_fall_back() {
    for text in [
        "schema_version=1",
        "schema_version=2\ncuda_root='cuda'",
        "schema_version=1\ncuda_root='cuda'\nunknown=true",
        "schema_version=1\ncuda_root='cuda'\ncuda_archs=[]",
    ] {
        assert!(parse_config(text, &file(), false).is_err(), "{text}");
    }
    assert!(parse_config("schema_version=1\ncuda_root='cuda'", &file(), true).is_err());
    assert!(parse_config("schema_version=1\ncuda_root='cuda'", &file(), false).is_ok());
}
#[test]
fn explicit_file_is_exclusive_and_registry_location_does_not_search_cwd() {
    let selected = file();
    assert_eq!(
        config_candidates(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")),
            Some(file().with_file_name("user.toml")),
            Some(selected.clone())
        ),
        vec![selected]
    );
    let registry = std::env::temp_dir().join("cargo-registry/src/audio2face3d-0.1.0");
    assert_eq!(
        config_candidates(&registry, Some(file()), None),
        vec![file()]
    );
}

#[test]
fn version_headers_resolve_aliases_and_reject_cycles() {
    assert_eq!(
        native_build::parse_define("#define A B\n#define B (12)", "A").unwrap(),
        12
    );
    assert!(native_build::parse_define("#define A B\n#define B A", "A").is_err());
    assert!(native_build::parse_define("#define OTHER 1", "A").is_err());
}
