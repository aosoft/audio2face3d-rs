use audio2face3d::{
    Audio2Face3DContext,
    runtime::{
        NativeRuntimeConfig, NativeRuntimeErrorKind, NativeSearchPolicy, NativeVersion,
        VersionCompatibility,
    },
};
use std::path::PathBuf;

fn absolute(name: &str) -> PathBuf {
    std::env::temp_dir().join(name)
}

#[test]
fn native_settings_are_lazy_and_preserved_by_context_clone() {
    let path = absolute("audio2face3d-config-test-does-not-need-to-exist");
    let config = NativeRuntimeConfig::builder()
        .cuda_root(&path)
        .search_policy(NativeSearchPolicy::ExplicitOnly)
        .build()
        .unwrap();
    let context = Audio2Face3DContext::builder()
        .native_runtime(config.clone())
        .build();
    assert_eq!(context.clone().native_runtime(), &config);
    assert_eq!(context.native_runtime().cuda_root(), Some(path.as_path()));
    assert_eq!(
        Audio2Face3DContext::default()
            .native_runtime()
            .search_policy(),
        NativeSearchPolicy::Discover
    );
}

#[test]
fn invalid_paths_and_conflicting_sources_are_rejected_without_loading() {
    for path in ["", "relative", "bad\0path"] {
        assert_eq!(
            NativeRuntimeConfig::builder()
                .cuda_root(path)
                .build()
                .unwrap_err()
                .kind(),
            NativeRuntimeErrorKind::InvalidConfig
        );
    }
    let error = NativeRuntimeConfig::builder()
        .tensorrt_root(absolute("root"))
        .tensorrt_library_dirs([absolute("libs")])
        .build()
        .unwrap_err();
    assert_eq!(error.kind(), NativeRuntimeErrorKind::InvalidConfig);
    assert!(!error.restart_required());
}

#[test]
fn version_policy_is_symmetric_for_minor_and_ignores_patch_and_build() {
    let built = NativeVersion::new(12, 9, Some(1), Some(10));
    for minor in [8, 10] {
        assert_eq!(
            built.compatibility(NativeVersion::new(12, minor, None, None)),
            VersionCompatibility::MinorMismatch
        );
    }
    for major in [11, 13] {
        assert_eq!(
            built.compatibility(NativeVersion::new(major, 9, None, None)),
            VersionCompatibility::MajorMismatch
        );
    }
    assert_eq!(
        built.compatibility(NativeVersion::new(12, 9, Some(99), Some(99))),
        VersionCompatibility::Compatible
    );
    assert_eq!(NativeVersion::new(12, 9, None, None).to_string(), "12.9");
}

#[test]
fn an_explicit_empty_directory_list_is_not_discovery() {
    assert_eq!(
        NativeRuntimeConfig::builder()
            .cuda_library_dirs([])
            .build()
            .unwrap_err()
            .kind(),
        NativeRuntimeErrorKind::InvalidConfig
    );
}
