#![cfg(feature = "mock")]
use audio2face3d::{
    client::{DirectConfig, Limits, ServerConfig},
    inference::{BackendKind, Config as InferenceConfig},
    types::{AudioFormat, EmotionPostProcessing, RequestOptions},
};
use audio2face3d_server::{ServerConfig as ServiceConfig, config::BackendKind as ServiceBackend};
use std::time::Duration;

#[test]
fn builders_validate_nested_limits_and_backend_requirements() {
    let limits = Limits::builder().max_requests(3).build().unwrap();
    let engine = InferenceConfig::builder(BackendKind::Mock)
        .mock_curve("JawOpen")
        .mock_value(0.0)
        .build()
        .unwrap();
    let direct = DirectConfig::builder(engine.clone())
        .limits(limits.clone())
        .max_executions(2)
        .max_queued(0)
        .queue_timeout(Duration::ZERO)
        .build()
        .unwrap();
    assert_eq!(direct.max_executions(), 2);
    assert_eq!(direct.max_queued(), 0);
    assert_eq!(direct.engine().mock_value(), Some(0.0));
    assert!(
        DirectConfig::builder(engine)
            .max_executions(0)
            .build()
            .is_err()
    );
    assert!(Limits::builder().input_queue_bytes(1).build().is_err());
    assert!(
        InferenceConfig::builder(BackendKind::Regression)
            .build()
            .is_err()
    );
    let remote = ServerConfig::builder("http://127.0.0.1:52000")
        .limits(limits)
        .api_key("SentinelBuilderSecret")
        .build()
        .unwrap();
    assert_eq!(remote.api_key().as_deref(), Some("SentinelBuilderSecret"));
    assert!(!format!("{remote:?}").contains("SentinelBuilderSecret"));
    assert!(ServerConfig::builder("not a URL").build().is_err());
    assert!(
        ServerConfig::builder("http://localhost")
            .api_key("")
            .build()
            .is_err()
    );
    assert!(
        ServerConfig::builder("http://localhost")
            .connect_timeout(Duration::ZERO)
            .build()
            .is_err()
    );
    // Constructing a remote config does not require a running Tokio runtime.
    assert!(
        ServerConfig::builder("http://localhost")
            .api_key("key")
            .optional_api_key(None)
            .build()
            .unwrap()
            .api_key()
            .is_none()
    );
}

#[test]
fn request_and_service_builders_preserve_unset_zero_and_false() {
    let default = RequestOptions::builder(AudioFormat::MONO_16KHZ)
        .build()
        .unwrap();
    assert!(default.emotion_post_processing().is_none());
    let mut post = EmotionPostProcessing::default();
    post.use_preferred = Some(false);
    post.preferred_strength = Some(0.0);
    let options = RequestOptions::builder(AudioFormat::MONO_16KHZ)
        .emotion_post_processing(post.clone())
        .timeout(Duration::ZERO)
        .build()
        .unwrap();
    assert_eq!(options.emotion_post_processing(), &Some(post));
    assert_eq!(options.timeout(), Some(Duration::ZERO));
    let mut invalid = EmotionPostProcessing::default();
    invalid.smoothing = Some(f32::NAN);
    assert!(
        RequestOptions::builder(AudioFormat::MONO_16KHZ)
            .emotion_post_processing(invalid)
            .build()
            .is_err()
    );
    let service = ServiceConfig::builder(ServiceBackend::Mock)
        .max_streams(2)
        .request_queue_timeout_ms(0)
        .mock_curve("JawOpen")
        .mock_value(0.0)
        .build()
        .unwrap();
    assert_eq!(service.mock_value(), Some(0.0));
    assert_eq!(service.max_streams(), 2);
    assert!(
        ServiceConfig::builder(ServiceBackend::Mock)
            .max_streams(0)
            .build()
            .is_err()
    );
}
