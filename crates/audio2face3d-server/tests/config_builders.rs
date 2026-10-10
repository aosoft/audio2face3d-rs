#![cfg(feature = "mock")]
use audio2face3d::{
    client::{DirectConfig, Limits, ServerConfig},
    inference::{BackendKind, Config as InferenceConfig},
    types::{AudioFormat, EmotionPostProcessing, RequestOptions},
};
use audio2face3d_server::{ServerConfig as ServiceConfig, config::BackendKind as ServiceBackend};
use std::time::Duration;

#[cfg(feature = "native")]
#[test]
fn native_server_settings_check_files_without_initializing_their_contents_or_device() {
    let path = std::env::temp_dir().join(format!(
        "audio2face3d-r2-config-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, b"not a descriptor; only existence is checked here").unwrap();
    let engine = InferenceConfig::builder(BackendKind::Regression)
        .model(&path)
        .emotion_model(&path)
        .device(i32::MAX as usize)
        .max_audio_seconds(27)
        .build()
        .unwrap();
    let service = ServiceConfig::builder(ServiceBackend::Mock)
        .engine(engine.clone())
        .max_streams(3)
        .build()
        .unwrap()
        .into_builder()
        .build()
        .unwrap();
    assert_eq!(service.model(), engine.model());
    assert_eq!(service.emotion_model(), engine.emotion_model());
    assert_eq!(service.device(), i32::MAX as usize);
    assert_eq!(service.max_audio_seconds(), 27);
    assert_eq!(service.max_streams(), 3);
    assert!(
        service
            .clone()
            .into_builder()
            .optional_model(None)
            .build()
            .is_err()
    );
    std::fs::remove_file(&path).unwrap();
    assert!(
        service
            .validate()
            .unwrap_err()
            .contains("existing regression descriptor")
    );
}

#[test]
fn direct_and_service_share_domain_validation() {
    use std::path::PathBuf;
    type DomainCase = (
        BackendKind,
        Option<&'static str>,
        Option<f32>,
        Option<f32>,
        Option<&'static str>,
        Option<&'static str>,
        usize,
        u32,
        bool,
    );
    let cases: &[DomainCase] = &[
        (
            BackendKind::Mock,
            None,
            None,
            None,
            None,
            None,
            0,
            600,
            true,
        ),
        (
            BackendKind::Mock,
            Some("JawOpen"),
            Some(0.0),
            None,
            None,
            None,
            0,
            600,
            true,
        ),
        (
            BackendKind::Mock,
            Some("TongueOut"),
            Some(0.2),
            Some(0.8),
            None,
            None,
            0,
            600,
            true,
        ),
        (
            BackendKind::Mock,
            Some("TongueOut"),
            None,
            Some(0.0),
            None,
            None,
            0,
            1,
            true,
        ),
        (
            BackendKind::Mock,
            None,
            Some(0.0),
            None,
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            None,
            None,
            Some(0.0),
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            Some("JawOpen"),
            Some(0.2),
            Some(0.8),
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            Some("JawOpen"),
            None,
            Some(0.0),
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            Some("NotACurve"),
            None,
            None,
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            Some("TongueOut"),
            Some(f32::NAN),
            None,
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            Some("TongueOut"),
            Some(f32::INFINITY),
            None,
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            Some("TongueOut"),
            Some(-0.1),
            None,
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            Some("TongueOut"),
            Some(1.1),
            None,
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            Some("TongueOut"),
            None,
            Some(f32::NAN),
            None,
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            None,
            None,
            None,
            Some("unused.json"),
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            None,
            None,
            None,
            None,
            Some("unused.json"),
            0,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            None,
            None,
            None,
            None,
            None,
            1,
            600,
            false,
        ),
        (
            BackendKind::Mock,
            None,
            None,
            None,
            None,
            None,
            usize::MAX,
            600,
            false,
        ),
        (BackendKind::Mock, None, None, None, None, None, 0, 0, false),
        (
            BackendKind::Regression,
            Some("TongueOut"),
            None,
            None,
            Some("unused.json"),
            None,
            0,
            600,
            false,
        ),
        (
            BackendKind::Regression,
            None,
            None,
            None,
            None,
            None,
            0,
            600,
            false,
        ),
    ];
    for &(backend, curve, value, jaw, model, emotion, device, seconds, accepted) in cases {
        let direct = InferenceConfig::builder(backend)
            .optional_mock_curve(curve.map(str::to_owned))
            .optional_mock_value(value)
            .optional_mock_jaw_open(jaw)
            .optional_model(model.map(PathBuf::from))
            .optional_emotion_model(emotion.map(PathBuf::from))
            .device(device)
            .max_audio_seconds(seconds)
            .build();
        let service = ServiceConfig::builder(backend.into())
            .optional_mock_curve(curve.map(str::to_owned))
            .optional_mock_value(value)
            .optional_mock_jaw_open(jaw)
            .optional_model(model.map(PathBuf::from))
            .optional_emotion_model(emotion.map(PathBuf::from))
            .device(device)
            .max_audio_seconds(seconds)
            .build();
        assert_eq!(
            direct.is_ok(),
            accepted,
            "{backend:?} {curve:?} {value:?} {jaw:?}"
        );
        assert_eq!(
            service.is_ok(),
            accepted,
            "{backend:?} {curve:?} {value:?} {jaw:?}"
        );
        if let (Err(direct), Err(service)) = (direct, service) {
            assert_eq!(direct.message(), service.to_string());
        }
    }
    for curve in audio2face3d::inference::animation::CURVE_NAMES {
        assert!(
            InferenceConfig::builder(BackendKind::Mock)
                .mock_curve(curve)
                .mock_value(1.0)
                .build()
                .is_ok()
        );
        assert!(
            ServiceConfig::builder(ServiceBackend::Mock)
                .mock_curve(curve)
                .mock_value(1.0)
                .build()
                .is_ok()
        );
    }
}

#[test]
fn service_round_trip_and_engine_replacement_preserve_settings() {
    use audio2face3d::inference::MockPattern;
    let engine = InferenceConfig::builder(BackendKind::Mock)
        .mock_pattern(MockPattern::EyeBlinkRight)
        .mock_curve("TongueOut")
        .mock_value(0.0)
        .mock_jaw_open(0.7)
        .max_audio_seconds(23)
        .build()
        .unwrap();
    let service = ServiceConfig::builder(ServiceBackend::Regression)
        .model("discarded.json")
        .engine(engine.clone())
        .max_streams(2)
        .request_queue_capacity(3)
        .request_queue_timeout_ms(7)
        .max_message_bytes(8192)
        .output_queue_capacity(4)
        .input_idle_timeout_ms(11)
        .output_timeout_ms(13)
        .shutdown_timeout_ms(17)
        .build()
        .unwrap()
        .into_builder()
        .build()
        .unwrap();
    assert_eq!(service.backend(), ServiceBackend::Mock);
    assert!(service.model().is_none());
    assert!(service.emotion_model().is_none());
    assert_eq!(service.device(), 0);
    assert_eq!(service.mock_pattern(), MockPattern::EyeBlinkRight.into());
    assert_eq!(service.mock_curve(), engine.mock_curve());
    assert_eq!(service.mock_value(), Some(0.0));
    assert_eq!(service.mock_jaw_open(), Some(0.7));
    assert_eq!(service.max_audio_seconds(), 23);
    assert_eq!(service.max_streams(), 2);
    assert_eq!(service.request_queue_capacity(), 3);
    assert_eq!(service.request_queue_timeout_ms(), 7);
    assert_eq!(service.max_message_bytes(), 8192);
    assert_eq!(service.output_queue_capacity(), 4);
    assert_eq!(service.input_idle_timeout_ms(), 11);
    assert_eq!(service.input_timeout(), Duration::from_millis(11));
    assert_eq!(service.output_timeout_ms(), 13);
    assert_eq!(service.output_timeout(), Duration::from_millis(13));
    assert_eq!(service.shutdown_timeout_ms(), 17);
    assert_eq!(service.engine().mock_pattern(), MockPattern::EyeBlinkRight);

    let changed = service
        .into_builder()
        .engine(engine)
        .mock_value(0.5)
        .optional_mock_jaw_open(None)
        .build()
        .unwrap();
    assert_eq!(changed.mock_value(), Some(0.5));
    assert!(changed.mock_jaw_open().is_none());
    assert_eq!(changed.max_streams(), 2);
    let cleared = changed
        .into_builder()
        .optional_mock_value(None)
        .optional_mock_curve(None)
        .optional_model(None)
        .optional_emotion_model(None)
        .build()
        .unwrap();
    assert!(cleared.mock_value().is_none());
    assert!(cleared.mock_curve().is_none());
    assert!(cleared.into_builder().model("unused.json").build().is_err());
}

#[test]
fn service_keeps_its_transport_and_startup_validation() {
    let builder = ServiceConfig::builder(ServiceBackend::Mock);
    for invalid in [
        builder.clone().max_streams(0),
        builder.clone().request_queue_capacity(0),
        builder.clone().output_queue_capacity(0),
        builder.clone().max_message_bytes(4095),
        builder.clone().input_idle_timeout_ms(0),
        builder.clone().output_timeout_ms(0),
        builder.clone().shutdown_timeout_ms(0),
        builder
            .clone()
            .max_streams(tokio::sync::Semaphore::MAX_PERMITS + 1),
        builder
            .clone()
            .request_queue_capacity(tokio::sync::Semaphore::MAX_PERMITS + 1),
        builder
            .clone()
            .output_queue_capacity(tokio::sync::Semaphore::MAX_PERMITS + 1),
    ] {
        assert!(invalid.build().is_err());
    }
    assert_eq!(
        builder
            .request_queue_timeout_ms(0)
            .build()
            .unwrap()
            .request_queue_timeout_ms(),
        0
    );
    let engine = InferenceConfig::builder(BackendKind::Regression)
        .model("missing-regression-descriptor-for-r2.json")
        .build()
        .unwrap();
    assert!(
        ServiceConfig::builder(ServiceBackend::Mock)
            .engine(engine)
            .build()
            .is_err()
    );
}

#[test]
fn legacy_enum_paths_remain_distinct_and_parse_the_same_strings() {
    use audio2face3d::inference::MockPattern;
    use audio2face3d_server::config::MockPattern as ServicePattern;
    trait ConsumerTrait {
        fn consumer_marker(&self) -> u8;
    }
    impl ConsumerTrait for BackendKind {
        fn consumer_marker(&self) -> u8 {
            1
        }
    }
    impl ConsumerTrait for ServiceBackend {
        fn consumer_marker(&self) -> u8 {
            2
        }
    }
    impl ConsumerTrait for MockPattern {
        fn consumer_marker(&self) -> u8 {
            3
        }
    }
    impl ConsumerTrait for ServicePattern {
        fn consumer_marker(&self) -> u8 {
            4
        }
    }
    for name in ["mock", "regression"] {
        let direct: BackendKind = name.parse().unwrap();
        let service: ServiceBackend = name.parse().unwrap();
        assert_eq!(BackendKind::from(service), direct);
        assert_eq!(direct.consumer_marker(), 1);
        assert_eq!(service.consumer_marker(), 2);
    }
    for name in [
        "jaw-open-pulse",
        "eye-blink-left",
        "eye-blink-right",
        "mouth-smile-left",
        "mouth-smile-right",
    ] {
        let direct: MockPattern = name.parse().unwrap();
        let service: ServicePattern = name.parse().unwrap();
        assert_eq!(MockPattern::from(service), direct);
        assert_eq!(direct.consumer_marker(), 3);
        assert_eq!(service.consumer_marker(), 4);
    }
    for name in ["Mock", "invalid", ""] {
        assert_eq!(
            name.parse::<BackendKind>().unwrap_err(),
            name.parse::<ServiceBackend>().unwrap_err()
        );
        assert_eq!(
            name.parse::<MockPattern>().unwrap_err(),
            name.parse::<ServicePattern>().unwrap_err()
        );
    }
}

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
