#![cfg(any(feature = "mock", feature = "native", feature = "grpc-server"))]

use audio2face3d::inference::{BackendKind, Config, MockPattern};

#[test]
fn contradictory_mock_settings_are_rejected_without_loading_a_backend() {
    let mock = Config::builder(BackendKind::Mock);
    for invalid in [
        mock.clone().mock_value(0.0),
        mock.clone().mock_jaw_open(0.0),
        mock.clone()
            .mock_curve("JawOpen")
            .mock_value(0.2)
            .mock_jaw_open(0.8),
        mock.clone().model("unused.json"),
        mock.clone().emotion_model("unused.json"),
        mock.clone().device(1),
        Config::builder(BackendKind::Regression)
            .model("unused.json")
            .mock_curve("TongueOut"),
    ] {
        assert_eq!(
            invalid.build().unwrap_err().kind(),
            audio2face3d::types::ErrorKind::InvalidInput
        );
    }
}

#[test]
fn regression_settings_round_trip_without_loading_native_resources() {
    let config = Config::builder(BackendKind::Regression)
        .model("not-opened-regression.json")
        .emotion_model("not-opened-emotion.json")
        .device(3)
        .max_audio_seconds(17)
        .mock_pattern(MockPattern::EyeBlinkRight)
        .build()
        .unwrap();
    let rebuilt = config.clone().into_builder().build().unwrap();
    assert_eq!(rebuilt.backend(), config.backend());
    assert_eq!(rebuilt.model(), config.model());
    assert_eq!(rebuilt.emotion_model(), config.emotion_model());
    assert_eq!(rebuilt.device(), 3);
    assert_eq!(rebuilt.max_audio_seconds(), 17);
    assert_eq!(rebuilt.mock_pattern(), MockPattern::EyeBlinkRight);
    assert!(rebuilt.mock_curve().is_none());
    assert!(rebuilt.mock_value().is_none());
    assert!(rebuilt.mock_jaw_open().is_none());
}

#[test]
fn mock_configuration_validation_does_not_require_the_mock_feature() {
    let config = Config::builder(BackendKind::Mock)
        .mock_curve("TongueOut")
        .mock_value(0.0)
        .mock_jaw_open(0.0)
        .build()
        .unwrap();
    let rebuilt = config.into_builder().build().unwrap();
    assert_eq!(rebuilt.mock_value(), Some(0.0));
    assert_eq!(rebuilt.mock_jaw_open(), Some(0.0));
    let cleared = rebuilt
        .into_builder()
        .optional_mock_value(None)
        .optional_mock_jaw_open(None)
        .optional_mock_curve(None)
        .build()
        .unwrap();
    assert!(cleared.mock_value().is_none());
    assert!(cleared.mock_jaw_open().is_none());
    assert!(cleared.mock_curve().is_none());
}
