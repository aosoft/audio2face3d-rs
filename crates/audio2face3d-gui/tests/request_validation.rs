use audio2face3d_gui::inference::{Job, Mode, Request};
use std::sync::Arc;

#[test]
fn missing_model_is_rejected_before_opening_wav_or_starting_native_worker() {
    // The WAV need not exist: required-field validation must happen first.
    let request = Request {
        mode: Mode::Local,
        wav: "unopened.wav".into(),
        ..Default::default()
    };
    let result = Job::start(1, request, Arc::new(audio2face3d::logging::NoopLogger));
    let error = match result {
        Ok(_) => panic!("missing model started a worker"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("Model JSON is required"));
    assert!(error.contains("Browse next to Model JSON"));
}

#[test]
fn missing_wav_is_reported_before_a_session_is_created() {
    let result = Job::start(
        2,
        Request::default(),
        Arc::new(audio2face3d::logging::NoopLogger),
    );
    let error = match result {
        Ok(_) => panic!("missing WAV started a worker"),
        Err(error) => error.to_string(),
    };
    assert!(error.contains("Browse next to WAV"));
}
