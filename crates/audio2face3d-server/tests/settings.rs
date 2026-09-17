#![allow(dead_code)]
use audio2face3d_server::config::{BackendKind, Config};
/// Construct test settings independently from the executable's argument parser.
pub fn config(values: impl IntoIterator<Item = impl AsRef<str>>) -> Config {
    let values: Vec<String> = values.into_iter().map(|s| s.as_ref().to_owned()).collect();
    let mut config = Config {
        backend: BackendKind::Mock,
        ..Config::default()
    };
    for pair in values[1..].chunks_exact(2) {
        let v = &pair[1];
        match pair[0].as_str() {
            "--backend" => config.backend = v.parse().unwrap(),
            "--model" => config.model = Some(v.into()),
            "--emotion-model" => config.emotion_model = Some(v.into()),
            "--mock-curve" => config.mock_curve = Some(v.clone()),
            "--mock-value" => config.mock_value = Some(v.parse().unwrap()),
            "--mock-jaw-open" => config.mock_jaw_open = Some(v.parse().unwrap()),
            "--max-streams" => config.max_streams = v.parse().unwrap(),
            "--request-queue-capacity" => config.request_queue_capacity = v.parse().unwrap(),
            "--request-queue-timeout-ms" => config.request_queue_timeout_ms = v.parse().unwrap(),
            "--max-audio-seconds" => config.max_audio_seconds = v.parse().unwrap(),
            "--input-idle-timeout-ms" => config.input_idle_timeout_ms = v.parse().unwrap(),
            "--output-timeout-ms" => config.output_timeout_ms = v.parse().unwrap(),
            _ => panic!("unknown test setting"),
        }
    }
    config
}
