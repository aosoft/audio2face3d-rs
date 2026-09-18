#![allow(dead_code)]
use audio2face3d_server::config::{BackendKind, Config};
/// Construct test settings independently from the executable's argument parser.
pub fn config(values: impl IntoIterator<Item = impl AsRef<str>>) -> Config {
    try_config(values).unwrap()
}
pub fn try_config(
    values: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<Config, audio2face3d_server::ConfigError> {
    let values: Vec<String> = values.into_iter().map(|s| s.as_ref().to_owned()).collect();
    let mut config = Config::builder(BackendKind::Mock);
    for pair in values[1..].chunks_exact(2) {
        let v = &pair[1];
        match pair[0].as_str() {
            "--backend" => config = config.backend(v.parse().unwrap()),
            "--model" => config = config.optional_model(Some(v.into())),
            "--emotion-model" => config = config.optional_emotion_model(Some(v.into())),
            "--mock-curve" => config = config.optional_mock_curve(Some(v.clone())),
            "--mock-value" => config = config.optional_mock_value(Some(v.parse().unwrap())),
            "--mock-jaw-open" => config = config.optional_mock_jaw_open(Some(v.parse().unwrap())),
            "--max-streams" => config = config.max_streams(v.parse().unwrap()),
            "--request-queue-capacity" => {
                config = config.request_queue_capacity(v.parse().unwrap())
            }
            "--request-queue-timeout-ms" => {
                config = config.request_queue_timeout_ms(v.parse().unwrap())
            }
            "--max-audio-seconds" => config = config.max_audio_seconds(v.parse().unwrap()),
            "--input-idle-timeout-ms" => config = config.input_idle_timeout_ms(v.parse().unwrap()),
            "--output-timeout-ms" => config = config.output_timeout_ms(v.parse().unwrap()),
            _ => panic!("unknown test setting"),
        }
    }
    config.build()
}
