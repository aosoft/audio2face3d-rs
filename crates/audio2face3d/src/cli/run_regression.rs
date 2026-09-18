use crate::cli::async_util::block_on;
use audio2face3d::audio2face::bundle::GeometryExecutorBundleCreationParameters;
use audio2face3d::audio2face::regression::RegressionGeometryExecutorCreationParameters;
use audio2face3d::audio2face::{
    GeometryCallbacks, GeometryExecutionOption, GeometryExecutor, GeometryExecutorBundle,
    GeometryExecutorBundleFactory, GeometryExecutorCreationParameters, GeometryTrackResources,
};
use audio2face3d::audio2x::{AudioAccumulator, EmotionAccumulator, ExecutionState, FrameRate};
use audio2face3d::common::{GeometryParameters, NetworkDocument};
use audio2face3d::{Model, ModelKind};
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;

pub fn run(
    model_path: &Path,
    tracks: usize,
    samples: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::load(model_path)?;
    if model.kind() != ModelKind::Regression {
        return Err("expected a regression model".into());
    }
    let NetworkDocument::Geometry(network) = model.network() else {
        return Err("regression network is missing".into());
    };
    let GeometryParameters::Regression(parameters) = &network.params else {
        return Err("regression network parameters are missing".into());
    };
    let audio_parameters = match &network.audio_params {
        audio2face3d::common::GeometryAudioParameters::Regression(value) => value,
        _ => return Err("regression audio parameters are missing".into()),
    };
    let config = match model.parameters(0)? {
        audio2face3d::ModelParameters::Geometry(value) => value,
        _ => return Err("regression geometry config is missing".into()),
    };
    let frame_rate = FrameRate::new(30, 1)?;
    let resources = (0..tracks)
        .map(|_| {
            let audio = Arc::new(AudioAccumulator::new(audio_parameters.buffer_len, 0)?);
            let emotions = Arc::new(
                EmotionAccumulator::new(parameters.explicit_emotions.len(), 30)
                    .map_err(|error| audio2face3d::Error::InvalidSchema(error.to_string()))?,
            );
            emotions
                .accumulate(0, &parameters.default_emotion)
                .map_err(|error| audio2face3d::Error::InvalidSchema(error.to_string()))?;
            emotions
                .close()
                .map_err(|error| audio2face3d::Error::InvalidSchema(error.to_string()))?;
            Ok(GeometryTrackResources { audio, emotions })
        })
        .collect::<Result<Vec<_>, audio2face3d::Error>>()?;
    let common = GeometryExecutorCreationParameters {
        tracks: resources,
        device_ordinal: 0,
        execution_option: GeometryExecutionOption::ALL,
    };
    let mut bundle = block_on(GeometryExecutorBundleFactory::load(
        GeometryExecutorBundleCreationParameters::Regression(
            RegressionGeometryExecutorCreationParameters {
                model_path: model_path.to_owned(),
                common,
                input_strength: config.input_strength,
                frame_rate,
                source_emotion_shot: config.source_shot.clone(),
                source_emotion_frame: config
                    .source_frame
                    .and_then(|value| usize::try_from(value).ok())
                    .unwrap_or(0),
            },
        ),
    ))?;
    let audio = vec![0.0_f32; samples.unwrap_or(model.sample_rate())];
    for track in 0..tracks {
        bundle.audio_accumulator(track)?.accumulate(&audio)?;
        bundle.audio_accumulator(track)?.close()?;
    }
    let mut callbacks = 0_usize;
    loop {
        let execution = match &mut bundle {
            GeometryExecutorBundle::Regression(executor) => {
                executor.execute(GeometryCallbacks {
                    results: &mut |results| {
                        callbacks += 1;
                        let values = results.skin.map_or(0, |value| value.values.len())
                            + results.tongue.map_or(0, |value| value.values.len())
                            + results.jaw.map_or(0, |value| value.values.len())
                            + results.eyes.map_or(0, |value| value.values.len());
                        println!(
                            "{{\"track\":{},\"frame\":{},\"timestamp\":{},\"values\":{}}}",
                            results.metadata.track_index,
                            results.metadata.frame_index,
                            results.metadata.timestamp,
                            values
                        );
                        ControlFlow::Continue(())
                    },
                    emotions: None,
                })?
            }
            GeometryExecutorBundle::Diffusion(_) => {
                return Err("regression bundle resolved to diffusion executor".into());
            }
        };
        let report = block_on(execution.wait_all())?;
        match report.state {
            ExecutionState::Complete => break,
            ExecutionState::Progress => {}
            ExecutionState::AwaitingInput => {
                return Err("regression executor is awaiting input after close".into());
            }
        }
    }
    eprintln!("regression callbacks: {callbacks}");
    Ok(())
}
