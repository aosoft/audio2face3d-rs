use crate::cli::async_util::block_on;
use audio2face3d::audio2face::bundle::GeometryExecutorBundleCreationParameters;
use audio2face3d::audio2face::diffusion::DiffusionGeometryExecutorCreationParameters;
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
    if model.kind() != ModelKind::Diffusion {
        return Err("expected a diffusion model".into());
    }
    let NetworkDocument::Geometry(network) = model.network() else {
        return Err("diffusion network is missing".into());
    };
    let GeometryParameters::Diffusion(parameters) = &network.params else {
        return Err("diffusion network parameters are missing".into());
    };
    let audio_parameters = match &network.audio_params {
        audio2face3d::common::GeometryAudioParameters::Diffusion(value) => value,
        _ => return Err("diffusion audio parameters are missing".into()),
    };
    let config = match model.parameters(0)? {
        audio2face3d::ModelParameters::Geometry(value) => value,
        _ => return Err("diffusion geometry config is missing".into()),
    };
    let frame_rate = FrameRate::new(30, 1)?;
    let resources = (0..tracks)
        .map(|_| {
            let audio = Arc::new(AudioAccumulator::new(audio_parameters.buffer_len, 0)?);
            let emotions = Arc::new(
                EmotionAccumulator::new(parameters.emotions.len(), parameters.num_frames_center)
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
        GeometryExecutorBundleCreationParameters::Diffusion(
            DiffusionGeometryExecutorCreationParameters {
                model_path: model_path.to_owned(),
                common,
                input_strength: config.input_strength,
                frame_rate,
                identity_index: 0,
                constant_noise: false,
                noise_seed: 0,
            },
        ),
    ))?;
    let audio = vec![0.0_f32; samples.unwrap_or(model.sample_rate())];
    for track in 0..tracks {
        bundle.audio_accumulator(track)?.accumulate(&audio)?;
        bundle.audio_accumulator(track)?.close()?;
    }
    let mut callbacks = 0_usize;
    let mut inference = 0_usize;
    loop {
        let execution = match &mut bundle {
            GeometryExecutorBundle::Diffusion(executor) => executor.execute(GeometryCallbacks {
                results: &mut |results| {
                    callbacks += 1;
                    let values = results.skin.map_or(0, |value| value.values.len())
                        + results.tongue.map_or(0, |value| value.values.len())
                        + results.jaw.map_or(0, |value| value.values.len())
                        + results.eyes.map_or(0, |value| value.values.len());
                    println!(
                        "{{\"track\":{},\"inference\":{},\"frame\":{},\"timestamp\":{},\"values\":{}}}",
                        results.metadata.track_index,
                        inference,
                        results.metadata.frame_index,
                        results.metadata.timestamp,
                        values
                    );
                    ControlFlow::Continue(())
                },
                emotions: None,
            })?,
            GeometryExecutorBundle::Regression(_) => {
                return Err("diffusion bundle resolved to regression executor".into())
            }
        };
        let report = block_on(execution.wait_all())?;
        match report.state {
            ExecutionState::Complete => break,
            ExecutionState::Progress => inference += 1,
            ExecutionState::AwaitingInput => {
                return Err("diffusion executor is awaiting input after close".into());
            }
        }
    }
    eprintln!("diffusion callbacks: {callbacks}");
    Ok(())
}
