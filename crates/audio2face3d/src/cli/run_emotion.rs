use crate::cli::async_util::block_on;
use audio2face3d::audio2emotion::classifier::{
    ClassifierEmotionExecutorCreationParameters, ClassifierEmotionExecutorFactory,
};
use audio2face3d::audio2emotion::{
    EmotionExecutor, EmotionExecutorCreationParameters, EmotionTrackResources, PostProcessData,
    PostProcessParams,
};
use audio2face3d::audio2x::{AudioAccumulator, ExecutionState, FrameRate};
use audio2face3d::common::NetworkDocument;
use audio2face3d::{Model, ModelKind, ModelParameters};
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::Arc;

pub fn run(
    model_path: &Path,
    tracks: usize,
    samples: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::load(model_path)?;
    if model.kind() != ModelKind::Emotion {
        return Err("expected an Audio2Emotion model".into());
    }
    let NetworkDocument::Emotion(network) = model.network() else {
        return Err("emotion network is missing".into());
    };
    let ModelParameters::Emotion(config) = model.parameters(0)? else {
        return Err("emotion post-process config is missing".into());
    };
    let frame_rate = FrameRate::new(30, 1)?;
    let post_process_data = PostProcessData {
        inference_emotion_length: network.emotions.len(),
        output_emotion_length: config.output_emotion_length,
        emotion_correspondence: network
            .emotions
            .iter()
            .map(|name| {
                config
                    .emotion_correspondence
                    .get(name)
                    .copied()
                    .ok_or_else(|| format!("emotion correspondence is missing for {name}"))
                    .and_then(|value| {
                        i32::try_from(value)
                            .map_err(|_| format!("emotion correspondence is out of range: {value}"))
                    })
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    let post_process_params = PostProcessParams {
        emotion_contrast: config.emotion_contrast,
        max_emotions: config.max_emotions,
        beginning_emotion: vec![0.0; config.output_emotion_length],
        preferred_emotion: config.preferred_emotion.clone(),
        live_blend_coefficient: config.live_blend_coef,
        enable_preferred_emotion: config.enable_preferred_emotion,
        preferred_emotion_strength: config.preferred_emotion_strength,
        live_transition_time: config.transition_smoothing,
        fixed_dt: config.fixed_dt,
        emotion_strength: config.emotion_strength,
    };
    let resources = (0..tracks)
        .map(|_| {
            Ok(EmotionTrackResources {
                audio: Arc::new(AudioAccumulator::new(60_000, 0)?),
            })
        })
        .collect::<Result<Vec<_>, audio2face3d::Error>>()?;
    let mut executor = block_on(ClassifierEmotionExecutorFactory::load(
        ClassifierEmotionExecutorCreationParameters {
            model_path: model_path.to_owned(),
            common: EmotionExecutorCreationParameters {
                tracks: resources,
                device_ordinal: 0,
            },
            input_strength: 1.0,
            buffer_length: 60_000,
            frame_rate,
            inferences_to_skip: 0,
            post_process_data,
            post_process_params,
            preferred_emotions: Vec::new(),
        },
    ))?;
    let audio = vec![0.0_f32; samples.unwrap_or(model.sample_rate())];
    for track in 0..tracks {
        executor.audio_accumulator(track)?.accumulate(&audio)?;
        executor.audio_accumulator(track)?.close()?;
    }
    let mut callbacks = 0_usize;
    loop {
        let mut copy_error = None;
        let execution = executor.execute(&mut |results| {
            callbacks += 1;
            let mut values = vec![0.0; results.emotions.values.len()];
            if let Err(error) = results.emotions.copy_to(&mut values) {
                copy_error = Some(error);
                return ControlFlow::Break(());
            }
            println!(
                "{{\"track\":{},\"frame\":{},\"timestamp\":{},\"emotions\":{:?}}}",
                results.metadata.track_index,
                results.metadata.frame_index,
                results.metadata.timestamp,
                values
            );
            ControlFlow::Continue(())
        })?;
        if let Some(error) = copy_error {
            return Err(error.into());
        }
        let report = block_on(execution.wait_all())?;
        match report.state {
            ExecutionState::Complete => break,
            ExecutionState::Progress => {}
            ExecutionState::AwaitingInput => {
                return Err("emotion executor is awaiting input after close".into());
            }
        }
    }
    eprintln!("emotion callbacks: {callbacks}");
    Ok(())
}
