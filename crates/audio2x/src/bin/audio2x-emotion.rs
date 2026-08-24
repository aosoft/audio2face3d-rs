use audio2x::{
    Audio2xModel, ModelKind, PipelineOptions, PipelineOutput, PipelineStatus, TensorRtPipeline,
};
use std::path::PathBuf;

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let (model_path, tracks, samples) = arguments()?;
    let model = Audio2xModel::load(model_path)?;
    if model.kind() != ModelKind::Emotion {
        return Err("expected an Audio2Emotion model".into());
    }
    let mut pipeline = TensorRtPipeline::load(
        &model,
        PipelineOptions {
            track_count: tracks,
            ..PipelineOptions::default()
        },
    )?;
    let audio = vec![0.0_f32; samples.unwrap_or(model.sample_rate())];
    for track in 0..tracks {
        pipeline.accumulate_audio(track, &audio)?;
        pipeline.close_audio(track)?;
    }
    let mut callbacks = 0_usize;
    loop {
        match pipeline.execute(|metadata, output| {
            let PipelineOutput::Emotion(values) = output else {
                return false;
            };
            callbacks += 1;
            println!(
                "{{\"track\":{},\"frame\":{},\"timestamp\":{},\"emotions\":{:?}}}",
                metadata.track, metadata.frame, metadata.timestamp, values
            );
            true
        })? {
            PipelineStatus::Executed { .. } => {}
            PipelineStatus::Complete => break,
            status => return Err(format!("pipeline stopped with {status:?}").into()),
        }
    }
    eprintln!("emotion callbacks: {callbacks}");
    Ok(())
}

fn arguments() -> Result<(PathBuf, usize, Option<usize>), String> {
    let mut values = std::env::args_os().skip(1);
    let model = values
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| "usage: audio2x-emotion <model.json> [tracks] [samples]".to_owned())?;
    let tracks = values
        .next()
        .map(|value| value.to_string_lossy().parse())
        .transpose()
        .map_err(|_| "tracks must be an integer")?
        .unwrap_or(1);
    let samples = values
        .next()
        .map(|value| value.to_string_lossy().parse())
        .transpose()
        .map_err(|_| "samples must be an integer")?;
    Ok((model, tracks, samples))
}
