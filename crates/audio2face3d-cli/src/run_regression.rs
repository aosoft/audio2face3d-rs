use audio2face3d::{
    Model, ModelKind, PipelineOptions, PipelineOutput, PipelineStatus, TensorRtPipeline,
};
use std::path::Path;

pub fn run(
    model_path: &Path,
    tracks: usize,
    samples: Option<usize>,
) -> Result<(), Box<dyn std::error::Error>> {
    let model = Model::load(model_path)?;
    if model.kind() != ModelKind::Regression {
        return Err("expected a regression model".into());
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
        let status = pipeline.execute(|metadata, output| {
            let PipelineOutput::Geometry(values) = output else {
                return false;
            };
            callbacks += 1;
            println!(
                "{{\"track\":{},\"frame\":{},\"timestamp\":{},\"values\":{}}}",
                metadata.track,
                metadata.frame,
                metadata.timestamp,
                values.len()
            );
            true
        })?;
        if !matches!(status, PipelineStatus::Executed { .. }) {
            if status != PipelineStatus::Complete {
                return Err(format!("pipeline stopped with {status:?}").into());
            }
            break;
        }
    }
    eprintln!("regression callbacks: {callbacks}");
    Ok(())
}
