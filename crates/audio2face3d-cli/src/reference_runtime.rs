use audio2face3d::animation::{BlendshapeSolverKind, GeometryInvalidationLayer};
use audio2face3d::common::{GeometryParameters, NetworkDocument};
use audio2face3d::{
    BlendshapeExecutorBundle, BlendshapeOutput, CallbackMetadata, ComposedGeometryExecutorBundle,
    GeometryExecutorBundle, GeometryFrame, GeometryObserver, InteractiveGeometryExecutorBundle,
    InteractivePipelineOptions, Model, ModelKind, ModelParameters, PipelineOptions, PipelineOutput,
    PipelineStatus, TensorRtPipeline, TrackParameters,
};
use audio2face3d_cli::reference::{
    ArtifactWriter, Case, FileProvenance, Producer, RecordMetadata, load_fixture, sha256_file,
};
use std::cell::Cell;
use std::io;
use std::path::Path;
use std::rc::Rc;

#[derive(Clone, Copy, Debug)]
pub enum Execution {
    Standard,
    InteractiveRandom,
    InteractiveAll,
    BlendshapeCpu,
    BlendshapeGpu,
}

pub struct CaptureRequest<'a> {
    pub model_path: &'a Path,
    pub fixture_root: &'a Path,
    pub output: &'a Path,
    pub execution: Execution,
    pub precision: &'a str,
    pub tracks: usize,
    pub seed: u64,
    pub selected_frame: Option<usize>,
}

pub fn capture(request: CaptureRequest<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let CaptureRequest {
        model_path,
        fixture_root,
        output,
        execution,
        precision,
        tracks,
        seed,
        selected_frame,
    } = request;
    if tracks == 0 {
        return Err("reference capture requires at least one track".into());
    }
    if !matches!(precision, "fp32" | "fp16") {
        return Err("reference precision must be fp32 or fp16".into());
    }
    let (fixture, samples) = load_fixture(fixture_root)?;
    let model = Model::load(model_path)?;
    if model.sample_rate() != fixture.sample_rate as usize {
        return Err(format!(
            "model sample rate {} differs from fixture rate {}",
            model.sample_rate(),
            fixture.sample_rate
        )
        .into());
    }
    let pipeline = kind_name(model.kind());
    let execution_name = match execution {
        Execution::Standard => "standard",
        Execution::InteractiveRandom => "interactive-random",
        Execution::InteractiveAll => "interactive-all",
        Execution::BlendshapeCpu => "blendshape-cpu",
        Execution::BlendshapeGpu => "blendshape-gpu",
    };
    if model.kind() == ModelKind::Emotion && !matches!(execution, Execution::Standard) {
        return Err("Audio2Emotion capture currently supports standard execution".into());
    }
    let fixture_path = fixture_root.join("samples.f32le");
    let mut writer = ArtifactWriter::create(
        output,
        Producer {
            implementation: "rust".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            revision: option_env!("GIT_COMMIT").map(str::to_owned),
        },
        Case {
            name: format!("{pipeline}-{execution_name}"),
            pipeline: pipeline.into(),
            execution: execution_name.into(),
            precision: precision.into(),
            seed,
            track_count: tracks,
        },
        FileProvenance {
            path: fixture_path.display().to_string(),
            sha256: fixture.samples_sha256,
        },
    )?;
    add_provenance(&mut writer, &model)?;
    match execution {
        Execution::Standard => capture_standard(&model, tracks, seed, &samples, &mut writer)?,
        Execution::InteractiveRandom => {
            capture_interactive(&model, seed, &samples, selected_frame, false, &mut writer)?
        }
        Execution::InteractiveAll => {
            capture_interactive(&model, seed, &samples, selected_frame, true, &mut writer)?
        }
        Execution::BlendshapeCpu => capture_blendshape(
            &model,
            tracks,
            seed,
            &samples,
            BlendshapeSolverKind::Cpu,
            &mut writer,
        )?,
        Execution::BlendshapeGpu => capture_blendshape(
            &model,
            tracks,
            seed,
            &samples,
            BlendshapeSolverKind::Gpu,
            &mut writer,
        )?,
    }
    let manifest = writer.finish()?;
    println!("artifact: {}", output.join("artifact.json").display());
    println!("records: {}", manifest.records.len());
    println!("data sha256: {}", manifest.data_sha256);
    Ok(())
}

fn capture_standard(
    model: &Model,
    tracks: usize,
    seed: u64,
    samples: &[f32],
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = options(tracks, seed);
    if model.kind() == ModelKind::Emotion {
        let mut pipeline = TensorRtPipeline::load(model, options)?;
        configure_tracks(&mut pipeline, model, tracks, samples)?;
        execute_to_completion(&mut pipeline, |metadata, output| {
            let PipelineOutput::Emotion(values) = output else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "emotion pipeline returned geometry",
                ));
            };
            writer.push_f32(
                record("postprocess", "emotion", metadata, values.len()),
                values,
            )
        })?;
    } else {
        let observed_frames = Rc::new(Cell::new(0_u64));
        let mut geometry = GeometryExecutorBundle::builder(model, options)?
            .observer(ReferenceGeometryObserver(Rc::clone(&observed_frames)))
            .build()?;
        configure_geometry(&mut geometry, model, tracks, samples)?;
        execute_geometry_to_completion(&mut geometry, writer, "postprocess")?;
        writer
            .manifest_mut()
            .counters
            .insert("geometry-observer-frames".into(), observed_frames.get());
    }
    Ok(())
}

fn capture_blendshape(
    model: &Model,
    tracks: usize,
    seed: u64,
    samples: &[f32],
    kind: BlendshapeSolverKind,
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let options = options(tracks, seed);
    let mut bundle = BlendshapeExecutorBundle::load(model, options, kind)?;
    for track in 0..tracks {
        configure_track_parameters_bundle(&mut bundle, model, track)?;
        bundle.accumulate_audio(track, samples)?;
        bundle.close_audio(track)?;
    }
    let mut callback_frames = vec![0_usize; tracks];
    loop {
        let mut callback_error = None;
        let status = bundle.execute(|mut metadata, output| {
            metadata.frame = callback_frames[metadata.track];
            callback_frames[metadata.track] += 1;
            let result = match output {
                BlendshapeOutput::Host {
                    skin_weights,
                    tongue_weights,
                } => push_blendshape(writer, metadata, skin_weights, tongue_weights),
                BlendshapeOutput::Device {
                    skin_weights,
                    tongue_weights,
                    stream,
                } => {
                    let mut skin = vec![0.0; skin_weights.map_or(0, |values| values.len())];
                    let mut tongue = vec![0.0; tongue_weights.map_or(0, |values| values.len())];
                    skin_weights
                        .map(|values| values.copy_to(&mut skin, stream))
                        .transpose()
                        .and_then(|_| {
                            tongue_weights
                                .map(|values| values.copy_to(&mut tongue, stream))
                                .transpose()
                        })
                        .map_err(io::Error::other)
                        .and_then(|_| push_blendshape(writer, metadata, &skin, &tongue))
                }
            };
            if let Err(error) = result {
                callback_error = Some(error);
                false
            } else {
                true
            }
        })?;
        if let Some(error) = callback_error {
            return Err(error.into());
        }
        if matches!(status, PipelineStatus::Complete) {
            break;
        }
        if matches!(status, PipelineStatus::Interrupted) {
            return Err("blendshape capture was interrupted".into());
        }
    }
    bundle.wait()?;
    Ok(())
}

fn capture_interactive(
    model: &Model,
    seed: u64,
    samples: &[f32],
    selected_frame: Option<usize>,
    all_frames: bool,
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bundle = InteractiveGeometryExecutorBundle::load(
        model,
        InteractivePipelineOptions {
            diffusion_seed: seed,
            ..InteractivePipelineOptions::default()
        },
    )?;
    bundle.audio().accumulate(samples)?;
    bundle.audio().close()?;
    bundle.emotions().close()?;
    let total = bundle.total_frames()?;
    let frame = selected_frame.unwrap_or(total / 2);
    if frame >= total {
        return Err(format!("selected frame {frame} is outside {total} frames").into());
    }
    if all_frames {
        let mut callback_error = None;
        bundle.compute_all_frames(|metadata, geometry| {
            if let Err(error) = push_interactive(writer, "interactive-all", metadata, geometry) {
                callback_error = Some(error);
                false
            } else {
                true
            }
        })?;
        if let Some(error) = callback_error {
            return Err(error.into());
        }
    } else {
        for (layer, invalidate) in [
            ("interactive-random", None),
            ("interactive-replay", None),
            (
                "interactive-invalidation",
                Some(GeometryInvalidationLayer::Skin),
            ),
        ] {
            if let Some(invalidation) = invalidate {
                bundle.invalidate(invalidation);
                *writer
                    .manifest_mut()
                    .counters
                    .entry("invalidation.skin".into())
                    .or_default() += 1;
            }
            let mut callback_error = None;
            bundle.compute_frame(frame, |metadata, geometry| {
                if let Err(error) = push_interactive(writer, layer, metadata, geometry) {
                    callback_error = Some(error);
                    false
                } else {
                    true
                }
            })?;
            if let Some(error) = callback_error {
                return Err(error.into());
            }
        }
    }
    writer
        .manifest_mut()
        .counters
        .insert("total_frames".into(), total as u64);
    Ok(())
}

fn execute_to_completion(
    pipeline: &mut TensorRtPipeline,
    mut callback: impl for<'a> FnMut(CallbackMetadata, PipelineOutput<'a>) -> io::Result<()>,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let mut callback_error = None;
        let status = pipeline.execute(|metadata, output| {
            if let Err(error) = callback(metadata, output) {
                callback_error = Some(error);
                false
            } else {
                true
            }
        })?;
        if let Some(error) = callback_error {
            return Err(error.into());
        }
        match status {
            PipelineStatus::Complete => return Ok(()),
            PipelineStatus::Interrupted => return Err("reference capture was interrupted".into()),
            PipelineStatus::AwaitingInput => {
                return Err("closed fixture unexpectedly needs more input".into());
            }
            PipelineStatus::Executed { .. } => {}
        }
    }
}

fn execute_geometry_to_completion(
    geometry: &mut ComposedGeometryExecutorBundle<GeometryExecutorBundle>,
    writer: &mut ArtifactWriter,
    layer: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut callback_frames = vec![0_usize; geometry.track_count()];
    loop {
        let mut callback_error = None;
        let status = geometry.execute(|mut metadata, frame| {
            metadata.frame = callback_frames[metadata.track];
            callback_frames[metadata.track] += 1;
            if let Err(error) = push_geometry(writer, layer, metadata, frame) {
                callback_error = Some(error);
                false
            } else {
                true
            }
        })?;
        if let Some(error) = callback_error {
            return Err(error.into());
        }
        match status {
            PipelineStatus::Complete => return Ok(()),
            PipelineStatus::Interrupted => return Err("geometry capture was interrupted".into()),
            PipelineStatus::AwaitingInput => {
                return Err("closed fixture unexpectedly needs more input".into());
            }
            PipelineStatus::Executed { .. } => {}
        }
    }
}

fn push_geometry(
    writer: &mut ArtifactWriter,
    layer: &str,
    mut metadata: CallbackMetadata,
    frame: &GeometryFrame,
) -> io::Result<()> {
    // The original public geometry callback has no inference index. Keep it
    // out of final callback records; low-level inference records carry it.
    metadata.inference = None;
    for (component, values) in [
        ("skin", frame.skin.as_slice()),
        ("tongue", frame.tongue.as_slice()),
        ("jaw", frame.jaw_transform.as_slice()),
        ("eyes-right", frame.eyes_rotation.right.as_slice()),
        ("eyes-left", frame.eyes_rotation.left.as_slice()),
    ] {
        writer.push_f32(record(layer, component, metadata, values.len()), values)?;
    }
    Ok(())
}

fn push_blendshape(
    writer: &mut ArtifactWriter,
    metadata: CallbackMetadata,
    skin: &[f32],
    tongue: &[f32],
) -> io::Result<()> {
    let mut weights = Vec::with_capacity(skin.len() + tongue.len());
    weights.extend_from_slice(skin);
    weights.extend_from_slice(tongue);
    writer.push_f32(
        record("blendshape", "weights", metadata, weights.len()),
        &weights,
    )
}

fn push_interactive(
    writer: &mut ArtifactWriter,
    layer: &str,
    metadata: audio2face3d::animation::InteractiveGeometryMetadata,
    geometry: &audio2face3d::animation::RegressionGeometry,
) -> io::Result<()> {
    let metadata = CallbackMetadata {
        kind: ModelKind::Regression,
        track: 0,
        inference: None,
        frame: metadata.frame,
        timestamp: metadata.timestamp,
        next_timestamp: metadata.next_timestamp,
    };
    let frame = GeometryFrame::from(geometry.clone());
    push_geometry(writer, layer, metadata, &frame)
}

fn record(layer: &str, component: &str, metadata: CallbackMetadata, len: usize) -> RecordMetadata {
    RecordMetadata {
        layer: layer.into(),
        component: component.into(),
        track: metadata.track,
        frame: Some(metadata.frame),
        inference: metadata.inference,
        timestamp: Some(metadata.timestamp),
        next_timestamp: Some(metadata.next_timestamp),
        shape: vec![len],
    }
}

fn configure_tracks(
    pipeline: &mut TensorRtPipeline,
    model: &Model,
    tracks: usize,
    samples: &[f32],
) -> Result<(), audio2face3d::Error> {
    for track in 0..tracks {
        configure_track_parameters(pipeline, model, track)?;
        pipeline.accumulate_audio(track, samples)?;
        pipeline.close_audio(track)?;
    }
    Ok(())
}

fn configure_geometry(
    geometry: &mut ComposedGeometryExecutorBundle<GeometryExecutorBundle>,
    model: &Model,
    tracks: usize,
    samples: &[f32],
) -> Result<(), audio2face3d::Error> {
    for track in 0..tracks {
        if let Some(parameters) = geometry_parameters(model, track) {
            geometry.set_track_parameters(track, parameters)?;
        }
        geometry.accumulate_audio(track, samples)?;
        geometry.close_audio(track)?;
    }
    Ok(())
}

struct ReferenceGeometryObserver(Rc<Cell<u64>>);

impl GeometryObserver for ReferenceGeometryObserver {
    fn observe(&mut self, _: CallbackMetadata, _: &GeometryFrame) -> audio2face3d::Result<()> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
}

fn configure_track_parameters_bundle(
    bundle: &mut BlendshapeExecutorBundle,
    model: &Model,
    track: usize,
) -> Result<(), audio2face3d::Error> {
    if let Some(parameters) = geometry_parameters(model, track) {
        bundle.set_track_parameters(track, parameters)?;
    }
    Ok(())
}

fn configure_track_parameters(
    pipeline: &mut TensorRtPipeline,
    model: &Model,
    track: usize,
) -> Result<(), audio2face3d::Error> {
    if let Some(parameters) = geometry_parameters(model, track) {
        pipeline.set_track_parameters(track, parameters)?;
    }
    Ok(())
}

fn geometry_parameters(model: &Model, _track: usize) -> Option<TrackParameters> {
    let NetworkDocument::Geometry(network) = model.network() else {
        return None;
    };
    let input_strength = match model.parameters(0).ok()? {
        ModelParameters::Geometry(parameters) => parameters.input_strength,
        ModelParameters::Emotion(_) => return None,
    };
    match &network.params {
        GeometryParameters::Regression(parameters) => Some(TrackParameters::Regression {
            input_strength,
            implicit_emotion: vec![0.0; parameters.implicit_emotion_len],
        }),
        GeometryParameters::Diffusion(_) => Some(TrackParameters::Diffusion {
            input_strength,
            identity_index: 0,
        }),
    }
}

fn options(tracks: usize, seed: u64) -> PipelineOptions {
    PipelineOptions {
        track_count: tracks,
        diffusion_seed: seed,
        ..PipelineOptions::default()
    }
}

fn add_provenance(writer: &mut ArtifactWriter, model: &Model) -> io::Result<()> {
    for (name, path) in [
        ("model-descriptor", model.descriptor_path()),
        ("engine", model.engine_path()),
    ] {
        writer.manifest_mut().model_files.insert(
            name.into(),
            FileProvenance {
                path: path.display().to_string(),
                sha256: sha256_file(path)?,
            },
        );
    }
    writer
        .manifest_mut()
        .environment
        .insert("os".into(), std::env::consts::OS.into());
    writer
        .manifest_mut()
        .environment
        .insert("arch".into(), std::env::consts::ARCH.into());
    writer.manifest_mut().environment.insert(
        "runtime".into(),
        audio2face3d::RuntimeDiscovery::discover().diagnostic(),
    );
    Ok(())
}

fn kind_name(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Regression => "regression",
        ModelKind::Diffusion => "diffusion",
        ModelKind::Emotion => "emotion",
    }
}
