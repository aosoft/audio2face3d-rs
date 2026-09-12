use crate::async_util::block_on;
use audio2face3d::animation::{
    BlendshapeData, BlendshapeSolverKind, GeometryModelData, GpuBlendshapeSolver,
    InteractiveGpuBlendshapeLayer, RegressionGeometry,
};
use audio2face3d::audio2emotion::classifier::{
    ClassifierEmotionExecutorCreationParameters, ClassifierEmotionExecutorFactory,
};
use audio2face3d::audio2emotion::{
    EmotionExecutor, EmotionExecutorCreationParameters, EmotionTrackResources, PostProcessData,
    PostProcessParams,
};
use audio2face3d::audio2face::diffusion::DiffusionGeometryExecutorCreationParameters;
use audio2face3d::audio2face::diffusion::DiffusionGeometryInteractiveExecutorCreationParameters;
use audio2face3d::audio2face::regression::RegressionGeometryExecutorCreationParameters;
use audio2face3d::audio2face::regression::RegressionGeometryInteractiveExecutorCreationParameters;
use audio2face3d::audio2face::{
    AnimatorTeethParams, BlendshapeInteractiveExecutor, BlendshapeInvalidationLayer,
    BlendshapeSolveComponentParameters, BlendshapeSolveExecutorCreationParameters,
    BlendshapeSolverConfigView, BlendshapeSolverDataView, BlendshapeSolverParams,
    DeviceBlendshapeSolveExecutorCreationParameters, DeviceBlendshapeSolveInteractiveExecutor,
    GeometryExecutionOption, GeometryExecutorBundle as FacadeGeometryExecutorBundle,
    GeometryExecutorBundleCreationParameters, GeometryExecutorBundleFactory,
    GeometryInteractiveExecutor, GeometryInteractiveExecutorCreationParameters,
    GeometryInvalidationLayer, GeometryResults, GeometryTrackResources,
    HostBlendshapeSolveExecutorCreationParameters, InteractiveGeometryBundleCreationParameters,
    InteractiveGeometryExecutorBundle as FacadeInteractiveGeometryExecutorBundle,
    InteractiveGeometryExecutorBundleFactory, create_animator_teeth,
};
use audio2face3d::audio2x::{
    AudioAccumulator, DeviceComponentResults, EmotionAccumulator, ExecutionState, FrameRate,
    InteractiveExecutionReport, InteractiveExecutor,
};
use audio2face3d::common::{
    BlendshapeConfig, GeometryParameters, NetworkDocument, load_blendshape_config,
};
use audio2face3d::cuda::GpuDevice;
use audio2face3d::{Model, ModelKind, ModelParameters};

#[derive(Clone, Copy)]
struct CallbackMetadata {
    track: usize,
    inference: Option<usize>,
    frame: usize,
    timestamp: i64,
    next_timestamp: i64,
}

struct GeometryFrame {
    skin: Vec<f32>,
    tongue: Vec<f32>,
    jaw_transform: [f32; 16],
    eyes_rotation: audio2face3d::animation::EyesRotation,
}
use audio2face3d_cli::reference::{
    ArtifactWriter, Case, FileProvenance, Producer, RecordMetadata, load_fixture, sha256_file,
};
use std::io;
use std::ops::ControlFlow;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub enum Execution {
    Standard,
    InteractiveRandom,
    InteractiveAll,
    InteractiveBlendshapeRandom,
    InteractiveBlendshapeAll,
    BlendshapeCpu,
    BlendshapeGpu,
    TeethStandalone,
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
        Execution::InteractiveBlendshapeRandom => "interactive-blendshape-random",
        Execution::InteractiveBlendshapeAll => "interactive-blendshape-all",
        Execution::BlendshapeCpu => "blendshape-cpu",
        Execution::BlendshapeGpu => "blendshape-gpu",
        Execution::TeethStandalone => "teeth-standalone",
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
            bytes: Some((fixture.sample_count * size_of::<f32>()) as u64),
            license: Some(fixture.license),
            revision: fixture.source.and_then(|source| source.revision),
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
        Execution::InteractiveBlendshapeRandom => capture_interactive_blendshape(
            &model,
            seed,
            &samples,
            selected_frame,
            false,
            &mut writer,
        )?,
        Execution::InteractiveBlendshapeAll => capture_interactive_blendshape(
            &model,
            seed,
            &samples,
            selected_frame,
            true,
            &mut writer,
        )?,
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
        Execution::TeethStandalone => capture_teeth(&model, tracks, &mut writer)?,
    }
    let manifest = writer.finish()?;
    println!("artifact: {}", output.join("artifact.json").display());
    println!("records: {}", manifest.records.len());
    println!("data sha256: {}", manifest.data_sha256);
    Ok(())
}

fn capture_teeth(
    model: &Model,
    tracks: usize,
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let data = match model.kind() {
        ModelKind::Regression => GeometryModelData::load_regression(model.model_data_path(0)?)?,
        ModelKind::Diffusion => GeometryModelData::load_diffusion(model.model_data_path(0)?)?,
        ModelKind::Emotion => {
            return Err("standalone teeth capture requires a geometry model".into());
        }
    };
    let default = match model.parameters(0)? {
        ModelParameters::Geometry(value) => AnimatorTeethParams {
            lower_teeth_strength: value.lower_teeth_strength,
            lower_teeth_height_offset: value.lower_teeth_height_offset,
            lower_teeth_depth_offset: value.lower_teeth_depth_offset,
        },
        ModelParameters::Emotion(_) => {
            return Err("geometry model does not expose teeth parameters".into());
        }
    };
    let mut animator = create_animator_teeth(default, data.jaw_neutral_pose.clone())?;
    let deltas = teeth_deltas(tracks, data.jaw_neutral_pose.len());
    for track in 0..tracks {
        animator.set_parameters(teeth_parameters(default, track))?;
        let start = track * data.jaw_neutral_pose.len();
        let transform = animator.compute(&deltas[start..start + data.jaw_neutral_pose.len()])?;
        writer.push_f32(
            RecordMetadata {
                layer: "standalone-teeth".into(),
                component: "jaw".into(),
                track,
                frame: Some(0),
                inference: None,
                timestamp: Some(0),
                next_timestamp: Some(0),
                shape: vec![transform.len()],
            },
            &transform,
        )?;
    }
    writer
        .manifest_mut()
        .counters
        .insert("teeth-tracks".into(), tracks as u64);
    Ok(())
}

fn teeth_parameters(default: AnimatorTeethParams, track: usize) -> AnimatorTeethParams {
    match track % 3 {
        0 => default,
        1 => AnimatorTeethParams {
            lower_teeth_strength: 0.5,
            lower_teeth_height_offset: 0.25,
            lower_teeth_depth_offset: -0.5,
        },
        _ => AnimatorTeethParams {
            lower_teeth_strength: 2.0,
            lower_teeth_height_offset: -3.0,
            lower_teeth_depth_offset: 3.0,
        },
    }
}

fn teeth_deltas(tracks: usize, size: usize) -> Vec<f32> {
    let mut values = vec![0.0; size * tracks];
    for track in 0..tracks {
        for index in 0..size {
            values[track * size + index] = ((track + 1) * (index % 7 + 1)) as f32 * 0.001;
        }
    }
    values
}

fn capture_standard(
    model: &Model,
    tracks: usize,
    seed: u64,
    samples: &[f32],
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    if model.kind() == ModelKind::Emotion {
        capture_standard_emotion(model, tracks, samples, writer)?;
    } else {
        let mut geometry = create_facade_geometry(model, tracks, seed)?;
        for track in 0..tracks {
            geometry.audio_accumulator(track)?.accumulate(samples)?;
            geometry.audio_accumulator(track)?.close()?;
        }
        execute_facade_geometry(&mut geometry, writer, "postprocess", &mut vec![0; tracks])?;
    }
    Ok(())
}

fn capture_standard_emotion(
    model: &Model,
    tracks: usize,
    samples: &[f32],
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let audio_length = 60_000;
    let (network, config) = match (model.network(), model.parameters(0)?) {
        (NetworkDocument::Emotion(network), ModelParameters::Emotion(config)) => (network, config),
        _ => return Err("emotion model schema is unavailable".into()),
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
                audio: std::sync::Arc::new(AudioAccumulator::new(audio_length, 0)?),
            })
        })
        .collect::<Result<Vec<_>, audio2face3d::Error>>()?;
    let mut executor = block_on(ClassifierEmotionExecutorFactory::load(
        ClassifierEmotionExecutorCreationParameters {
            model_path: model.descriptor_path().to_owned(),
            common: EmotionExecutorCreationParameters {
                tracks: resources,
                device_ordinal: 0,
            },
            input_strength: 1.0,
            buffer_length: audio_length,
            frame_rate,
            inferences_to_skip: 0,
            post_process_data,
            post_process_params,
            preferred_emotions: Vec::new(),
        },
    ))?;
    for track in 0..tracks {
        executor.audio_accumulator(track)?.accumulate(samples)?;
        executor.audio_accumulator(track)?.close()?;
    }
    let mut callback_frames = vec![0; tracks];
    loop {
        let mut callback_error = None;
        let execution = executor.execute(&mut |results| {
            let frame = callback_frames[results.metadata.track_index];
            callback_frames[results.metadata.track_index] += 1;
            let mut values = vec![0.0; results.emotions.values.len()];
            if let Err(error) = results.emotions.copy_to(&mut values) {
                callback_error = Some(io::Error::other(error));
                return ControlFlow::Break(());
            }
            if let Err(error) = writer.push_f32(
                record(
                    "postprocess",
                    "emotion",
                    CallbackMetadata {
                        track: results.metadata.track_index,
                        inference: None,
                        frame,
                        timestamp: results.metadata.timestamp,
                        next_timestamp: results.metadata.next_timestamp,
                    },
                    results.emotions.values.len(),
                ),
                &values,
            ) {
                callback_error = Some(error);
                return ControlFlow::Break(());
            }
            ControlFlow::Continue(())
        })?;
        if let Some(error) = callback_error {
            return Err(error.into());
        }
        let report = block_on(execution.wait_all())?;
        match report.state {
            ExecutionState::Complete => break,
            ExecutionState::Progress => {}
            ExecutionState::AwaitingInput => {
                return Err("emotion executor is awaiting input".into());
            }
        }
    }
    Ok(())
}

fn create_facade_geometry(
    model: &Model,
    tracks: usize,
    seed: u64,
) -> Result<FacadeGeometryExecutorBundle, Box<dyn std::error::Error>> {
    let NetworkDocument::Geometry(network) = model.network() else {
        return Err("geometry model schema is unavailable".into());
    };
    let config = match model.parameters(0)? {
        ModelParameters::Geometry(value) => value,
        _ => return Err("geometry config is unavailable".into()),
    };
    let frame_rate = FrameRate::new(30, 1)?;
    let resources = match &network.params {
        GeometryParameters::Regression(parameters) => (0..tracks)
            .map(|_| {
                let audio = std::sync::Arc::new(AudioAccumulator::new(
                    match &network.audio_params {
                        audio2face3d::common::GeometryAudioParameters::Regression(value) => {
                            value.buffer_len
                        }
                        _ => {
                            return Err(audio2face3d::Error::InvalidSchema(
                                "regression audio schema is unavailable".into(),
                            ));
                        }
                    },
                    0,
                )?);
                let emotions = std::sync::Arc::new(
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
            .collect::<Result<Vec<_>, audio2face3d::Error>>()?,
        GeometryParameters::Diffusion(parameters) => (0..tracks)
            .map(|_| {
                let audio_len = match &network.audio_params {
                    audio2face3d::common::GeometryAudioParameters::Diffusion(value) => {
                        value.buffer_len
                    }
                    _ => {
                        return Err(audio2face3d::Error::InvalidSchema(
                            "diffusion audio schema is unavailable".into(),
                        ));
                    }
                };
                let audio = std::sync::Arc::new(AudioAccumulator::new(audio_len, 0)?);
                let emotions = std::sync::Arc::new(
                    EmotionAccumulator::new(parameters.emotions.len(), 300)
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
            .collect::<Result<Vec<_>, audio2face3d::Error>>()?,
    };
    let common = audio2face3d::audio2face::GeometryExecutorCreationParameters {
        tracks: resources,
        device_ordinal: 0,
        execution_option: GeometryExecutionOption::ALL,
    };
    let parameters = match &network.params {
        GeometryParameters::Regression(_) => GeometryExecutorBundleCreationParameters::Regression(
            RegressionGeometryExecutorCreationParameters {
                model_path: model.descriptor_path().to_owned(),
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
        GeometryParameters::Diffusion(_) => GeometryExecutorBundleCreationParameters::Diffusion(
            DiffusionGeometryExecutorCreationParameters {
                model_path: model.descriptor_path().to_owned(),
                common,
                input_strength: config.input_strength,
                frame_rate,
                identity_index: 0,
                constant_noise: false,
                noise_seed: seed,
            },
        ),
    };
    let mut bundle = block_on(GeometryExecutorBundleFactory::load(parameters))?;
    match (&mut bundle, &network.params) {
        (FacadeGeometryExecutorBundle::Regression(executor), GeometryParameters::Regression(_)) => {
            executor.set_input_strength(config.input_strength)?;
        }
        (FacadeGeometryExecutorBundle::Diffusion(executor), GeometryParameters::Diffusion(_)) => {
            executor.set_input_strength(config.input_strength)?;
            executor.set_identity_index(0)?;
        }
        _ => return Err("geometry bundle kind differs from the model".into()),
    }
    Ok(bundle)
}

fn execute_facade_geometry(
    geometry: &mut FacadeGeometryExecutorBundle,
    writer: &mut ArtifactWriter,
    layer: &str,
    _callback_frames: &mut [usize],
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let mut callback_error = None;
        let mut callback = |results: GeometryResults<'_>| {
            if callback_error.is_some() {
                return ControlFlow::Break(());
            }
            if let Err(error) = write_facade_geometry_result(writer, layer, results) {
                callback_error = Some(error);
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        };
        let execution = match geometry {
            FacadeGeometryExecutorBundle::Regression(executor) => {
                audio2face3d::audio2face::GeometryExecutor::execute(
                    executor,
                    audio2face3d::audio2face::GeometryCallbacks {
                        results: &mut callback,
                        emotions: None,
                    },
                )?
            }
            FacadeGeometryExecutorBundle::Diffusion(executor) => {
                audio2face3d::audio2face::GeometryExecutor::execute(
                    executor,
                    audio2face3d::audio2face::GeometryCallbacks {
                        results: &mut callback,
                        emotions: None,
                    },
                )?
            }
        };
        if let Some(error) = callback_error {
            return Err(error.into());
        }
        match block_on(execution.wait_all())?.state {
            ExecutionState::Complete => return Ok(()),
            ExecutionState::Progress => {}
            ExecutionState::AwaitingInput => {
                return Err("geometry executor is awaiting input".into());
            }
        }
    }
}

fn write_facade_geometry_result(
    writer: &mut ArtifactWriter,
    layer: &str,
    results: GeometryResults<'_>,
) -> io::Result<()> {
    let metadata = results.metadata;
    let frame = copy_facade_geometry_result(results)?;
    push_geometry(
        writer,
        layer,
        CallbackMetadata {
            track: metadata.track_index,
            inference: None,
            frame: metadata.frame_index,
            timestamp: metadata.timestamp,
            next_timestamp: metadata.next_timestamp,
        },
        &frame,
    )
}

fn copy_facade_geometry_result(results: GeometryResults<'_>) -> io::Result<GeometryFrame> {
    let copy = |component: Option<DeviceComponentResults<'_>>| {
        component.map_or_else(
            || Ok::<Vec<f32>, io::Error>(Vec::new()),
            |component| {
                let mut values = vec![0.0; component.values.len()];
                component.copy_to(&mut values).map_err(io::Error::other)?;
                Ok(values)
            },
        )
    };
    let skin = copy(results.skin)?;
    let tongue = copy(results.tongue)?;
    let jaw = copy(results.jaw)?;
    let eyes = copy(results.eyes)?;
    if jaw.len() != 16 || eyes.len() != 6 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unexpected geometry component sizes: jaw={}, eyes={}",
                jaw.len(),
                eyes.len()
            ),
        ));
    }
    Ok(GeometryFrame {
        skin,
        tongue,
        jaw_transform: jaw.try_into().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "jaw transform must contain 16 values",
            )
        })?,
        eyes_rotation: audio2face3d::animation::EyesRotation {
            right: eyes[..3].try_into().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "right eye rotation must contain 3 values",
                )
            })?,
            left: eyes[3..].try_into().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "left eye rotation must contain 3 values",
                )
            })?,
        },
    })
}

fn capture_blendshape(
    model: &Model,
    tracks: usize,
    seed: u64,
    samples: &[f32],
    kind: BlendshapeSolverKind,
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let geometry = create_facade_geometry(model, tracks, seed)?;
    let skin = load_reference_blendshape_component(model, "skin")?;
    let tongue = load_reference_blendshape_component(model, "tongue")?;
    let skin_names = skin
        .as_ref()
        .map(|component| {
            component
                .data
                .pose_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let tongue_names = tongue
        .as_ref()
        .map(|component| {
            component
                .data
                .pose_names
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let components = BlendshapeSolveExecutorCreationParameters {
        skin: skin
            .as_ref()
            .map(|component| component.creation_parameters(&skin_names)),
        tongue: tongue
            .as_ref()
            .map(|component| component.creation_parameters(&tongue_names)),
    };
    match kind {
        BlendshapeSolverKind::Cpu => {
            let mut executor = geometry
                .try_into_host_blendshape(HostBlendshapeSolveExecutorCreationParameters {
                    components,
                    job_runner: None,
                })
                .map_err(|failure| failure.error)?;
            for track in 0..tracks {
                executor.audio_accumulator(track)?.accumulate(samples)?;
                executor.audio_accumulator(track)?.close()?;
            }
            capture_host_blendshape(&mut executor, writer)?;
        }
        BlendshapeSolverKind::Gpu => {
            let mut executor = geometry
                .try_into_device_blendshape(DeviceBlendshapeSolveExecutorCreationParameters {
                    components,
                })
                .map_err(|failure| failure.error)?;
            for track in 0..tracks {
                executor.audio_accumulator(track)?.accumulate(samples)?;
                executor.audio_accumulator(track)?.close()?;
            }
            capture_device_blendshape(&mut executor, writer)?;
        }
    }
    Ok(())
}

struct ReferenceBlendshapeComponent {
    data: BlendshapeData,
    config: BlendshapeConfig,
}

impl ReferenceBlendshapeComponent {
    fn creation_parameters<'a>(
        &'a self,
        pose_names: &'a [&'a str],
    ) -> BlendshapeSolveComponentParameters<'a> {
        BlendshapeSolveComponentParameters {
            params: BlendshapeSolverParams {
                l1_regularization: self.config.l1_regularization,
                l2_regularization: self.config.l2_regularization,
                symmetry_regularization: self.config.symmetry_regularization,
                temporal_regularization: self.config.temporal_regularization,
                template_bounding_box_size: self.config.template_bb_size,
                tolerance: self.config.tolerance,
            },
            config: BlendshapeSolverConfigView {
                active_poses: &self.config.active_poses,
                cancel_poses: &self.config.cancel_poses,
                symmetry_poses: &self.config.symmetry_poses,
                multipliers: &self.config.multipliers,
                offsets: &self.config.offsets,
            },
            data: BlendshapeSolverDataView {
                neutral_pose: &self.data.neutral_pose,
                delta_poses: &self.data.delta_poses,
                pose_mask: self.data.pose_mask.as_deref(),
                pose_names,
            },
        }
    }
}

fn load_reference_blendshape_component(
    model: &Model,
    name: &str,
) -> Result<Option<ReferenceBlendshapeComponent>, Box<dyn std::error::Error>> {
    let paths = model.blendshape_paths(0)?;
    let Some(paths) = paths.get(name) else {
        return Ok(None);
    };
    Ok(Some(ReferenceBlendshapeComponent {
        data: BlendshapeData::load_npz(&paths.data)?,
        config: load_blendshape_config(&paths.config)?.blendshape_params,
    }))
}

fn capture_host_blendshape(
    executor: &mut audio2face3d::audio2face::HostBlendshapeSolveExecutor,
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::{Arc, Mutex};

    loop {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let callback_results = Arc::clone(&captured);
        let execution = executor.execute(Arc::new(move |event| {
            callback_results
                .lock()
                .expect("reference BlendShape callback mutex poisoned")
                .push(event.map(|result| (result.metadata, result.weights.to_vec())));
        }))?;
        let report = block_on(execution.wait_all())?;
        let mut captured = captured
            .lock()
            .map_err(|_| "reference BlendShape callback mutex poisoned")?;
        captured.sort_by_key(|event| {
            event
                .as_ref()
                .map(|(metadata, _)| (metadata.frame_index, metadata.track_index))
                .unwrap_or((usize::MAX, usize::MAX))
        });
        for event in captured.drain(..) {
            let (metadata, weights) = event?;
            push_blendshape(writer, facade_callback_metadata(metadata), &weights, &[])?;
        }
        match report.state {
            ExecutionState::Complete => return Ok(()),
            ExecutionState::Progress => {}
            ExecutionState::AwaitingInput => {
                return Err("host BlendShape executor is awaiting input".into());
            }
        }
    }
}

fn capture_device_blendshape(
    executor: &mut audio2face3d::audio2face::DeviceBlendshapeSolveExecutor,
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    loop {
        let mut callback_error = None;
        let execution = executor.execute(&mut |result| {
            let mut weights = vec![0.0; result.weights.values.len()];
            let captured = result
                .weights
                .copy_to(&mut weights)
                .map_err(io::Error::other)
                .and_then(|_| {
                    push_blendshape(
                        writer,
                        facade_callback_metadata(result.metadata),
                        &weights,
                        &[],
                    )
                });
            if let Err(error) = captured {
                callback_error = Some(error);
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })?;
        if let Some(error) = callback_error {
            return Err(error.into());
        }
        let report = block_on(execution.wait_all())?;
        match report.state {
            ExecutionState::Complete => return Ok(()),
            ExecutionState::Progress => {}
            ExecutionState::AwaitingInput => {
                return Err("device BlendShape executor is awaiting input".into());
            }
        }
    }
}

fn facade_callback_metadata(metadata: audio2face3d::audio2x::CallbackMetadata) -> CallbackMetadata {
    CallbackMetadata {
        track: metadata.track_index,
        inference: None,
        frame: metadata.frame_index,
        timestamp: metadata.timestamp,
        next_timestamp: metadata.next_timestamp,
    }
}

fn capture_interactive(
    model: &Model,
    seed: u64,
    samples: &[f32],
    selected_frame: Option<usize>,
    all_frames: bool,
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut bundle = create_interactive_facade_geometry(model, seed, samples)?;
    let total = match &bundle {
        FacadeInteractiveGeometryExecutorBundle::Regression(executor) => {
            InteractiveExecutor::total_frame_count(executor.as_ref())?
        }
        FacadeInteractiveGeometryExecutorBundle::Diffusion(executor) => {
            InteractiveExecutor::total_frame_count(executor.as_ref())?
        }
    };
    let frame = selected_frame.unwrap_or(total / 2);
    if frame >= total {
        return Err(format!("selected frame {frame} is outside {total} frames").into());
    }
    if all_frames {
        let report = compute_interactive_geometry(&mut bundle, None, "interactive-all", writer)?;
        ensure_interactive_complete(report)?;
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
                match &mut bundle {
                    FacadeInteractiveGeometryExecutorBundle::Regression(executor) => {
                        executor.invalidate_geometry(invalidation)?;
                    }
                    FacadeInteractiveGeometryExecutorBundle::Diffusion(executor) => {
                        executor.invalidate_geometry(invalidation)?;
                    }
                }
                *writer
                    .manifest_mut()
                    .counters
                    .entry("invalidation.skin".into())
                    .or_default() += 1;
            }
            let report = compute_interactive_geometry(&mut bundle, Some(frame), layer, writer)?;
            ensure_interactive_complete(report)?;
        }
    }
    writer
        .manifest_mut()
        .counters
        .insert("total_frames".into(), total as u64);
    Ok(())
}

fn create_interactive_facade_geometry(
    model: &Model,
    seed: u64,
    samples: &[f32],
) -> Result<FacadeInteractiveGeometryExecutorBundle, Box<dyn std::error::Error>> {
    let NetworkDocument::Geometry(network) = model.network() else {
        return Err("geometry model schema is unavailable".into());
    };
    let config = match model.parameters(0)? {
        ModelParameters::Geometry(value) => value,
        _ => return Err("geometry config is unavailable".into()),
    };
    let (audio_size, emotion_size, default_emotion) = match (&network.params, &network.audio_params)
    {
        (
            GeometryParameters::Regression(parameters),
            audio2face3d::common::GeometryAudioParameters::Regression(audio),
        ) => (
            audio.buffer_len,
            parameters.explicit_emotions.len(),
            parameters.default_emotion.as_slice(),
        ),
        (
            GeometryParameters::Diffusion(parameters),
            audio2face3d::common::GeometryAudioParameters::Diffusion(audio),
        ) => (
            audio.buffer_len,
            parameters.emotions.len(),
            parameters.default_emotion.as_slice(),
        ),
        _ => return Err("geometry model schema is inconsistent".into()),
    };
    let audio = std::sync::Arc::new(AudioAccumulator::new(audio_size, 0)?);
    audio.accumulate(samples)?;
    audio.close()?;
    let emotions = std::sync::Arc::new(
        EmotionAccumulator::new(emotion_size, 300)
            .map_err(|error| audio2face3d::Error::InvalidSchema(error.to_string()))?,
    );
    emotions
        .accumulate(0, default_emotion)
        .map_err(|error| audio2face3d::Error::InvalidSchema(error.to_string()))?;
    emotions
        .close()
        .map_err(|error| audio2face3d::Error::InvalidSchema(error.to_string()))?;
    let common = GeometryInteractiveExecutorCreationParameters {
        audio,
        emotions,
        device_ordinal: 0,
        execution_option: GeometryExecutionOption::ALL,
    };
    let frame_rate = FrameRate::new(30, 1)?;
    let parameters = match &network.params {
        GeometryParameters::Regression(_) => {
            InteractiveGeometryBundleCreationParameters::Regression(
                RegressionGeometryInteractiveExecutorCreationParameters {
                    model_path: model.descriptor_path().to_owned(),
                    common,
                    input_strength: config.input_strength,
                    frame_rate,
                    source_emotion_shot: config.source_shot.clone(),
                    source_emotion_frame: config
                        .source_frame
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(0),
                    batch_size: 1,
                },
            )
        }
        GeometryParameters::Diffusion(_) => InteractiveGeometryBundleCreationParameters::Diffusion(
            DiffusionGeometryInteractiveExecutorCreationParameters {
                model_path: model.descriptor_path().to_owned(),
                common,
                input_strength: config.input_strength,
                identity_index: 0,
                constant_noise: false,
                preview_inference_count: 0,
                noise_seed: seed,
            },
        ),
    };
    Ok(block_on(InteractiveGeometryExecutorBundleFactory::load(
        parameters,
    ))?)
}

fn compute_interactive_geometry(
    bundle: &mut FacadeInteractiveGeometryExecutorBundle,
    frame: Option<usize>,
    layer: &str,
    writer: &mut ArtifactWriter,
) -> Result<InteractiveExecutionReport, Box<dyn std::error::Error>> {
    let mut callback_error = None;
    let mut callback = |results: GeometryResults<'_>| {
        if callback_error.is_some() {
            return ControlFlow::Break(());
        }
        match write_facade_geometry_result(writer, layer, results) {
            Ok(()) => ControlFlow::Continue(()),
            Err(error) => {
                callback_error = Some(error);
                ControlFlow::Break(())
            }
        }
    };
    let report = match bundle {
        FacadeInteractiveGeometryExecutorBundle::Regression(executor) => match frame {
            Some(frame) => block_on(executor.compute_frame(frame, &mut callback))?,
            None => block_on(executor.compute_all_frames(&mut callback))?,
        },
        FacadeInteractiveGeometryExecutorBundle::Diffusion(executor) => match frame {
            Some(frame) => block_on(executor.compute_frame(frame, &mut callback))?,
            None => block_on(executor.compute_all_frames(&mut callback))?,
        },
    };
    if let Some(error) = callback_error {
        return Err(error.into());
    }
    Ok(report)
}

fn ensure_interactive_complete(
    report: InteractiveExecutionReport,
) -> Result<(), Box<dyn std::error::Error>> {
    match report.status {
        audio2face3d::audio2x::InteractiveExecutionStatus::Complete => Ok(()),
        audio2face3d::audio2x::InteractiveExecutionStatus::Interrupted => {
            Err("interactive geometry capture was interrupted".into())
        }
    }
}

fn capture_interactive_blendshape(
    model: &Model,
    seed: u64,
    samples: &[f32],
    selected_frame: Option<usize>,
    all_frames: bool,
    writer: &mut ArtifactWriter,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut geometry = create_interactive_facade_geometry(model, seed, samples)?;
    let total = match &geometry {
        FacadeInteractiveGeometryExecutorBundle::Regression(executor) => {
            InteractiveExecutor::total_frame_count(executor.as_ref())?
        }
        FacadeInteractiveGeometryExecutorBundle::Diffusion(executor) => {
            InteractiveExecutor::total_frame_count(executor.as_ref())?
        }
    };
    let frame = selected_frame.unwrap_or(total / 2);
    if frame >= total {
        return Err(format!("selected frame {frame} is outside {total} frames").into());
    }
    let frame_rate = match &geometry {
        FacadeInteractiveGeometryExecutorBundle::Regression(executor) => executor.frame_rate(),
        FacadeInteractiveGeometryExecutorBundle::Diffusion(executor) => executor.frame_rate(),
    };
    let mut blendshape = create_interactive_device_blendshape(model, frame_rate)?;
    if all_frames {
        let mut frame_metadata = Vec::new();
        let geometry_frames =
            collect_interactive_geometry(&mut geometry, None, &mut frame_metadata)?;
        let mut callback_error = None;
        let mut callback_frame = 0_usize;
        let report = block_on(blendshape.compute_all_frames(&geometry_frames, |output| {
            let metadata = frame_metadata[callback_frame];
            callback_frame += 1;
            if let Err(error) = push_interactive_gpu_blendshape(
                writer,
                "interactive-blendshape-all",
                metadata,
                output,
            ) {
                callback_error = Some(error);
                false
            } else {
                true
            }
        }))?;
        if let Some(error) = callback_error {
            return Err(error.into());
        }
        ensure_interactive_complete(report)?;
    } else {
        for (layer, invalidate) in [
            ("interactive-blendshape-random", false),
            ("interactive-blendshape-replay", false),
            ("interactive-blendshape-invalidation", true),
        ] {
            if invalidate {
                match &mut geometry {
                    FacadeInteractiveGeometryExecutorBundle::Regression(executor) => {
                        executor.invalidate_geometry(GeometryInvalidationLayer::Skin)?;
                    }
                    FacadeInteractiveGeometryExecutorBundle::Diffusion(executor) => {
                        executor.invalidate_geometry(GeometryInvalidationLayer::Skin)?;
                    }
                }
                blendshape.invalidate_blendshape(BlendshapeInvalidationLayer::BlendshapeWeights)?;
                *writer
                    .manifest_mut()
                    .counters
                    .entry("invalidation.weights".into())
                    .or_default() += 1;
            }
            let mut frame_metadata = Vec::new();
            let geometry_frame =
                collect_interactive_geometry(&mut geometry, Some(frame), &mut frame_metadata)?
                    .into_iter()
                    .next()
                    .ok_or("interactive geometry callback returned no frame")?;
            let mut callback_error = None;
            let report =
                block_on(
                    blendshape.compute_frame(frame, total, &geometry_frame, |output| {
                        let metadata = frame_metadata[0];
                        if let Err(error) =
                            push_interactive_gpu_blendshape(writer, layer, metadata, output)
                        {
                            callback_error = Some(error);
                            false
                        } else {
                            true
                        }
                    }),
                )?;
            if let Some(error) = callback_error {
                return Err(error.into());
            }
            ensure_interactive_complete(report)?;
        }
    }
    writer
        .manifest_mut()
        .counters
        .insert("total_frames".into(), total as u64);
    Ok(())
}

fn create_interactive_device_blendshape(
    model: &Model,
    frame_rate: FrameRate,
) -> Result<DeviceBlendshapeSolveInteractiveExecutor, Box<dyn std::error::Error>> {
    let device = GpuDevice::new(0)?;
    let stream = device.create_stream()?;
    let skin = load_reference_blendshape_component(model, "skin")?
        .map(|component| {
            GpuBlendshapeSolver::new(&device, &stream, component.data, &component.config)
        })
        .transpose()?;
    let tongue = load_reference_blendshape_component(model, "tongue")?
        .map(|component| {
            GpuBlendshapeSolver::new(&device, &stream, component.data, &component.config)
        })
        .transpose()?;
    let layer = InteractiveGpuBlendshapeLayer::with_default_cache(
        std::sync::Arc::clone(&device),
        stream,
        skin,
        tongue,
    )?;
    Ok(DeviceBlendshapeSolveInteractiveExecutor::from_layer(
        layer,
        model.sample_rate(),
        frame_rate,
    ))
}

fn collect_interactive_geometry(
    bundle: &mut FacadeInteractiveGeometryExecutorBundle,
    frame: Option<usize>,
    metadata: &mut Vec<audio2face3d::animation::InteractiveGeometryMetadata>,
) -> Result<Vec<RegressionGeometry>, Box<dyn std::error::Error>> {
    let mut frames = Vec::new();
    let mut callback_error = None;
    let mut callback = |results: GeometryResults<'_>| {
        // Preserve the producer's timestamps: Diffusion uses its model frame
        // rate, which need not be the Regression capture rate of 30 fps.
        let frame_metadata = audio2face3d::animation::InteractiveGeometryMetadata {
            frame: results.metadata.frame_index,
            inference: None,
            timestamp: results.metadata.timestamp,
            next_timestamp: results.metadata.next_timestamp,
        };
        match copy_facade_geometry_result(results) {
            Ok(frame) => {
                metadata.push(frame_metadata);
                frames.push(RegressionGeometry {
                    skin: frame.skin,
                    tongue: frame.tongue,
                    jaw_transform: frame.jaw_transform,
                    eyes_rotation: frame.eyes_rotation,
                });
                ControlFlow::Continue(())
            }
            Err(error) => {
                callback_error = Some(error);
                ControlFlow::Break(())
            }
        }
    };
    let report = match bundle {
        FacadeInteractiveGeometryExecutorBundle::Regression(executor) => match frame {
            Some(frame) => block_on(executor.compute_frame(frame, &mut callback))?,
            None => block_on(executor.compute_all_frames(&mut callback))?,
        },
        FacadeInteractiveGeometryExecutorBundle::Diffusion(executor) => match frame {
            Some(frame) => block_on(executor.compute_frame(frame, &mut callback))?,
            None => block_on(executor.compute_all_frames(&mut callback))?,
        },
    };
    if let Some(error) = callback_error {
        return Err(error.into());
    }
    ensure_interactive_complete(report)?;
    Ok(frames)
}

fn push_interactive_gpu_blendshape(
    writer: &mut ArtifactWriter,
    layer: &str,
    metadata: audio2face3d::animation::InteractiveGeometryMetadata,
    output: audio2face3d::animation::InteractiveGpuBlendshapeOutput<'_>,
) -> io::Result<()> {
    let mut skin = vec![0.0; output.skin_weight_count()];
    let mut tongue = vec![0.0; output.tongue_weight_count()];
    output
        .skin_weights
        .map(|weights| weights.copy_to(&mut skin, output.stream))
        .transpose()
        .map_err(io::Error::other)?;
    output
        .tongue_weights
        .map(|weights| weights.copy_to(&mut tongue, output.stream))
        .transpose()
        .map_err(io::Error::other)?;
    let mut weights = skin;
    weights.extend(tongue);
    writer.push_f32(
        RecordMetadata {
            layer: layer.into(),
            component: "weights".into(),
            track: 0,
            frame: Some(metadata.frame),
            inference: metadata.inference,
            timestamp: Some(metadata.timestamp),
            next_timestamp: Some(metadata.next_timestamp),
            shape: vec![weights.len()],
        },
        &weights,
    )
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
                bytes: Some(path.metadata()?.len()),
                license: Some(model_license(model.kind()).into()),
                revision: None,
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

fn model_license(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Regression | ModelKind::Diffusion => "NVIDIA Open Model License",
        ModelKind::Emotion => {
            "License Agreement for NVIDIA Audio2Emotion Model for Use with Audio2Face Project"
        }
    }
}

fn kind_name(kind: ModelKind) -> &'static str {
    match kind {
        ModelKind::Regression => "regression",
        ModelKind::Diffusion => "diffusion",
        ModelKind::Emotion => "emotion",
    }
}
