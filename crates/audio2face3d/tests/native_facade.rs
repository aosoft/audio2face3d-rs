#![cfg(all(feature = "animation", feature = "emotion", feature = "tensorrt"))]

use std::future::Future;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

use audio2face3d::audio2emotion::classifier::{
    ClassifierEmotionExecutorCreationParameters, ClassifierEmotionExecutorFactory,
    ClassifierEmotionInteractiveExecutorCreationParameters,
    ClassifierEmotionInteractiveExecutorFactory,
};
use audio2face3d::audio2emotion::post_process::{PostProcessData, PostProcessParams};
use audio2face3d::audio2emotion::{
    EmotionExecutor, EmotionExecutorCreationParameters, EmotionInteractiveExecutor,
    EmotionInteractiveExecutorCreationParameters, EmotionInvalidationLayer, EmotionTrackResources,
};
use audio2face3d::audio2face::diffusion::{
    DiffusionGeometryExecutorCreationParameters, DiffusionGeometryExecutorFactory,
    DiffusionGeometryInteractiveExecutorCreationParameters,
    DiffusionGeometryInteractiveExecutorFactory,
};
use audio2face3d::audio2face::regression::{
    RegressionGeometryExecutorCreationParameters, RegressionGeometryExecutorFactory,
    RegressionGeometryInteractiveExecutorCreationParameters,
    RegressionGeometryInteractiveExecutorFactory,
};
use audio2face3d::audio2face::{
    GeometryCallbacks, GeometryExecutionOption, GeometryExecutor,
    GeometryExecutorCreationParameters, GeometryInteractiveExecutor,
    GeometryInteractiveExecutorCreationParameters, GeometryInvalidationLayer,
    GeometryTrackResources,
};
use audio2face3d::audio2x::{
    AudioAccumulator, EmotionAccumulator, Executor, FrameRate, InteractiveExecutionStatus,
    InteractiveExecutor,
};
use audio2face3d::common::{GeometryParameters, NetworkDocument};
use audio2face3d::{Model, ModelKind, ModelParameters};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeFacadeManifest {
    regression_model: PathBuf,
    diffusion_model: PathBuf,
    emotion_model: PathBuf,
    input_wav: PathBuf,
    #[serde(default)]
    device_ordinal: i32,
    #[serde(default = "default_emotion_buffer_length")]
    emotion_buffer_length: usize,
}

fn default_emotion_buffer_length() -> usize {
    16_000
}

struct ThreadWaker(std::thread::Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    let mut future = std::pin::pin!(future);
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut context = Context::from_waker(&waker);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

fn resolve(base: &Path, path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        base.join(path)
    }
}

fn read_pcm16_mono(path: &Path) -> (usize, Vec<f32>) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()));
    assert!(bytes.len() >= 12, "truncated WAV header");
    assert_eq!(&bytes[0..4], b"RIFF", "input must be a RIFF WAV");
    assert_eq!(&bytes[8..12], b"WAVE", "input must be a WAVE file");
    let mut cursor = 12;
    let mut format = None;
    let mut data = None;
    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let start = cursor + 8;
        let end = start.checked_add(size).expect("WAV chunk size overflow");
        assert!(end <= bytes.len(), "truncated WAV chunk");
        if id == b"fmt " {
            assert!(size >= 16, "WAV fmt chunk is too short");
            let encoding = u16::from_le_bytes(bytes[start..start + 2].try_into().unwrap());
            let channels = u16::from_le_bytes(bytes[start + 2..start + 4].try_into().unwrap());
            let rate = u32::from_le_bytes(bytes[start + 4..start + 8].try_into().unwrap());
            let bits = u16::from_le_bytes(bytes[start + 14..start + 16].try_into().unwrap());
            format = Some((encoding, channels, rate as usize, bits));
        } else if id == b"data" {
            data = Some(&bytes[start..end]);
        }
        cursor = end + (size & 1);
    }
    let (encoding, channels, sample_rate, bits) = format.expect("WAV fmt chunk is missing");
    assert_eq!(
        (encoding, channels, bits),
        (1, 1, 16),
        "input must be mono PCM16"
    );
    let samples = data
        .expect("WAV data chunk is missing")
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]) as f32 / 32_768.0)
        .collect();
    (sample_rate, samples)
}

fn audio(samples: &[f32]) -> Arc<AudioAccumulator> {
    // Keep the short fixture in one chunk so standard execution can reset
    // without having released whole historical chunks behind its read cursor.
    let value = Arc::new(AudioAccumulator::new(samples.len().max(1), 0).unwrap());
    value.accumulate(samples).unwrap();
    value.close().unwrap();
    value
}

fn emotions(size: usize) -> Arc<EmotionAccumulator> {
    let value = Arc::new(EmotionAccumulator::new(size, 32).unwrap());
    value.accumulate(0, &vec![0.0; size]).unwrap();
    value.close().unwrap();
    value
}

fn geometry_emotion_size(model: &Model) -> usize {
    let NetworkDocument::Geometry(network) = model.network() else {
        panic!("geometry model expected")
    };
    match &network.params {
        GeometryParameters::Regression(parameters) => parameters.explicit_emotions.len(),
        GeometryParameters::Diffusion(parameters) => parameters.emotions.len(),
    }
}

/// Runs only in the native TensorRT tier. `AUDIO2FACE3D_TEST_FACADE_MODELS`
/// names a JSON manifest whose paths may be absolute or relative to the manifest.
#[test]
#[ignore = "requires CUDA, TensorRT, three acquired models, and licensed PCM input"]
fn acquired_models_execute_through_completed_facades() {
    let manifest_path = std::env::var_os("AUDIO2FACE3D_TEST_FACADE_MODELS")
        .map(PathBuf::from)
        .expect("AUDIO2FACE3D_TEST_FACADE_MODELS must name the native facade manifest");
    let manifest_bytes = std::fs::read(&manifest_path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", manifest_path.display()));
    let mut manifest: NativeFacadeManifest = serde_json::from_slice(&manifest_bytes)
        .unwrap_or_else(|error| panic!("invalid {}: {error}", manifest_path.display()));
    let base = manifest_path.parent().unwrap_or(Path::new("."));
    manifest.regression_model = resolve(base, manifest.regression_model);
    manifest.diffusion_model = resolve(base, manifest.diffusion_model);
    manifest.emotion_model = resolve(base, manifest.emotion_model);
    manifest.input_wav = resolve(base, manifest.input_wav);
    for path in [
        &manifest.regression_model,
        &manifest.diffusion_model,
        &manifest.emotion_model,
        &manifest.input_wav,
    ] {
        assert!(
            path.is_file(),
            "required native fixture is missing: {}",
            path.display()
        );
    }

    let (sample_rate, samples) = read_pcm16_mono(&manifest.input_wav);
    assert!(!samples.is_empty(), "native input WAV must contain samples");
    let frame_rate = FrameRate::new(30, 1).unwrap();

    for (expected, path) in [
        (ModelKind::Regression, &manifest.regression_model),
        (ModelKind::Diffusion, &manifest.diffusion_model),
    ] {
        eprintln!("native standard facade: {expected:?}");
        let model = Model::load(path).unwrap();
        assert_eq!(model.kind(), expected);
        let model_sample_rate = match model.network() {
            NetworkDocument::Geometry(network) => match &network.audio_params {
                audio2face3d::common::GeometryAudioParameters::Regression(value) => {
                    value.samplerate
                }
                audio2face3d::common::GeometryAudioParameters::Diffusion(value) => value.samplerate,
            },
            NetworkDocument::Emotion(_) => unreachable!(),
        };
        assert_eq!(
            sample_rate, model_sample_rate,
            "WAV/model sample rates differ"
        );
        let track = GeometryTrackResources {
            audio: audio(&samples),
            emotions: emotions(geometry_emotion_size(&model)),
        };
        let shared_audio = Arc::clone(&track.audio);
        let shared_emotions = Arc::clone(&track.emotions);
        shared_audio.reset().unwrap();
        let common = GeometryExecutorCreationParameters {
            tracks: vec![track],
            device_ordinal: manifest.device_ordinal,
            execution_option: GeometryExecutionOption::ALL,
        };
        let mut executor: Box<dyn GeometryExecutor> = match expected {
            ModelKind::Regression => Box::new(
                block_on(RegressionGeometryExecutorFactory::load(
                    RegressionGeometryExecutorCreationParameters {
                        model_path: path.clone(),
                        common,
                        input_strength: 1.0,
                        frame_rate,
                        source_emotion_shot: None,
                        source_emotion_frame: 0,
                    },
                ))
                .unwrap(),
            ),
            ModelKind::Diffusion => Box::new(
                block_on(DiffusionGeometryExecutorFactory::load(
                    DiffusionGeometryExecutorCreationParameters {
                        model_path: path.clone(),
                        common,
                        input_strength: 1.0,
                        frame_rate,
                        identity_index: 0,
                        constant_noise: true,
                        noise_seed: 0,
                    },
                ))
                .unwrap(),
            ),
            ModelKind::Emotion => unreachable!(),
        };
        assert_eq!(executor.track_count(), 1);
        assert_eq!(executor.total_frame_count(0).unwrap(), None);
        let mut awaiting = false;
        for _ in 0..100 {
            let mut empty_callback = |_: audio2face3d::audio2face::GeometryResults<'_>| {
                panic!("empty open audio must not produce a frame")
            };
            let report = block_on(
                executor
                    .execute(GeometryCallbacks {
                        results: &mut empty_callback,
                        emotions: None,
                    })
                    .unwrap(),
            )
            .unwrap();
            if report.state == audio2face3d::audio2x::ExecutionState::AwaitingInput {
                awaiting = true;
                break;
            }
        }
        assert!(awaiting, "empty input did not reach AwaitingInput");
        shared_audio.accumulate(&samples).unwrap();
        shared_audio.close().unwrap();
        let mut metadata = Vec::new();
        let mut callback = |result: audio2face3d::audio2face::GeometryResults<'_>| {
            metadata.push((
                result.metadata.track_index,
                result.metadata.frame_index,
                result.metadata.timestamp,
            ));
            ControlFlow::Continue(())
        };
        let mut calls = 0;
        while executor.ready_track_count() > 0 {
            assert!(
                calls < 10_000,
                "geometry executor failed to reach completion"
            );
            let execution = executor
                .execute(GeometryCallbacks {
                    results: &mut callback,
                    emotions: None,
                })
                .unwrap();
            block_on(execution.wait_all()).unwrap();
            calls += 1;
        }
        assert!(
            !metadata.is_empty(),
            "completed geometry facade produced no frames"
        );
        assert!(
            metadata.windows(2).all(|pair| pair[0] < pair[1]),
            "geometry callback metadata is not ordered"
        );
        assert_eq!(metadata[0].1, 0, "public frames must start at zero");
        assert_eq!(executor.total_frame_count(0).unwrap(), Some(metadata.len()));
        for (frame, &(track, index, timestamp)) in metadata.iter().enumerate() {
            assert_eq!((track, index), (0, frame));
            assert_eq!(executor.frame_timestamp(frame).unwrap(), timestamp);
        }
        if shared_audio.nb_dropped_samples() > 0 || shared_emotions.state().dropped_emotions > 0 {
            assert!(matches!(
                executor.reset_track(0),
                Err(audio2face3d::Error::InputHistoryUnavailable { track: 0 })
            ));
        }
        // Repopulate the same shared inputs before replaying an executor whose
        // normal streaming policy has released historical chunks.
        shared_audio.reset().unwrap();
        shared_audio.accumulate(&samples).unwrap();
        shared_audio.close().unwrap();
        shared_emotions.reset();
        shared_emotions
            .accumulate(0, &vec![0.0; geometry_emotion_size(&model)])
            .unwrap();
        shared_emotions.close().unwrap();
        executor.reset_track(0).unwrap();
        let mut replay_first = None;
        for _ in 0..100 {
            let mut callback = |result: audio2face3d::audio2face::GeometryResults<'_>| {
                replay_first = Some((
                    result.metadata.track_index,
                    result.metadata.frame_index,
                    result.metadata.timestamp,
                ));
                ControlFlow::Break(())
            };
            block_on(
                executor
                    .execute(GeometryCallbacks {
                        results: &mut callback,
                        emotions: None,
                    })
                    .unwrap(),
            )
            .unwrap();
            if replay_first.is_some() {
                break;
            }
        }
        assert_eq!(
            replay_first,
            metadata.first().copied(),
            "reset changed first callback metadata"
        );
        drop(executor);
        let common = GeometryInteractiveExecutorCreationParameters {
            audio: audio(&samples),
            emotions: emotions(geometry_emotion_size(&model)),
            device_ordinal: manifest.device_ordinal,
            execution_option: GeometryExecutionOption::ALL,
        };
        let mut interactive: Box<dyn GeometryInteractiveExecutor> = match expected {
            ModelKind::Regression => Box::new(
                block_on(RegressionGeometryInteractiveExecutorFactory::load(
                    RegressionGeometryInteractiveExecutorCreationParameters {
                        model_path: path.clone(),
                        common,
                        input_strength: 1.0,
                        frame_rate,
                        source_emotion_shot: None,
                        source_emotion_frame: 0,
                        batch_size: 1,
                    },
                ))
                .unwrap(),
            ),
            ModelKind::Diffusion => Box::new(
                block_on(DiffusionGeometryInteractiveExecutorFactory::load(
                    DiffusionGeometryInteractiveExecutorCreationParameters {
                        model_path: path.clone(),
                        common,
                        input_strength: 1.0,
                        identity_index: 0,
                        constant_noise: true,
                        preview_inference_count: 1,
                        noise_seed: 0,
                    },
                ))
                .unwrap(),
            ),
            _ => unreachable!(),
        };
        verify_geometry_transitions(interactive.as_mut());
    }

    let emotion_model = Model::load(&manifest.emotion_model).unwrap();
    assert_eq!(emotion_model.kind(), ModelKind::Emotion);
    let NetworkDocument::Emotion(network) = emotion_model.network() else {
        unreachable!()
    };
    assert_eq!(
        sample_rate, network.audio_params.samplerate,
        "WAV/model sample rates differ"
    );
    let ModelParameters::Emotion(config) = emotion_model.parameters(0).unwrap() else {
        unreachable!()
    };
    let correspondence = network
        .emotions
        .iter()
        .map(|name| i32::try_from(*config.emotion_correspondence.get(name).unwrap()).unwrap())
        .collect();
    let post_process_data = PostProcessData {
        inference_emotion_length: network.emotions.len(),
        output_emotion_length: config.output_emotion_length,
        emotion_correspondence: correspondence,
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
    let preferred = emotions(config.output_emotion_length);
    let standard_audio = audio(&samples);
    let parameters = ClassifierEmotionExecutorCreationParameters {
        model_path: manifest.emotion_model,
        common: EmotionExecutorCreationParameters {
            tracks: vec![EmotionTrackResources {
                audio: Arc::clone(&standard_audio),
            }],
            device_ordinal: manifest.device_ordinal,
        },
        input_strength: 1.0,
        buffer_length: manifest.emotion_buffer_length,
        frame_rate,
        inferences_to_skip: 0,
        post_process_data: post_process_data.clone(),
        post_process_params: post_process_params.clone(),
        preferred_emotions: vec![preferred],
    };
    let mut executor = block_on(ClassifierEmotionExecutorFactory::load(parameters)).unwrap();
    assert_eq!(executor.track_count(), 1);
    assert_eq!(executor.emotion_count(), config.output_emotion_length);
    let mut metadata = Vec::new();
    let mut callback = |result: audio2face3d::audio2emotion::EmotionResults<'_>| {
        metadata.push((
            result.metadata.track_index,
            result.metadata.frame_index,
            result.metadata.timestamp,
        ));
        ControlFlow::Continue(())
    };
    let mut calls = 0;
    while executor.ready_track_count() > 0 {
        assert!(
            calls < 10_000,
            "emotion executor failed to reach completion"
        );
        let execution = executor.execute(&mut callback).unwrap();
        block_on(execution.wait_all()).unwrap();
        calls += 1;
    }
    assert!(
        !metadata.is_empty(),
        "completed emotion facade produced no frames"
    );
    assert!(
        metadata.windows(2).all(|pair| pair[0] < pair[1]),
        "emotion callback metadata is not ordered"
    );
    executor.reset_track(0).unwrap();

    let interactive_audio = audio(&samples);
    let mut interactive = block_on(ClassifierEmotionInteractiveExecutorFactory::load(
        ClassifierEmotionInteractiveExecutorCreationParameters {
            model_path: emotion_model.descriptor_path().to_owned(),
            common: EmotionInteractiveExecutorCreationParameters {
                audio: interactive_audio,
                preferred_emotions: None,
                device_ordinal: manifest.device_ordinal,
            },
            input_strength: 1.0,
            frame_rate,
            inferences_to_skip: 0,
            batch_size: 1,
            post_process_data,
            post_process_params,
        },
    ))
    .unwrap();
    assert!(interactive.total_frame_count().unwrap() > 0);
    assert!(!interactive.is_fully_valid());
    let mut interactive_frames = Vec::new();
    let mut callback = |result: audio2face3d::audio2emotion::EmotionResults<'_>| {
        interactive_frames.push(result.metadata.frame_index);
        ControlFlow::Continue(())
    };
    block_on(interactive.compute_all_frames(&mut callback)).unwrap();
    assert!(!interactive_frames.is_empty());
    assert!(interactive_frames.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(interactive.is_fully_valid());
    interactive
        .invalidate_emotion(EmotionInvalidationLayer::PostProcessing)
        .unwrap();
    assert!(!interactive.is_emotion_valid(EmotionInvalidationLayer::PostProcessing));
    let total = interactive.total_frame_count().unwrap();
    for frame in [total / 2, total - 1, 0] {
        let expected_timestamp = interactive.frame_timestamp(frame).unwrap();
        let mut one_frame = 0;
        let mut callback = |result: audio2face3d::audio2emotion::EmotionResults<'_>| {
            assert_eq!(result.metadata.frame_index, frame);
            assert_eq!(result.metadata.timestamp, expected_timestamp);
            one_frame += 1;
            ControlFlow::Continue(())
        };
        block_on(interactive.compute_frame(frame, &mut callback)).unwrap();
        assert_eq!(one_frame, 1);
    }

    // Exercise the public facade's state transitions with actual inference,
    // not only the mock scheduler. A new compute_all call replays all frames.
    for layer in [
        EmotionInvalidationLayer::PostProcessing,
        EmotionInvalidationLayer::Inference,
        EmotionInvalidationLayer::All,
    ] {
        interactive.invalidate_emotion(layer).unwrap();
        assert!(!interactive.is_emotion_valid(layer));
        assert!(!interactive.is_fully_valid());
        let mut frames = Vec::new();
        let mut callback = |result: audio2face3d::audio2emotion::EmotionResults<'_>| {
            frames.push(result.metadata.frame_index);
            ControlFlow::Continue(())
        };
        let report = block_on(interactive.compute_all_frames(&mut callback)).unwrap();
        assert_eq!(report.status, InteractiveExecutionStatus::Complete);
        assert_eq!(report.emitted_frames, total);
        assert_eq!(frames, interactive_frames);
        assert!(interactive.is_fully_valid());
    }
    interactive
        .invalidate_emotion(EmotionInvalidationLayer::None)
        .unwrap();
    assert!(interactive.is_fully_valid());

    for use_interrupt_handle in [false, true] {
        let interrupt = interactive.interrupt_handle();
        let mut frames = Vec::new();
        let mut callback = |result: audio2face3d::audio2emotion::EmotionResults<'_>| {
            frames.push(result.metadata.frame_index);
            if use_interrupt_handle {
                interrupt.interrupt();
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        };
        let report = block_on(interactive.compute_all_frames(&mut callback)).unwrap();
        assert_eq!(report.status, InteractiveExecutionStatus::Interrupted);
        assert_eq!(report.emitted_frames, 1);
        assert_eq!(frames, vec![0]);
        assert!(!interactive.is_fully_valid());

        let mut replay = Vec::new();
        let mut callback = |result: audio2face3d::audio2emotion::EmotionResults<'_>| {
            replay.push(result.metadata.frame_index);
            ControlFlow::Continue(())
        };
        let report = block_on(interactive.compute_all_frames(&mut callback)).unwrap();
        assert_eq!(report.status, InteractiveExecutionStatus::Complete);
        assert_eq!(report.emitted_frames, total);
        assert_eq!(replay, interactive_frames);
        assert!(interactive.is_fully_valid());
    }
}

fn verify_geometry_transitions(executor: &mut dyn GeometryInteractiveExecutor) {
    let total = executor.total_frame_count().unwrap();
    assert!(total > 1);
    let expected: Vec<_> = (0..total)
        .map(|frame| (frame, executor.frame_timestamp(frame).unwrap()))
        .collect();
    for layer in [
        GeometryInvalidationLayer::All,
        GeometryInvalidationLayer::Inference,
        GeometryInvalidationLayer::Skin,
        GeometryInvalidationLayer::Tongue,
        GeometryInvalidationLayer::Teeth,
        GeometryInvalidationLayer::Eyes,
    ] {
        executor.invalidate_geometry(layer).unwrap();
        assert!(!executor.is_geometry_valid(layer), "{layer:?}");
        let mut observed = Vec::new();
        let mut callback = |result: audio2face3d::audio2face::GeometryResults<'_>| {
            assert_eq!(result.metadata.track_index, 0);
            observed.push((result.metadata.frame_index, result.metadata.timestamp));
            ControlFlow::Continue(())
        };
        let report = block_on(executor.compute_all_frames(&mut callback)).unwrap();
        assert_eq!(report.status, InteractiveExecutionStatus::Complete);
        assert_eq!(report.emitted_frames, total);
        assert_eq!(observed, expected);
        assert!(executor.is_fully_valid());
    }
    executor
        .invalidate_geometry(GeometryInvalidationLayer::None)
        .unwrap();
    assert!(executor.is_fully_valid());
    for interrupt_requested in [false, true] {
        let interrupt = executor.interrupt_handle();
        let mut observed = Vec::new();
        let mut callback = |result: audio2face3d::audio2face::GeometryResults<'_>| {
            observed.push(result.metadata.frame_index);
            if interrupt_requested {
                interrupt.interrupt();
                ControlFlow::Continue(())
            } else {
                ControlFlow::Break(())
            }
        };
        let report = block_on(executor.compute_all_frames(&mut callback)).unwrap();
        assert_eq!(report.status, InteractiveExecutionStatus::Interrupted);
        assert_eq!(report.emitted_frames, 1);
        assert_eq!(observed, vec![0]);
        let mut replay = Vec::new();
        let mut callback = |result: audio2face3d::audio2face::GeometryResults<'_>| {
            replay.push((result.metadata.frame_index, result.metadata.timestamp));
            ControlFlow::Continue(())
        };
        let report = block_on(executor.compute_all_frames(&mut callback)).unwrap();
        assert_eq!(report.status, InteractiveExecutionStatus::Complete);
        assert_eq!(report.emitted_frames, total);
        assert_eq!(replay, expected);
        assert!(executor.is_fully_valid());
    }
}
