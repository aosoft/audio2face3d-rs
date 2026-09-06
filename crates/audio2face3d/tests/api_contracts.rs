//! Compile-pass coverage for the Step 1 declaration surface.
//! No CUDA installation or model execution is required for portable contracts.

use audio2face3d::audio2x::{
    AudioAccumulator, CallbackMetadata, CudaStreamRef, DeviceComponentResults, DeviceView,
    EmotionAccumulator, Error, Execution, ExecutionReport, ExecutionState, Executor,
    ExecutorFuture, FloatAccumulator, FrameRate, InteractiveExecutionReport,
    InteractiveExecutionStatus, InteractiveExecutor, InteractiveInterruptHandle, RangeConfig,
    Result, TransferError,
};
use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

fn assert_send<T: Send>() {}
fn assert_send_sync<T: Send + Sync>() {}
fn assert_completion<T: Future<Output = Result<ExecutionReport>> + Send>() {}

// A conflicting blanket implementation makes inference ambiguous if a type
// accidentally acquires a forbidden trait.
macro_rules! assert_not_impl {
    ($ty:ty, $bound:path) => {{
        trait AmbiguousIfImplemented<A> {
            fn check() {}
        }
        impl<T: ?Sized> AmbiguousIfImplemented<()> for T {}
        impl<T: ?Sized + $bound> AmbiguousIfImplemented<u8> for T {}
        let _ = <$ty as AmbiguousIfImplemented<_>>::check;
    }};
}

// Taking these function items type-checks the consumer's await signatures without
// manufacturing an opaque Step 1 Execution or claiming worker scheduling exists.
async fn wait_for_track(mut execution: Execution, track: usize) -> Result<ExecutionReport> {
    execution.wait_track(track).await?;
    execution.await
}

fn wait_for_all(execution: Execution) -> ExecutorFuture<'static, ExecutionReport> {
    execution.wait_all()
}

#[test]
fn portable_contract_types_and_dyn_roots_are_available() {
    assert_completion::<Execution>();
    assert_send::<ExecutorFuture<'_, InteractiveExecutionReport>>();
    assert_send_sync::<InteractiveInterruptHandle>();
    assert_send_sync::<AudioAccumulator>();
    assert_send_sync::<EmotionAccumulator>();
    assert_send_sync::<FloatAccumulator>();
    assert_send_sync::<Error>();
    let _: Option<&dyn Executor> = None;
    let _: Option<&dyn InteractiveExecutor> = None;
    let _: Option<DeviceView<'_, f32>> = None;
    let _: Option<CudaStreamRef<'_>> = None;
    let _: Option<DeviceComponentResults<'_>> = None;
    let _: Option<TransferError<Box<dyn Executor>>> = None;
    let _ = wait_for_track;
    let _ = wait_for_all;
    let _ = RangeConfig {
        default_value: 1_f32,
        minimum: 0.0,
        maximum: 2.0,
        description: "strength",
    };
    let _ = CallbackMetadata {
        track_index: 2,
        frame_index: 3,
        timestamp: -1,
        next_timestamp: 532,
    };
    let _ = ExecutionReport {
        state: ExecutionState::AwaitingInput,
        executed_tracks: 0,
        emitted_frames: 0,
    };
    let _ = FrameRate::new(60, 1).unwrap();
}

#[test]
fn boxed_send_future_can_be_polled_without_a_runtime() {
    let mut future: ExecutorFuture<'_, InteractiveExecutionReport> = Box::pin(async {
        Ok(InteractiveExecutionReport {
            status: InteractiveExecutionStatus::Complete,
            emitted_frames: 4,
        })
    });
    let mut context = Context::from_waker(Waker::noop());
    assert_eq!(
        Pin::new(&mut future).poll(&mut context),
        Poll::Ready(Ok(InteractiveExecutionReport {
            status: InteractiveExecutionStatus::Complete,
            emitted_frames: 4
        }))
    );
}

#[test]
fn symbol_ledger_is_well_formed_and_internally_consistent() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/api-compatibility-symbols.json");
    let document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    assert_eq!(document["schema_version"], 1);
    let symbols = document["symbols"].as_array().unwrap();
    assert_eq!(
        document["coverage"]["counts"]["total_records"].as_u64(),
        Some(symbols.len() as u64)
    );

    let mut ids = HashSet::new();
    let mut declared_paths = HashSet::new();
    let mut statuses = std::collections::BTreeMap::<&str, u64>::new();
    for symbol in symbols {
        let id = symbol["id"].as_str().unwrap();
        assert!(ids.insert(id), "duplicate ledger id: {id}");
        let kind = symbol["kind"].as_str().unwrap();
        assert!(!kind.is_empty(), "missing kind: {id}");
        assert!(
            symbol["step"].as_u64().is_some()
                || matches!(symbol["step"].as_str(), Some("P2" | "P3")),
            "invalid step: {id}"
        );
        let feature = symbol["feature"].as_array().unwrap();
        assert!(!feature.is_empty(), "missing feature: {id}");
        let status = symbol["status"].as_str().unwrap();
        assert!(matches!(status, "declared" | "planned" | "deferred"));
        *statuses.entry(status).or_default() += 1;

        if status == "declared" {
            let rust_path = symbol["new_rust_path"].as_str().unwrap();
            assert!(
                !rust_path.is_empty(),
                "declared symbol lacks Rust path: {id}"
            );
            assert!(
                declared_paths.insert(rust_path),
                "duplicate declared Rust path: {rust_path}"
            );
        }

        for origin in symbol["origins"].as_array().unwrap() {
            let name = origin["fq_name"].as_str().unwrap();
            let header = origin["header"].as_str().unwrap();
            assert!(name.starts_with("nva2"), "invalid SDK name: {name}");
            assert!(!std::path::Path::new(header).is_absolute());
            assert!(!header.contains("\\"));
            assert!(!header.split('/').any(|part| part == "internal"));
            assert!(
                header.starts_with("audio2x-common/include/audio2x/")
                    || header.starts_with("audio2face-sdk/include/audio2face/")
                    || header.starts_with("audio2emotion-sdk/include/audio2emotion/"),
                "header is outside public roots: {header}"
            );
        }
    }

    for (status, actual) in statuses {
        assert_eq!(
            document["coverage"]["counts"]["by_status"][status].as_u64(),
            Some(actual),
            "incorrect status count for {status}"
        );
    }
}

#[test]
fn borrowed_cuda_contracts_follow_the_audited_thread_safety_matrix() {
    assert_send_sync::<DeviceView<'static, f32>>();
    assert_send_sync::<CudaStreamRef<'static>>();
}

#[cfg(all(feature = "animation", feature = "emotion"))]
#[test]
fn face_and_emotion_creation_parameters_share_one_audio_accumulator() {
    let audio = std::sync::Arc::new(AudioAccumulator::new(16, 1).unwrap());
    let face = audio2face3d::audio2face::GeometryExecutorCreationParameters {
        tracks: vec![audio2face3d::audio2face::GeometryTrackResources {
            audio: std::sync::Arc::clone(&audio),
            emotions: std::sync::Arc::new(EmotionAccumulator::new(2, 1).unwrap()),
        }],
        device_ordinal: 0,
        execution_option: audio2face3d::audio2face::GeometryExecutionOption::ALL,
    };
    let emotion = audio2face3d::audio2emotion::EmotionExecutorCreationParameters {
        tracks: vec![audio2face3d::audio2emotion::EmotionTrackResources {
            audio: std::sync::Arc::clone(&audio),
        }],
        device_ordinal: 0,
    };
    assert!(std::sync::Arc::ptr_eq(&face.tracks[0].audio, &audio));
    assert!(std::sync::Arc::ptr_eq(
        &face.tracks[0].audio,
        &emotion.tracks[0].audio
    ));
}

#[cfg(feature = "animation")]
mod face {
    use super::*;
    use audio2face3d::audio2face::*;
    use std::cell::Cell;
    use std::ops::ControlFlow;
    use std::rc::Rc;
    use std::sync::Arc;

    // A synchronous callback deliberately captures a non-Send stack borrow.
    fn start(executor: &mut dyn GeometryExecutor) -> Result<Execution> {
        let count = Rc::new(Cell::new(0));
        let mut results = |_: GeometryResults<'_>| {
            count.set(count.get() + 1);
            ControlFlow::Continue(())
        };
        let mut emotion_count = 0;
        let mut emotions = |_: FaceEmotions<'_>| {
            emotion_count += 1;
        };
        executor.execute(GeometryCallbacks {
            results: &mut results,
            emotions: Some(&mut emotions),
        })
    }

    fn compute<'a>(
        executor: &'a mut dyn GeometryInteractiveExecutor,
        callback: &'a mut (dyn for<'r> FnMut(GeometryResults<'r>) -> ControlFlow<()> + Send),
    ) -> ExecutorFuture<'a, InteractiveExecutionReport> {
        executor.compute_frame(0, callback)
    }

    struct CustomRunner;
    impl JobRunner for CustomRunner {
        fn enqueue(&self, task: JobRunnerTask) -> Result<()> {
            // Consuming the owning task makes this an at-most-once invocation.
            task.run();
            Ok(())
        }
    }

    #[test]
    fn specialized_traits_and_custom_runner_are_dyn_compatible() {
        let _: Option<&dyn FaceExecutor> = None;
        let _: Option<&dyn GeometryExecutor> = None;
        let _: Option<&dyn BlendshapeExecutor> = None;
        let _: Option<&dyn GeometryInteractiveExecutor> = None;
        let _: Option<&dyn BlendshapeInteractiveExecutor> = None;
        let _ = start;
        let _ = compute;
        assert_send::<JobRunnerTask>();
        assert_not_impl!(JobRunnerTask, Clone);
        assert_send_sync::<ThreadPoolJobRunner>();
        fn accepts_runner<T: JobRunner>() {}
        accepts_runner::<ThreadPoolJobRunner>();
        let runner: Arc<dyn JobRunner> = Arc::new(CustomRunner);
        let parameters = HostBlendshapeSolveExecutorCreationParameters {
            components: BlendshapeSolveExecutorCreationParameters {
                skin: None,
                tongue: None,
            },
            job_runner: Some(Arc::clone(&runner)),
        };
        assert!(Arc::ptr_eq(
            parameters.job_runner.as_ref().unwrap(),
            &runner
        ));
        let callback: HostBlendshapeCallback = Arc::new(|event: HostBlendshapeEvent<'_>| {
            if let Ok(result) = event {
                let _ = result.weights.len();
            }
        });
        drop(callback);
    }

    #[test]
    fn geometry_resources_preserve_accumulator_identity() {
        let audio = Arc::new(AudioAccumulator::new(8, 1).unwrap());
        let emotions = Arc::new(EmotionAccumulator::new(2, 1).unwrap());
        let parameters = GeometryExecutorCreationParameters {
            tracks: vec![GeometryTrackResources {
                audio: Arc::clone(&audio),
                emotions: Arc::clone(&emotions),
            }],
            device_ordinal: 0,
            execution_option: GeometryExecutionOption::ALL,
        };
        assert!(Arc::ptr_eq(&parameters.tracks[0].audio, &audio));
        assert!(Arc::ptr_eq(&parameters.tracks[0].emotions, &emotions));
    }

    #[test]
    fn geometry_invalidation_uses_sdk_layer_numbers() {
        assert_eq!(GeometryInvalidationLayer::None as usize, 0);
        assert_eq!(GeometryInvalidationLayer::All as usize, 1);
        assert_eq!(GeometryInvalidationLayer::Inference as usize, 2);
        assert_eq!(GeometryInvalidationLayer::Skin as usize, 3);
        assert_eq!(GeometryInvalidationLayer::Tongue as usize, 4);
        assert_eq!(GeometryInvalidationLayer::Teeth as usize, 5);
        assert_eq!(GeometryInvalidationLayer::Eyes as usize, 6);
        assert_eq!(BlendshapeInvalidationLayer::SkinSolverPrepare as usize, 101);
        assert_eq!(
            BlendshapeInvalidationLayer::TongueSolverPrepare as usize,
            102
        );
        assert_eq!(BlendshapeInvalidationLayer::BlendshapeWeights as usize, 103);
        assert_eq!(GeometryExecutionOption::ALL.bits(), 15);
        assert!(GeometryExecutionOption::SKIN_TONGUE.contains(GeometryExecutionOption::SKIN));
    }

    #[test]
    fn sdk_named_animator_and_solver_factories_are_concrete() {
        let teeth = create_animator_teeth(
            AnimatorTeethParams {
                lower_teeth_strength: 1.0,
                lower_teeth_height_offset: 0.0,
                lower_teeth_depth_offset: 0.0,
            },
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        assert_eq!(teeth.neutral_pose().len(), 9);

        let names = ["pose"];
        let solver = create_blendshape_solver(BlendshapeSolveComponentParameters {
            params: BlendshapeSolverParams::default(),
            config: BlendshapeSolverConfigView {
                active_poses: &[1],
                cancel_poses: &[-1],
                symmetry_poses: &[-1],
                multipliers: &[1.0],
                offsets: &[0.0],
            },
            data: BlendshapeSolverDataView {
                neutral_pose: &[0.0, 0.0, 0.0],
                delta_poses: &[1.0, 0.0, 0.0],
                pose_mask: None,
                pose_names: &names,
            },
        })
        .unwrap();
        assert_eq!(solver.pose_name(0), Some("pose"));
    }

    #[cfg(feature = "tensorrt")]
    #[test]
    fn native_model_facades_are_unique_sendable_types() {
        use audio2face3d::audio2face::{diffusion::*, regression::*};
        assert_send::<RegressionGeometryExecutor>();
        assert_send::<DiffusionGeometryExecutor>();
        assert_send::<RegressionGeometryInteractiveExecutor>();
        assert_send::<DiffusionGeometryInteractiveExecutor>();
        assert_not_impl!(RegressionGeometryExecutor, Sync);
        assert_not_impl!(DiffusionGeometryExecutor, Sync);
        assert_not_impl!(RegressionGeometryInteractiveExecutor, Sync);
        assert_not_impl!(DiffusionGeometryInteractiveExecutor, Sync);
        assert_not_impl!(RegressionGeometryExecutor, Clone);
        assert_not_impl!(DiffusionGeometryExecutor, Clone);
        assert_send::<GeometryExecutorBundle>();
        assert_send::<BlendshapeExecutorBundle>();
        assert_not_impl!(GeometryExecutorBundle, Clone);

        use audio2face3d::audio2face::{
            InteractiveGeometryBundleCreationParameters, InteractiveGeometryExecutorBundle,
            InteractiveGeometryExecutorBundleFactory, InteractiveGeometryExecutorMut,
            InteractiveGeometryExecutorRef,
        };
        assert_send::<InteractiveGeometryExecutorBundle>();
        assert_not_impl!(InteractiveGeometryExecutorBundle, Clone);

        fn interactive_bundle_accessors(bundle: &InteractiveGeometryExecutorBundle) {
            let _: InteractiveGeometryExecutorRef<'_> = bundle.executor();
            let _: &audio2face3d::cuda::CudaStream = bundle.cuda_stream();
            let _: &std::sync::Arc<AudioAccumulator> = bundle.audio_accumulator();
            let _: &std::sync::Arc<EmotionAccumulator> = bundle.emotion_accumulator();
        }
        fn interactive_bundle_mut_accessor(bundle: &mut InteractiveGeometryExecutorBundle) {
            let _: InteractiveGeometryExecutorMut<'_> = bundle.executor_mut();
        }
        fn interactive_bundle_factory(
            parameters: InteractiveGeometryBundleCreationParameters,
        ) -> ExecutorFuture<'static, InteractiveGeometryExecutorBundle> {
            InteractiveGeometryExecutorBundleFactory::load(parameters)
        }
        let _ = interactive_bundle_accessors;
        let _ = interactive_bundle_mut_accessor;
        let _ = interactive_bundle_factory;
        let _ = RegressionGeometryExecutorFactory::load;
        let _ = RegressionGeometryInteractiveExecutorFactory::load;
        let _ = DiffusionGeometryExecutorFactory::load;
        let _ = DiffusionGeometryInteractiveExecutorFactory::load;
        let _ = GeometryExecutorBundleFactory::load;
        let _ = create_regression_geometry_executor;
        let _ = create_diffusion_geometry_executor;
        let _ = create_host_blendshape_solve_executor;
        let _ = create_device_blendshape_solve_executor;
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn native_blendshape_declarations_do_not_require_tensorrt() {
        assert_send::<HostBlendshapeSolveExecutor>();
        assert_send::<DeviceBlendshapeSolveExecutor>();
        assert_send::<HostBlendshapeSolveInteractiveExecutor>();
        assert_send::<DeviceBlendshapeSolveInteractiveExecutor>();
        assert_not_impl!(HostBlendshapeSolveExecutor, Sync);
        assert_not_impl!(DeviceBlendshapeSolveExecutor, Sync);
        assert_not_impl!(HostBlendshapeSolveExecutor, Clone);
        assert_not_impl!(DeviceBlendshapeSolveExecutor, Clone);
    }
}

#[cfg(feature = "emotion")]
mod emotion {
    use super::*;
    use audio2face3d::audio2emotion::*;
    use std::ops::ControlFlow;
    use std::sync::Arc;

    fn start(executor: &mut dyn EmotionExecutor) -> Result<Execution> {
        let mut timestamps = Vec::new();
        executor.execute(&mut |result: EmotionResults<'_>| {
            timestamps.push(result.metadata.timestamp);
            ControlFlow::Continue(())
        })
    }

    fn compute<'a>(
        executor: &'a mut dyn EmotionInteractiveExecutor,
        callback: &'a mut (dyn for<'r> FnMut(EmotionResults<'r>) -> ControlFlow<()> + Send),
    ) -> ExecutorFuture<'a, InteractiveExecutionReport> {
        executor.compute_all_frames(callback)
    }

    #[test]
    fn emotion_contracts_are_portable_and_dyn_compatible() {
        let _ = start;
        let _ = compute;
        let _: Option<&dyn EmotionExecutor> = None;
        let _: Option<&dyn EmotionInteractiveExecutor> = None;
        let _: Option<PostProcessData> = None;
        let _ = PostProcessParams::default();
        let audio = Arc::new(AudioAccumulator::new(8, 1).unwrap());
        let parameters = EmotionExecutorCreationParameters {
            tracks: vec![EmotionTrackResources {
                audio: Arc::clone(&audio),
            }],
            device_ordinal: 0,
        };
        assert!(Arc::ptr_eq(&parameters.tracks[0].audio, &audio));
        assert_eq!(EmotionInvalidationLayer::None as usize, 0);
        assert_eq!(EmotionInvalidationLayer::All as usize, 1);
        assert_eq!(EmotionInvalidationLayer::Inference as usize, 2);
        assert_eq!(EmotionInvalidationLayer::PostProcessing as usize, 3);

        let mut processor = post_process::create_post_processor(
            PostProcessData {
                inference_emotion_length: 1,
                output_emotion_length: 1,
                emotion_correspondence: vec![0],
            },
            PostProcessParams {
                max_emotions: 1,
                beginning_emotion: vec![0.0],
                preferred_emotion: vec![0.0],
                ..PostProcessParams::default()
            },
        )
        .unwrap();
        assert_eq!(processor.process(&[0.0]).unwrap().len(), 1);
    }

    #[cfg(feature = "tensorrt")]
    #[test]
    fn classifier_facades_are_unique_sendable_types() {
        use audio2face3d::audio2emotion::classifier::*;
        assert_send::<ClassifierEmotionExecutor>();
        assert_send::<ClassifierEmotionInteractiveExecutor>();
        assert_not_impl!(ClassifierEmotionExecutor, Sync);
        assert_not_impl!(ClassifierEmotionInteractiveExecutor, Sync);
        assert_not_impl!(ClassifierEmotionExecutor, Clone);
        assert_send::<EmotionExecutorBundle>();
        assert_not_impl!(EmotionExecutorBundle, Clone);
        let _ = ClassifierEmotionExecutorFactory::load;
        let _ = ClassifierEmotionInteractiveExecutorFactory::load;
        let _ = EmotionExecutorBundleFactory::load;
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn post_process_facades_do_not_require_tensorrt() {
        use audio2face3d::audio2emotion::post_process::*;
        assert_send::<PostProcessEmotionExecutor>();
        assert_send::<PostProcessEmotionInteractiveExecutor>();
        assert_not_impl!(PostProcessEmotionExecutor, Sync);
        assert_not_impl!(PostProcessEmotionInteractiveExecutor, Sync);
        assert_not_impl!(PostProcessEmotionExecutor, Clone);
        assert_send::<EmotionExecutorBundle>();
        let _ = PostProcessEmotionExecutorFactory::load;
        let _ = PostProcessEmotionInteractiveExecutorFactory::load;
        let _ = EmotionExecutorBundleFactory::post_process;
    }
}
