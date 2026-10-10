#[cfg(any(feature = "mock", feature = "cuda"))]
mod support;
use audio2face3d::{
    Audio2Face3DContext,
    runtime::{NativeRuntimeConfig, NativeSearchPolicy},
};

#[cfg(any(feature = "mock", feature = "cuda"))]
mod explicit_context_tests {
    use super::*;
    use audio2face3d::logging::{LogLevel, LogRecord, Logger, integration::LogScope};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct Sink(Mutex<Vec<String>>);
    impl Logger for Sink {
        fn log_level(&self) -> LogLevel {
            LogLevel::Trace
        }
        fn write_log(&self, _: LogLevel, record: LogRecord) {
            self.0.lock().unwrap().push(record.message);
        }
    }

    #[cfg(feature = "mock")]
    #[test]
    fn explicit_mock_factory_keeps_its_context_inside_foreign_scopes_and_threads() {
        let owner = Arc::new(Sink::default());
        let foreign = Arc::new(Sink::default());
        let context = Audio2Face3DContext::builder()
            .logger(owner.clone())
            .native_runtime(
                NativeRuntimeConfig::builder()
                    .search_policy(NativeSearchPolicy::ExplicitOnly)
                    .build()
                    .unwrap(),
            )
            .build();
        let other = Audio2Face3DContext::builder()
            .logger(foreign.clone())
            .build();
        let scope = LogScope::new(other.clone());
        let factory = scope
            .in_scope(|| {
                super::support::wait(audio2face3d::inference::Factory::prepare_with_context(
                    audio2face3d::inference::Config::builder(
                        audio2face3d::inference::BackendKind::Mock,
                    )
                    .build()
                    .unwrap(),
                    context.clone(),
                ))
            })
            .unwrap();
        std::thread::spawn(move || {
            scope.in_scope(|| {
                let options = audio2face3d::types::RequestOptions::builder(
                    audio2face3d::types::AudioFormat::MONO_16KHZ,
                )
                .build()
                .unwrap();
                let mut backend = super::support::wait(factory.start(options)).unwrap();
                super::support::wait(backend.close()).unwrap();
                super::support::wait(factory.release_prepared()).unwrap();
            })
        })
        .join()
        .unwrap();
        let client = LogScope::new(other.clone())
            .in_scope(|| {
                super::support::wait(audio2face3d::client::Client::direct_with_context(
                    audio2face3d::client::DirectConfig::builder(
                        audio2face3d::inference::Config::builder(
                            audio2face3d::inference::BackendKind::Mock,
                        )
                        .build()
                        .unwrap(),
                    )
                    .build()
                    .unwrap(),
                    context.clone(),
                ))
            })
            .unwrap();
        super::support::wait(client.shutdown()).unwrap();
        assert!(context.native_runtime_info().is_none());
        assert!(other.native_runtime_info().is_none());
        assert!(!owner.0.lock().unwrap().is_empty());
        assert!(foreign.0.lock().unwrap().is_empty());
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires explicit AUDIO2FACE3D_TEST_CUDA_ROOT; run in its own process"]
    fn explicit_device_context_ignores_foreign_scope_and_survives_thread_move() {
        let owner = Arc::new(Sink::default());
        let foreign = Arc::new(Sink::default());
        let mut runtime = NativeRuntimeConfig::builder()
            .cuda_root(std::env::var_os("AUDIO2FACE3D_TEST_CUDA_ROOT").unwrap())
            .search_policy(NativeSearchPolicy::ExplicitOnly);
        if let Some(root) = std::env::var_os("AUDIO2FACE3D_TEST_TENSORRT_ROOT") {
            runtime = runtime.tensorrt_root(root);
        }
        let context = Audio2Face3DContext::builder()
            .native_runtime(runtime.build().unwrap())
            .logger(owner.clone())
            .build();
        let other = Audio2Face3DContext::builder()
            .logger(foreign.clone())
            .native_runtime(
                NativeRuntimeConfig::builder()
                    .search_policy(NativeSearchPolicy::ExplicitOnly)
                    .build()
                    .unwrap(),
            )
            .build();
        let scope = LogScope::new(other.clone());
        let device = scope.in_scope(|| {
            let device =
                audio2face3d::cuda::GpuDevice::new_with_context(0, context.clone()).unwrap();
            audio2face3d::logging::integration::log(LogLevel::Debug, || {
                LogRecord::new("ambient restored")
            });
            device
        });
        let worker_scope = scope.clone();
        std::thread::spawn(move || {
            worker_scope.in_scope(|| {
                let stream = device.create_stream().unwrap();
                let buffer = device.from_slice(&[7_u32, 11], &stream).unwrap();
                let mut result = [0; 2];
                buffer.copy_to(&mut result, &stream).unwrap();
                assert_eq!(result, [7, 11]);
            })
        })
        .join()
        .unwrap();
        #[cfg(feature = "emotion")]
        scope.in_scope(|| {
            use audio2face3d::audio2emotion::{
                EmotionExecutorBundleCreationParameters, EmotionExecutorBundleFactory,
                EmotionExecutorCreationParameters, EmotionInteractiveExecutorCreationParameters,
                EmotionTrackResources,
                post_process::{
                    PostProcessData, PostProcessEmotionExecutorCreationParameters,
                    PostProcessEmotionInteractiveExecutorCreationParameters,
                    PostProcessEmotionInteractiveExecutorFactory, PostProcessParams,
                },
            };
            use audio2face3d::audio2x::{AudioAccumulator, FrameRate};
            let audio = Arc::new(AudioAccumulator::new(1600, 0).unwrap());
            audio.accumulate(&[0.0; 1600]).unwrap();
            audio.close().unwrap();
            let data = PostProcessData {
                inference_emotion_length: 1,
                output_emotion_length: 1,
                emotion_correspondence: vec![0],
            };
            let parameters = PostProcessParams {
                beginning_emotion: vec![0.0],
                preferred_emotion: vec![0.0],
                ..PostProcessParams::default()
            };
            let frame_rate = FrameRate::new(30, 1).unwrap();
            let bundle = super::support::wait(EmotionExecutorBundleFactory::load_with_context(
                EmotionExecutorBundleCreationParameters::PostProcess(
                    PostProcessEmotionExecutorCreationParameters {
                        common: EmotionExecutorCreationParameters {
                            tracks: vec![EmotionTrackResources {
                                audio: audio.clone(),
                            }],
                            device_ordinal: 0,
                        },
                        sample_rate: 16000,
                        input_strength: 1.0,
                        frame_rate,
                        post_process_data: data.clone(),
                        post_process_params: parameters.clone(),
                        preferred_emotions: vec![],
                    },
                ),
                context.clone(),
            ))
            .unwrap();
            let interactive = super::support::wait(
                PostProcessEmotionInteractiveExecutorFactory::load_with_context(
                    PostProcessEmotionInteractiveExecutorCreationParameters {
                        common: EmotionInteractiveExecutorCreationParameters {
                            audio,
                            preferred_emotions: None,
                            device_ordinal: 0,
                        },
                        sample_rate: 16000,
                        input_strength: 1.0,
                        frame_rate,
                        batch_size: 1,
                        post_process_data: data,
                        post_process_params: parameters,
                    },
                    context.clone(),
                ),
            )
            .unwrap();
            drop(bundle);
            drop(interactive);
        });
        assert!(context.native_runtime_info().is_some());
        assert!(other.native_runtime_info().is_none());
        assert!(
            owner
                .0
                .lock()
                .unwrap()
                .iter()
                .any(|message| message == "retained CUDA primary context")
        );
        assert_eq!(*foreign.0.lock().unwrap(), ["ambient restored"]);
    }
}

#[test]
fn building_and_observing_a_context_never_initializes_native_code() {
    let config = NativeRuntimeConfig::builder()
        .cuda_root(
            std::env::current_dir()
                .unwrap()
                .join("temp/nonexistent-sdk"),
        )
        .search_policy(NativeSearchPolicy::ExplicitOnly)
        .build()
        .unwrap();
    let context = Audio2Face3DContext::builder()
        .native_runtime(config)
        .build();
    assert!(context.native_runtime_info().is_none());
    assert!(context.clone().native_runtime_info().is_none());
}
#[cfg(feature = "mock")]
#[test]
fn mock_direct_ignores_unavailable_native_paths() {
    use audio2face3d::{
        client::{Client, DirectConfig},
        inference::{BackendKind, Config},
    };
    let config = NativeRuntimeConfig::builder()
        .search_policy(NativeSearchPolicy::ExplicitOnly)
        .build()
        .unwrap();
    let context = Audio2Face3DContext::builder()
        .native_runtime(config)
        .build();
    let client = support::wait(Client::direct_with_context(
        DirectConfig::builder(Config::builder(BackendKind::Mock).build().unwrap())
            .build()
            .unwrap(),
        context.clone(),
    ))
    .unwrap();
    assert!(context.native_runtime_info().is_none());
    support::wait(client.shutdown()).unwrap();
    assert!(context.native_runtime_info().is_none());
}
#[cfg(feature = "tensorrt")]
#[test]
#[ignore = "requires explicit AUDIO2FACE3D_TEST_CUDA_ROOT, AUDIO2FACE3D_TEST_TENSORRT_ROOT and AUDIO2FACE3D_TEST_ENGINE"]
fn explicit_and_lazy_initialization_share_context_resources_and_outlive_caller() {
    use audio2face3d::{cuda::GpuDevice, runtime::NativeRuntimeState, tensorrt::TensorRtSession};
    let config = NativeRuntimeConfig::builder()
        .cuda_root(std::env::var_os("AUDIO2FACE3D_TEST_CUDA_ROOT").unwrap())
        .tensorrt_root(std::env::var_os("AUDIO2FACE3D_TEST_TENSORRT_ROOT").unwrap())
        .search_policy(NativeSearchPolicy::ExplicitOnly)
        .build()
        .unwrap();
    let context = Audio2Face3DContext::builder()
        .native_runtime(config)
        .build();
    let clone = context.clone();
    assert_eq!(
        context.initialize_native().unwrap().state(),
        NativeRuntimeState::Loaded
    );
    assert_eq!(
        clone.native_runtime_info().unwrap().state(),
        NativeRuntimeState::Loaded
    );
    let device = GpuDevice::new_with_context(0, clone.clone()).unwrap();
    assert_eq!(
        context.native_runtime_info().unwrap().state(),
        NativeRuntimeState::DeviceReady
    );
    let engine = std::path::PathBuf::from(std::env::var_os("AUDIO2FACE3D_TEST_ENGINE").unwrap());
    let session = TensorRtSession::load(device.clone(), &engine).unwrap();
    assert!(
        session
            .native_runtime_info()
            .libraries()
            .iter()
            .any(|library| library.name() == "TensorRT" && library.runtime_version().is_some())
    );
    drop(context);
    drop(clone);
    let stream = device.create_stream().unwrap();
    let mut buffer = device.allocate::<u32>(2).unwrap();
    buffer.copy_from(&[7, 11], &stream).unwrap();
    let mut output = [0; 2];
    buffer.copy_to(&mut output, &stream).unwrap();
    assert_eq!(output, [7, 11]);
    assert!(session.environment().is_ok());
}

#[cfg(feature = "tensorrt")]
#[test]
#[ignore = "requires explicit SDK roots; run alone in a fresh test process"]
fn real_native_logger_reentry_and_concurrent_initialization() {
    use audio2face3d::{
        logging::{LogLevel, Logger},
        runtime::NativeRuntimeErrorKind,
    };
    use std::sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    struct Callback {
        context: Mutex<Option<Audio2Face3DContext>>,
        calls: AtomicUsize,
    }
    impl Logger for Callback {
        fn log_level(&self) -> LogLevel {
            LogLevel::Debug
        }
        fn write_log(&self, _: LogLevel, _: audio2face3d::logging::LogRecord) {
            let context = self.context.lock().unwrap().clone().unwrap();
            assert_eq!(
                context.initialize_native().unwrap_err().kind(),
                NativeRuntimeErrorKind::InitializationReentered
            );
            self.calls.fetch_add(1, Ordering::SeqCst);
        }
    }
    let callback = Arc::new(Callback {
        context: Mutex::new(None),
        calls: AtomicUsize::new(0),
    });
    let config = NativeRuntimeConfig::builder()
        .cuda_root(std::env::var_os("AUDIO2FACE3D_TEST_CUDA_ROOT").unwrap())
        .tensorrt_root(std::env::var_os("AUDIO2FACE3D_TEST_TENSORRT_ROOT").unwrap())
        .search_policy(NativeSearchPolicy::ExplicitOnly)
        .build()
        .unwrap();
    let context = Audio2Face3DContext::builder()
        .native_runtime(config)
        .logger(callback.clone())
        .build();
    *callback.context.lock().unwrap() = Some(context.clone());
    let barrier = Barrier::new(8);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| {
                barrier.wait();
                context.initialize_native().unwrap();
            });
        }
    });
    assert!(callback.calls.load(Ordering::SeqCst) > 0);
    // Break the test-only callback cycle before dropping application resources.
    callback.context.lock().unwrap().take();
    let count = callback.calls.load(Ordering::SeqCst);
    context.initialize_native().unwrap();
    assert_eq!(callback.calls.load(Ordering::SeqCst), count);
}
