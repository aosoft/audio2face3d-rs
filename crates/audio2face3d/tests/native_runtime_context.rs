#[cfg(feature = "mock")]
mod support;
use audio2face3d::{
    Audio2Face3DContext,
    runtime::{NativeRuntimeConfig, NativeSearchPolicy},
};

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
