#[cfg(any(feature = "mock", feature = "native"))]
mod support;
use audio2face3d::{
    Audio2Face3DContext,
    logging::{LogLevel, Logger},
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
#[derive(Default)]
struct Sink {
    level: AtomicUsize,
    lines: Mutex<Vec<(LogLevel, String)>>,
}
impl Logger for Sink {
    fn log_level(&self) -> LogLevel {
        match self.level.load(Ordering::SeqCst) {
            0 => LogLevel::Trace,
            1 => LogLevel::Info,
            2 => LogLevel::Error,
            _ => LogLevel::Off,
        }
    }
    fn write_log(&self, level: LogLevel, text: String) {
        self.lines.lock().unwrap().push((level, text));
    }
}
fn context(sink: Arc<Sink>) -> Audio2Face3DContext {
    Audio2Face3DContext::builder().logger(sink).build()
}
#[test]
fn lazy_log_contract_and_owned_message() {
    let sink = Arc::new(Sink::default());
    sink.level.store(2, Ordering::SeqCst);
    let count = std::rc::Rc::new(std::cell::Cell::new(0));
    for level in [
        LogLevel::Trace,
        LogLevel::Debug,
        LogLevel::Info,
        LogLevel::Warn,
        LogLevel::Off,
    ] {
        sink.log(level, || {
            count.set(count.get() + 1);
            "unexpected".into()
        });
    }
    assert_eq!(count.get(), 0);
    assert!(sink.lines.lock().unwrap().is_empty());
    let text = String::from("move without copying");
    let ptr = text.as_ptr() as usize;
    let slot = std::cell::RefCell::new(Some(text));
    let erased: Arc<dyn Logger> = sink.clone();
    erased.log(LogLevel::Error, || {
        count.set(count.get() + 1);
        slot.borrow_mut().take().unwrap()
    });
    assert_eq!(count.get(), 1);
    assert_eq!(sink.lines.lock().unwrap()[0].1.as_ptr() as usize, ptr);
    sink.level.store(3, Ordering::SeqCst);
    erased.log(LogLevel::Error, || panic!("Off must suppress generation"));
}
#[test]
fn context_clones_share_resources_without_requiring_a_runtime() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Audio2Face3DContext>();
    let sink = Arc::new(Sink::default());
    let ctx = context(sink.clone());
    let clone = ctx.clone();
    assert!(Arc::ptr_eq(ctx.logger(), clone.logger()));
    drop(ctx);
    clone.logger().log(LogLevel::Info, || "still owned".into());
    assert_eq!(sink.lines.lock().unwrap().len(), 1);
    assert_eq!(
        Audio2Face3DContext::default().logger().log_level(),
        LogLevel::Off
    );
}
#[cfg(feature = "tracing")]
mod bridge {
    use super::*;
    use audio2face3d::logging::integration::LogScope;
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll, Waker},
    };
    fn expensive(counter: &AtomicUsize) -> usize {
        counter.fetch_add(1, Ordering::SeqCst)
    }
    fn emit(counter: &AtomicUsize) {
        tracing::info!(cost = expensive(counter), "dynamic");
    }
    #[test]
    fn dynamic_level_suppresses_expression_before_formatting() {
        let sink = Arc::new(Sink::default());
        let scope = LogScope::new(context(sink.clone()));
        let count = AtomicUsize::new(0);
        scope.in_scope(|| {
            sink.level.store(2, Ordering::SeqCst);
            emit(&count);
            assert_eq!(count.load(Ordering::SeqCst), 0);
            sink.level.store(1, Ordering::SeqCst);
            emit(&count);
            sink.level.store(2, Ordering::SeqCst);
            emit(&count);
        });
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(sink.lines.lock().unwrap().len(), 1);
    }
    struct PendingLog;
    impl Future for PendingLog {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<()> {
            tracing::info!("poll");
            Poll::Pending
        }
    }
    impl Drop for PendingLog {
        fn drop(&mut self) {
            tracing::warn!("drop");
        }
    }
    #[test]
    fn alternating_poll_drop_and_thread_scopes_do_not_cross() {
        let a = Arc::new(Sink::default());
        let b = Arc::new(Sink::default());
        let sa = LogScope::new(context(a.clone()));
        let sb = LogScope::new(context(b.clone()));
        let mut fa = Box::pin(sa.wrap_future(PendingLog));
        let mut fb = Box::pin(sb.wrap_future(PendingLog));
        let mut cx = Context::from_waker(Waker::noop());
        assert!(fa.as_mut().poll(&mut cx).is_pending());
        assert!(fb.as_mut().poll(&mut cx).is_pending());
        drop(fa);
        std::thread::spawn(move || drop(fb)).join().unwrap();
        for sink in [a, b] {
            let lines = sink.lines.lock().unwrap();
            assert_eq!(lines.len(), 2);
            assert!(lines[0].1.contains("poll"));
            assert!(lines[1].1.contains("drop"));
        }
    }
    #[test]
    fn span_fields_and_messages_are_bounded() {
        let sink = Arc::new(Sink::default());
        let scope = LogScope::new(context(sink.clone()));
        scope.in_scope(|| {
            let span = tracing::info_span!("request", id = 123, field = "値".repeat(5000));
            let _entered = span.enter();
            tracing::info!(payload=%"文".repeat(30000),"bounded");
        });
        let lines = sink.lines.lock().unwrap();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].1.contains("id=123"));
        assert!(lines[0].1.len() <= 16384);
    }
    #[test]
    fn logger_reentry_is_suppressed() {
        struct Reentrant(AtomicUsize);
        impl Logger for Reentrant {
            fn log_level(&self) -> LogLevel {
                LogLevel::Trace
            }
            fn write_log(&self, _: LogLevel, _: String) {
                self.0.fetch_add(1, Ordering::SeqCst);
                tracing::warn!("reentrant");
            }
        }
        let logger = Arc::new(Reentrant(AtomicUsize::new(0)));
        let ctx = Audio2Face3DContext::builder()
            .logger(logger.clone())
            .build();
        LogScope::new(ctx).in_scope(|| tracing::info!("outer"));
        assert_eq!(logger.0.load(Ordering::SeqCst), 1);
    }
}
#[cfg(feature = "mock")]
#[test]
fn direct_and_factory_keep_logger_after_initialization() {
    let a = Arc::new(Sink::default());
    let b = Arc::new(Sink::default());
    let config =
        audio2face3d::inference::Config::builder(audio2face3d::inference::BackendKind::Mock)
            .build()
            .unwrap();
    let client = support::wait(audio2face3d::client::Client::direct_with_context(
        audio2face3d::client::DirectConfig::builder(config.clone())
            .build()
            .unwrap(),
        context(a.clone()),
    ))
    .unwrap();
    let factory = support::wait(audio2face3d::inference::Factory::prepare_with_context(
        config,
        context(b.clone()),
    ))
    .unwrap();
    let mut engine = support::wait(
        factory.start(
            audio2face3d::types::RequestOptions::builder(
                audio2face3d::types::AudioFormat::MONO_16KHZ,
            )
            .build()
            .unwrap(),
        ),
    )
    .unwrap();
    support::wait(engine.close()).unwrap();
    support::wait(factory.release_prepared()).unwrap();
    drop(engine);
    drop(factory);
    support::wait(client.shutdown()).unwrap();
    drop(client);
    assert!(
        a.lines
            .lock()
            .unwrap()
            .iter()
            .any(|(_, m)| m.contains("inference prepared"))
    );
    assert!(
        b.lines
            .lock()
            .unwrap()
            .iter()
            .any(|(_, m)| m.contains("closing inference"))
    );
}

#[cfg(feature = "native")]
#[test]
#[ignore = "requires AUDIO2FACE3D_LOG_MODEL and native SDKs"]
fn native_logger_reaches_model_worker_and_cleanup_without_tokio() {
    let path = std::env::var_os("AUDIO2FACE3D_LOG_MODEL").expect("set AUDIO2FACE3D_LOG_MODEL");
    let sink = Arc::new(Sink::default());
    let ctx = context(sink.clone());
    let model = audio2face3d::Model::load_with_context(&path, ctx.clone()).unwrap();
    assert!(Arc::ptr_eq(model.context().logger(), ctx.logger()));
    drop(model);
    let config =
        audio2face3d::inference::Config::builder(audio2face3d::inference::BackendKind::Regression)
            .model(path)
            .build()
            .unwrap();
    let factory = support::wait(audio2face3d::inference::Factory::prepare_with_context(
        config,
        ctx.clone(),
    ))
    .unwrap();
    drop(ctx);
    let mut engine = support::wait(
        factory.start(
            audio2face3d::types::RequestOptions::builder(
                audio2face3d::types::AudioFormat::MONO_16KHZ,
            )
            .build()
            .unwrap(),
        ),
    )
    .unwrap();
    let chunk = audio2face3d::types::InputChunk::new(
        audio2face3d::types::PcmBuffer::from_vec(vec![0; 3200]).unwrap(),
        vec![],
    );
    support::wait(engine.push(chunk)).unwrap();
    support::wait(engine.finish()).unwrap();
    let cancel = audio2face3d::inference::Cancellation::default();
    while support::wait(engine.next_frame(&cancel)).unwrap().is_some() {}
    support::wait(engine.close()).unwrap();
    drop(engine);
    support::wait(factory.release_prepared()).unwrap();
    drop(factory);
    let lines = sink.lines.lock().unwrap();
    assert!(
        lines
            .iter()
            .any(|(_, m)| m.contains("loading model descriptor"))
    );
    assert!(
        lines
            .iter()
            .any(|(_, m)| m.contains("retained CUDA primary context"))
    );
    assert!(
        lines
            .iter()
            .any(|(_, m)| m.contains("regression model and host solver ready"))
    );
    assert!(
        lines
            .iter()
            .any(|(_, m)| m.contains("release prepared inference"))
    );
}
