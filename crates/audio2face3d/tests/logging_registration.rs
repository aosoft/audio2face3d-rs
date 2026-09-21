#![cfg(feature = "tracing")]
use audio2face3d::{
    Audio2Face3DContext,
    logging::{LogLevel, Logger, integration::LogScope},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
struct Sink(AtomicUsize);
impl Logger for Sink {
    fn log_level(&self) -> LogLevel {
        LogLevel::Trace
    }
    fn write_log(&self, _: LogLevel, _: audio2face3d::logging::LogRecord) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn shared_callsite() {
    tracing::info!("callsite first encountered outside its sole scoped dispatcher");
}
#[test]
fn first_registration_without_current_dispatcher_does_not_disable_scoped_logging() {
    let sink = Arc::new(Sink(AtomicUsize::new(0)));
    let scope = LogScope::new(Audio2Face3DContext::builder().logger(sink.clone()).build());
    // No global subscriber and only one application logger in this process.
    std::thread::spawn(shared_callsite).join().unwrap();
    scope.in_scope(shared_callsite);
    assert_eq!(sink.0.load(Ordering::SeqCst), 1);
}
