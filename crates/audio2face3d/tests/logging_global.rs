#![cfg(feature = "tracing")]
use audio2face3d::{
    Audio2Face3DContext,
    logging::{LogLevel, Logger, integration::LogScope},
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tracing_subscriber::{Layer, prelude::*};
struct Global(Arc<AtomicUsize>);
impl<S: tracing::Subscriber> Layer<S> for Global {
    fn on_event(&self, _: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
struct Local(AtomicUsize);
impl Logger for Local {
    fn log_level(&self) -> LogLevel {
        LogLevel::Trace
    }
    fn write_log(&self, _: LogLevel, _: String) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
#[test]
fn explicit_noop_overrides_existing_global_and_legacy_preserves_it() {
    let global = Arc::new(AtomicUsize::new(0));
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(Global(global.clone())),
    )
    .unwrap();
    LogScope::capture().in_scope(|| tracing::info!("legacy"));
    let cost = AtomicUsize::new(0);
    LogScope::new(Audio2Face3DContext::default())
        .in_scope(|| tracing::error!(cost = cost.fetch_add(1, Ordering::SeqCst), "suppressed"));
    let local = Arc::new(Local(AtomicUsize::new(0)));
    LogScope::new(Audio2Face3DContext::builder().logger(local.clone()).build())
        .in_scope(|| tracing::info!("local"));
    assert_eq!(global.load(Ordering::SeqCst), 1);
    assert_eq!(local.0.load(Ordering::SeqCst), 1);
    assert_eq!(cost.load(Ordering::SeqCst), 0);
}
