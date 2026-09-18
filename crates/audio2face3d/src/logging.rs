//! Application-owned logging, without a public dependency on a logging crate.
use std::sync::Arc;
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Off,
}
pub trait Logger: Send + Sync {
    fn log_level(&self) -> LogLevel;
    fn write_log(&self, level: LogLevel, message: String);
    fn log<F>(&self, level: LogLevel, message: F)
    where
        Self: Sized,
        F: Fn() -> String,
    {
        if level != LogLevel::Off && level >= self.log_level() {
            self.write_log(level, message());
        }
    }
}
impl<T: Logger + ?Sized> Logger for Arc<T> {
    fn log_level(&self) -> LogLevel {
        (**self).log_level()
    }
    fn write_log(&self, level: LogLevel, message: String) {
        (**self).write_log(level, message)
    }
}
#[derive(Default, Debug)]
pub struct NoopLogger;
impl Logger for NoopLogger {
    fn log_level(&self) -> LogLevel {
        LogLevel::Off
    }
    fn write_log(&self, _: LogLevel, _: String) {}
}
#[cfg(feature = "tracing")]
mod bridge;
pub mod integration;
mod scope;
