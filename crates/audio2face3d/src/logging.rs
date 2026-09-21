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
/// An owned, typed field value. No formatting or serialization is required by the API.
#[derive(Clone, Debug, PartialEq)]
pub enum LogValue {
    String(String),
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
}
impl From<String> for LogValue {
    fn from(value: String) -> Self {
        Self::String(value)
    }
}
impl From<&str> for LogValue {
    fn from(value: &str) -> Self {
        Self::String(value.into())
    }
}
impl From<i64> for LogValue {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}
impl From<u64> for LogValue {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}
impl From<f64> for LogValue {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}
impl From<bool> for LogValue {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}
/// A message and optional application-defined fields, constructed after level filtering.
#[derive(Clone, Debug, PartialEq)]
pub struct LogRecord {
    pub message: String,
    pub fields: Vec<(String, LogValue)>,
}
impl LogRecord {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fields: Vec::new(),
        }
    }
    /// Adds a field, replacing its value when the key already exists.
    pub fn field(mut self, key: impl Into<String>, value: impl Into<LogValue>) -> Self {
        let key = key.into();
        let value = value.into();
        if let Some((_, old)) = self.fields.iter_mut().find(|(k, _)| *k == key) {
            *old = value;
        } else {
            self.fields.push((key, value));
        }
        self
    }
}
impl From<String> for LogRecord {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}
impl From<&str> for LogRecord {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}
pub trait Logger: Send + Sync {
    fn log_level(&self) -> LogLevel;
    fn write_log(&self, level: LogLevel, message: LogRecord);
    fn log<F>(&self, level: LogLevel, message: F)
    where
        Self: Sized,
        F: FnOnce() -> LogRecord,
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
    fn write_log(&self, level: LogLevel, message: LogRecord) {
        (**self).write_log(level, message)
    }
}
#[derive(Default, Debug)]
pub struct NoopLogger;
impl Logger for NoopLogger {
    fn log_level(&self) -> LogLevel {
        LogLevel::Off
    }
    fn write_log(&self, _: LogLevel, _: LogRecord) {}
}
pub mod integration;
mod scope;

#[cfg(any(
    feature = "mock",
    feature = "native",
    feature = "client-grpc",
    feature = "grpc-server"
))]
pub(crate) mod operation;
