use super::{LogLevel, LogRecord, integration::LogScope};
use crate::types::{ErrorKind, Result};
use std::time::Instant;

/// Internal boundary observation. It never records arbitrary remote error text or credentials.
pub(crate) struct Operation {
    scope: LogScope,
    name: &'static str,
    source: &'static str,
    started: Instant,
    pub stage: &'static str,
    finished: bool,
    pub cleanup_failed: bool,
}
impl Operation {
    pub fn new(name: &'static str, source: &'static str) -> Self {
        Self {
            scope: LogScope::capture(),
            name,
            source,
            started: Instant::now(),
            stage: "initialization",
            finished: false,
            cleanup_failed: false,
        }
    }
    pub fn finish<T>(&mut self, result: &Result<T>) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.emit(result.as_ref().err().map(|e| e.kind()));
    }
    fn emit(&self, error: Option<ErrorKind>) {
        let (mut level, outcome) = match error {
            None => (LogLevel::Info, "success"),
            Some(ErrorKind::Cancelled | ErrorKind::ShuttingDown) => (LogLevel::Info, "cancelled"),
            Some(ErrorKind::InvalidInput | ErrorKind::Unsupported) => (LogLevel::Warn, "rejected"),
            Some(ErrorKind::QueueFull | ErrorKind::LimitExceeded) => (LogLevel::Warn, "capacity"),
            Some(ErrorKind::DeadlineExceeded) => (LogLevel::Warn, "timeout"),
            Some(ErrorKind::Transport | ErrorKind::IncompleteResponse) => {
                (LogLevel::Warn, "transport_error")
            }
            _ => (LogLevel::Error, "internal_error"),
        };
        if self.cleanup_failed {
            level = LogLevel::Error;
        }
        self.scope.log(level, || {
            let mut record = LogRecord::new(self.name)
                .field("source", self.source)
                .field("outcome", outcome)
                .field("stage", self.stage)
                .field("elapsed_us", self.started.elapsed().as_micros() as u64);
            if self.cleanup_failed {
                record = record.field("cleanup_failed", true);
            }
            if let Some(error) = error {
                record = record.field("error_kind", format!("{error:?}"));
            }
            record
        });
    }
}
impl Drop for Operation {
    fn drop(&mut self) {
        if !self.finished {
            self.emit(Some(if std::thread::panicking() {
                ErrorKind::Inference
            } else {
                ErrorKind::Cancelled
            }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Audio2Face3DContext, logging::Logger, types::Error};
    use std::sync::{Arc, Mutex};
    #[derive(Default)]
    struct Sink(Mutex<Vec<(LogLevel, LogRecord)>>);
    impl Logger for Sink {
        fn log_level(&self) -> LogLevel {
            LogLevel::Trace
        }
        fn write_log(&self, level: LogLevel, record: LogRecord) {
            self.0.lock().unwrap().push((level, record));
        }
    }
    #[test]
    fn cleanup_failure_overrides_expected_cancellation_and_terminal_is_once() {
        let sink = Arc::new(Sink::default());
        let scope = LogScope::new(Audio2Face3DContext::builder().logger(sink.clone()).build())
            .field("client_request_id", 8_u64);
        scope.in_scope(|| {
            let mut observation = Operation::new("request finished", "test_source");
            observation.stage = "cleanup";
            observation.cleanup_failed = true;
            let result: Result<()> =
                Err(Error::new(ErrorKind::Cancelled, "private error contents"));
            observation.finish(&result);
            observation.finish(&result);
        });
        let logs = sink.0.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].0, LogLevel::Error);
        assert!(
            logs[0]
                .1
                .fields
                .contains(&("cleanup_failed".into(), true.into()))
        );
        assert!(
            logs[0]
                .1
                .fields
                .contains(&("source".into(), "test_source".into()))
        );
        assert!(
            logs[0]
                .1
                .fields
                .contains(&("client_request_id".into(), 8_u64.into()))
        );
        assert!(!format!("{logs:?}").contains("private error contents"));
    }
}
