//! One terminal observation per accepted RPC, including dropped handler futures.
use audio2face3d::logging::{LogLevel, LogRecord, integration::LogScope};
use std::time::Instant;
use tonic::{Code, Status};

pub(crate) fn classification(code: Code) -> (LogLevel, &'static str) {
    match code {
        Code::Ok => (LogLevel::Info, "success"),
        Code::Cancelled => (LogLevel::Info, "cancelled"),
        Code::InvalidArgument
        | Code::Unauthenticated
        | Code::PermissionDenied
        | Code::NotFound
        | Code::AlreadyExists
        | Code::OutOfRange
        | Code::Unimplemented
        | Code::FailedPrecondition => (LogLevel::Warn, "rejected"),
        Code::DeadlineExceeded => (LogLevel::Warn, "timeout"),
        Code::ResourceExhausted => (LogLevel::Warn, "capacity"),
        Code::Unavailable => (LogLevel::Warn, "unavailable"),
        _ => (LogLevel::Error, "internal_error"),
    }
}
pub(crate) struct RequestLog {
    scope: LogScope,
    started: Instant,
    pub stage: &'static str,
    pub input_audio_bytes: u64,
    pub output_batches: u64,
    pub cleanup_failed: bool,
    finished: bool,
    shutdown: tokio_util::sync::CancellationToken,
}
impl RequestLog {
    pub fn new(shutdown: tokio_util::sync::CancellationToken) -> Self {
        Self {
            scope: LogScope::capture(),
            started: Instant::now(),
            stage: "metadata",
            input_audio_bytes: 0,
            output_batches: 0,
            cleanup_failed: false,
            finished: false,
            shutdown,
        }
    }
    pub fn finish<T>(&mut self, result: &Result<T, Status>) {
        if self.finished {
            return;
        }
        self.finished = true;
        if result.is_ok() {
            self.stage = "complete";
        }
        let code = result.as_ref().err().map_or(Code::Ok, Status::code);
        self.emit(code);
    }
    fn emit(&self, code: Code) {
        let (mut level, outcome) = if self.shutdown.is_cancelled()
            && matches!(code, Code::Unavailable | Code::Cancelled)
        {
            (LogLevel::Info, "shutdown")
        } else {
            classification(code)
        };
        if self.cleanup_failed {
            level = LogLevel::Error;
        }
        self.scope.log(level, || {
            LogRecord::new(match code {
                Code::Ok => "completed",
                Code::Cancelled => "cancelled",
                _ => "failed",
            })
            .field("source", module_path!())
            .field("outcome", outcome)
            .field("stage", self.stage)
            .field("code", format!("{code:?}"))
            .field("elapsed_us", self.started.elapsed().as_micros() as u64)
            .field("input_audio_bytes", self.input_audio_bytes)
            .field("output_batches_enqueued", self.output_batches)
            .field("cleanup_failed", self.cleanup_failed)
        });
    }
}
impl Drop for RequestLog {
    fn drop(&mut self) {
        if !self.finished {
            self.emit(if std::thread::panicking() {
                Code::Internal
            } else {
                Code::Cancelled
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use audio2face3d::{Audio2Face3DContext, logging::Logger};
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
    fn terminal_levels_drop_and_secrets() {
        for (code, level, outcome) in [
            (Code::Ok, LogLevel::Info, "success"),
            (Code::Cancelled, LogLevel::Info, "cancelled"),
            (Code::InvalidArgument, LogLevel::Warn, "rejected"),
            (Code::Internal, LogLevel::Error, "internal_error"),
        ] {
            let sink = Arc::new(Sink::default());
            let scope = LogScope::new(Audio2Face3DContext::builder().logger(sink.clone()).build())
                .field("rpc_id", 7_u64);
            scope.in_scope(|| {
                let mut observation = RequestLog::new(tokio_util::sync::CancellationToken::new());
                observation.stage = "input_header";
                let result = if code == Code::Ok {
                    Ok(())
                } else {
                    Err(Status::new(code, "secret-must-not-appear"))
                };
                observation.finish(&result);
                observation.finish(&result);
            });
            let logs = sink.0.lock().unwrap();
            assert_eq!(logs.len(), 1);
            assert_eq!(logs[0].0, level);
            assert!(
                logs[0]
                    .1
                    .fields
                    .contains(&("outcome".into(), outcome.into()))
            );
            assert!(!format!("{logs:?}").contains("secret-must-not-appear"));
        }
        let sink = Arc::new(Sink::default());
        let scope = LogScope::new(Audio2Face3DContext::builder().logger(sink.clone()).build());
        scope.in_scope(|| {
            let _observation = RequestLog::new(tokio_util::sync::CancellationToken::new());
        });
        let logs = sink.0.lock().unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].0, LogLevel::Info);
        assert_eq!(logs[0].1.message, "cancelled");
    }
}
