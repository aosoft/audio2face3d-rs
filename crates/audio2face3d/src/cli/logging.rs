mod worker;
use audio2face3d::logging::{LogLevel, LogRecord, LogValue, Logger};
use clap::{Args, ValueEnum};
use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use tracing_subscriber::{Layer, prelude::*};

#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub enum Format {
    #[default]
    Text,
    Json,
}
#[derive(Clone, Debug, Default, Args)]
pub struct LogArgs {
    /// JSONL queue capacity. Each record is limited to 64 KiB of owned payload.
    #[arg(long, global = true, default_value = "1024", value_parser = clap::value_parser!(u16).range(1..))]
    pub log_queue_capacity: u16,
    /// JSONL full queue behavior; drop reports losses, wait blocks the producer.
    #[arg(long, global = true, value_enum, default_value = "drop")]
    pub log_overflow: worker::Overflow,

    /// Text forwards application logs to tracing; json records typed application JSONL.
    #[arg(long, global = true, value_enum, default_value = "text")]
    pub log_format: Format,
    /// Append application logs to this file. Dependency diagnostics use stderr in json mode.
    #[arg(long, global = true)]
    pub log_file: Option<PathBuf>,
}
#[derive(Clone)]
struct Filter {
    default: LogLevel,
    targets: Vec<(String, LogLevel)>,
}
fn level(value: &str) -> Option<LogLevel> {
    Some(match value.to_ascii_lowercase().as_str() {
        "trace" => LogLevel::Trace,
        "debug" => LogLevel::Debug,
        "info" => LogLevel::Info,
        "warn" => LogLevel::Warn,
        "error" => LogLevel::Error,
        "off" => LogLevel::Off,
        _ => return None,
    })
}
impl Filter {
    fn from_env() -> Result<Self, String> {
        let value = match std::env::var("RUST_LOG") {
            Ok(v) => v,
            Err(std::env::VarError::NotPresent) => "info".into(),
            Err(_) => return Err("RUST_LOG must be Unicode".into()),
        };
        Self::parse(&value)
    }
    fn parse(value: &str) -> Result<Self, String> {
        let mut result = Self {
            default: LogLevel::Info,
            targets: vec![],
        };
        for item in value.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(level) = level(item) {
                result.default = level;
                continue;
            }
            let (target, threshold) = match item.split_once('=') {
                Some((t, v)) => (t, level(v).ok_or("invalid RUST_LOG level")?),
                None => (item, LogLevel::Trace),
            };
            if target.is_empty()
                || !target
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_:-".contains(&c))
            {
                return Err(
                    "RUST_LOG supports levels and targets; span/field expressions are unsupported"
                        .into(),
                );
            }
            result.targets.push((target.into(), threshold));
        }
        Ok(result)
    }
    fn minimum(&self) -> LogLevel {
        self.targets
            .iter()
            .map(|(_, v)| *v)
            .fold(self.default, LogLevel::min)
    }
    fn accepts(&self, level: LogLevel, target: &str) -> bool {
        let threshold = self
            .targets
            .iter()
            .filter(|(p, _)| target.starts_with(p))
            .max_by_key(|(p, _)| p.len())
            .map_or(self.default, |(_, v)| *v);
        level != LogLevel::Off && level >= threshold
    }
}
fn event_level(level: &tracing::Level) -> LogLevel {
    match *level {
        tracing::Level::TRACE => LogLevel::Trace,
        tracing::Level::DEBUG => LogLevel::Debug,
        tracing::Level::INFO => LogLevel::Info,
        tracing::Level::WARN => LogLevel::Warn,
        tracing::Level::ERROR => LogLevel::Error,
    }
}
struct Writer {
    output: Box<dyn Write + Send>,
    error: Option<String>,
}
#[derive(Clone)]
struct SharedWriter(Arc<Mutex<Writer>>);
impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut writer = self.0.lock().unwrap();
        match writer.output.write(bytes) {
            Ok(n) => Ok(n),
            Err(e) => {
                writer.error = Some(e.to_string());
                Err(e)
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        let mut writer = self.0.lock().unwrap();
        match writer.output.flush() {
            Ok(()) => Ok(()),
            Err(e) => {
                writer.error = Some(e.to_string());
                Err(e)
            }
        }
    }
}
fn json_line(level: LogLevel, record: LogRecord, received: SystemTime) -> io::Result<Vec<u8>> {
    let fields: serde_json::Map<String, serde_json::Value> = record
        .fields
        .into_iter()
        .map(|(key, value)| {
            let value = match value {
                LogValue::String(v) => v.into(),
                LogValue::I64(v) => v.into(),
                LogValue::U64(v) => v.into(),
                LogValue::Bool(v) => v.into(),
                LogValue::F64(v) => serde_json::Number::from_f64(v)
                    .map(serde_json::Value::Number)
                    .unwrap_or_else(|| v.to_string().into()),
            };
            (key, value)
        })
        .collect();
    let timestamp = received
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let mut bytes = serde_json::to_vec(
        &serde_json::json!({"timestamp_unix_ms":timestamp as u64,"level":format!("{level:?}").to_ascii_lowercase(),"message":record.message,"fields":fields}),
    )?;
    bytes.push(b'\n');
    Ok(bytes)
}
struct OutputLogger {
    sender: Option<worker::Sender>,
    filter: Filter,
    format: Format,
    writer: SharedWriter,
}
impl Logger for OutputLogger {
    fn log_level(&self) -> LogLevel {
        self.filter.minimum()
    }
    fn write_log(&self, level: LogLevel, record: LogRecord) {
        let source = record
            .fields
            .iter()
            .find_map(|(k, v)| match (k.as_str(), v) {
                ("source", LogValue::String(v)) => Some(v.as_str()),
                _ => None,
            })
            .unwrap_or("");
        if !self.filter.accepts(level, source) {
            return;
        }
        if let Some(sender) = &self.sender {
            sender.send(level, record);
            return;
        }
        if self.format == Format::Json {
            let result = json_line(level, record, SystemTime::now()).and_then(|bytes| {
                let mut writer = self.writer.0.lock().unwrap();
                let result = writer.output.write_all(&bytes);
                if let Err(error) = &result {
                    writer.error = Some(error.to_string());
                }
                result
            });
            if let Err(error) = result {
                self.writer.0.lock().unwrap().error = Some(error.to_string());
            }
        } else {
            macro_rules! emit { ($level:expr)=> { tracing::event!(target: "audio2face3d_log", $level, message=%record.message, fields=?record.fields) } }
            match level {
                LogLevel::Trace => emit!(tracing::Level::TRACE),
                LogLevel::Debug => emit!(tracing::Level::DEBUG),
                LogLevel::Info => emit!(tracing::Level::INFO),
                LogLevel::Warn => emit!(tracing::Level::WARN),
                LogLevel::Error => emit!(tracing::Level::ERROR),
                LogLevel::Off => {}
            }
        }
    }
}
pub struct Logging {
    worker: Mutex<Option<worker::Worker>>,
    pub logger: Arc<dyn Logger>,
    writer: SharedWriter,
}
impl Logging {
    pub fn start(args: &LogArgs) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let filter = Filter::from_env()?;
        let output: Box<dyn Write + Send> = match &args.log_file {
            Some(path) => Box::new(OpenOptions::new().create(true).append(true).open(path)?),
            None => Box::new(io::stderr()),
        };
        let writer = SharedWriter(Arc::new(Mutex::new(Writer {
            output,
            error: None,
        })));
        let trace_writer = if args.log_format == Format::Text {
            writer.clone()
        } else {
            SharedWriter(Arc::new(Mutex::new(Writer {
                output: Box::new(io::stderr()),
                error: None,
            })))
        };
        let trace_filter = filter.clone();
        let layer = tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(move || trace_writer.clone())
            .with_filter(tracing_subscriber::filter::filter_fn(move |meta| {
                meta.target() == "audio2face3d_log"
                    || trace_filter.accepts(event_level(meta.level()), meta.target())
            }));
        // Only the executable configures the process subscriber, never Logger or library code.
        tracing::subscriber::set_global_default(tracing_subscriber::registry().with(layer))?;
        let worker = if args.log_format == Format::Json {
            Some(worker::Worker::start(
                writer.clone(),
                usize::from(args.log_queue_capacity),
                args.log_overflow,
            )?)
        } else {
            None
        };
        let sender = worker.as_ref().map(worker::Worker::sender);
        Ok(Self {
            worker: Mutex::new(worker),
            logger: Arc::new(OutputLogger {
                sender,
                filter,
                format: args.log_format,
                writer: writer.clone(),
            }),
            writer,
        })
    }
    pub fn finish(&self) -> io::Result<()> {
        if let Some(mut worker) = self.worker.lock().unwrap().take() {
            let dropped = worker.finish(std::time::Duration::from_secs(5))?;
            if dropped > 0 {
                eprintln!("logging: dropped {dropped} records (queue full, oversized, or closed)");
            }
            return Ok(());
        }
        self.writer.clone().flush()?;
        if let Some(error) = &self.writer.0.lock().unwrap().error {
            return Err(io::Error::other(error.clone()));
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn levels_targets_and_invalid_syntax() {
        let f = Filter::parse("off,audio2face3d=debug").unwrap();
        assert_eq!(f.minimum(), LogLevel::Debug);
        assert!(f.accepts(LogLevel::Debug, "audio2face3d::inference"));
        assert!(!f.accepts(LogLevel::Error, "tonic"));
        assert!(Filter::parse("crate[span{x=1}]=info").is_err());
        assert!(Filter::parse("crate=no-level").is_err());
        assert_eq!(Filter::parse("off").unwrap().minimum(), LogLevel::Off);
    }
    #[test]
    fn json_preserves_types_and_escaping() {
        let r = LogRecord::new("quote\"\n日本語")
            .field("id", 1_u64)
            .field("id", 2_u64)
            .field("ok", true)
            .field("negative", -2_i64)
            .field("ratio", 0.5_f64)
            .field("nan", f64::NAN)
            .field("inf", f64::INFINITY);
        let bytes = json_line(
            LogLevel::Info,
            r,
            UNIX_EPOCH + std::time::Duration::from_millis(123),
        )
        .unwrap();
        assert_eq!(bytes.iter().filter(|&&c| c == b'\n').count(), 1);
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["fields"]["id"], 2);
        assert_eq!(v["fields"]["ok"], true);
        assert_eq!(v["fields"]["ratio"], 0.5);
        assert_eq!(v["fields"]["negative"], -2);
        assert_eq!(v["timestamp_unix_ms"], 123);
        assert!(v["fields"]["nan"].is_string());
        assert!(v["fields"]["inf"].is_string());
        assert_eq!(v["message"], "quote\"\n日本語");
    }
}

#[cfg(test)]
mod output_tests {
    use super::*;
    struct Broken;
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("write failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::other("flush failure"))
        }
    }
    #[test]
    fn output_failures_are_reported_at_finish() {
        let writer = SharedWriter(Arc::new(Mutex::new(Writer {
            output: Box::new(Broken),
            error: None,
        })));
        let logger = Arc::new(OutputLogger {
            sender: None,
            filter: Filter::parse("trace").unwrap(),
            format: Format::Json,
            writer: writer.clone(),
        });
        logger.log(LogLevel::Info, || "event".into());
        assert!(
            writer
                .0
                .lock()
                .unwrap()
                .error
                .as_ref()
                .unwrap()
                .contains("write failure")
        );
        let logging = Logging {
            logger,
            writer,
            worker: Mutex::new(None),
        };
        assert!(logging.finish().is_err());
    }
    #[test]
    fn empty_fields_are_json_object() {
        let bytes = json_line(LogLevel::Error, LogRecord::new("empty"), UNIX_EPOCH).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["fields"], serde_json::json!({}));
    }
}

impl Drop for Logging {
    fn drop(&mut self) {
        if self.worker.get_mut().unwrap().is_some()
            && let Err(error) = self.finish()
        {
            eprintln!("logging shutdown: {error}");
        }
    }
}
