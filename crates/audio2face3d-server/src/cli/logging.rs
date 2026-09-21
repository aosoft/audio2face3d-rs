mod worker;
use audio2face3d::logging::{LogLevel, LogRecord, LogValue, Logger};
use clap::{Args, ValueEnum};
#[cfg(test)]
use std::time::UNIX_EPOCH;
use std::{
    fs::OpenOptions,
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::SystemTime,
};
use tracing_subscriber::{Layer, prelude::*};

#[derive(Clone, Copy, Debug, Default, ValueEnum, PartialEq, Eq)]
pub enum Timezone {
    #[default]
    Utc,
    Local,
}
impl Timezone {
    fn datetime(self, time: SystemTime) -> chrono::DateTime<chrono::FixedOffset> {
        let utc: chrono::DateTime<chrono::Utc> = time.into();
        match self {
            Self::Utc => utc.fixed_offset(),
            Self::Local => utc.with_timezone(&chrono::Local).fixed_offset(),
        }
    }
    fn timestamp(self, time: SystemTime) -> String {
        self.datetime(time)
            .to_rfc3339_opts(chrono::SecondsFormat::Millis, false)
    }
}
impl tracing_subscriber::fmt::time::FormatTime for Timezone {
    fn format_time(
        &self,
        writer: &mut tracing_subscriber::fmt::format::Writer<'_>,
    ) -> std::fmt::Result {
        write!(writer, "{}", self.timestamp(SystemTime::now()))
    }
}

fn file_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        // Readers may observe the live log; other writers and deletion are excluded.
        const FILE_SHARE_READ: u32 = 1;
        options.share_mode(FILE_SHARE_READ);
    }
    options.append(true);
    options
}
fn open_log_file(path: &std::path::Path, create_new: bool) -> io::Result<std::fs::File> {
    let mut options = file_options();
    if create_new {
        options.create_new(true);
    } else {
        options.create(true);
    }
    let file = options.open(path)?;
    // Advisory on Unix: cooperating writers must acquire the same lock.
    #[cfg(not(windows))]
    file.try_lock().map_err(io::Error::other)?;
    Ok(file)
}

fn open_output(args: &LogArgs) -> io::Result<Box<dyn Write + Send>> {
    match (&args.log_file, &args.log_dir) {
        (Some(_), Some(_)) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "--log-file and --log-dir are mutually exclusive",
        )),
        (Some(path), None) => Ok(Box::new(open_log_file(path, false)?)),
        (None, Some(dir)) => {
            std::fs::create_dir_all(dir)?;
            let stamp = args
                .log_timezone
                .datetime(SystemTime::now())
                .format("%Y%m%dT%H%M%S%z")
                .to_string();
            let extension = if args.log_format == Format::Json {
                "jsonl"
            } else {
                "log"
            };
            for sequence in 0..1000 {
                let suffix = if sequence == 0 {
                    String::new()
                } else {
                    format!("-{sequence}")
                };
                let name = format!("{}-{stamp}{suffix}.{extension}", env!("CARGO_PKG_NAME"));
                match open_log_file(&dir.join(name), true) {
                    Ok(file) => return Ok(Box::new(file)),
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                    Err(error) => return Err(error),
                }
            }
            Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not allocate a unique log filename",
            ))
        }
        (None, None) => Ok(Box::new(io::stderr())),
    }
}

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
    /// Create a timestamped log file in this directory (created if missing).
    #[arg(long, global = true, conflicts_with = "log_file")]
    pub log_dir: Option<PathBuf>,
    /// Time zone for log timestamps and generated filenames.
    #[arg(long, global = true, value_enum, default_value = "utc")]
    pub log_timezone: Timezone,
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
        match writer.output.write(bytes).and_then(|n| {
            writer.output.flush()?;
            Ok(n)
        }) {
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
fn json_line(
    level: LogLevel,
    record: LogRecord,
    received: SystemTime,
    timezone: Timezone,
) -> io::Result<Vec<u8>> {
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
    let timestamp = timezone.timestamp(received);
    let mut bytes = serde_json::to_vec(
        &serde_json::json!({"timestamp":timestamp,"level":format!("{level:?}").to_ascii_lowercase(),"message":record.message,"fields":fields}),
    )?;
    bytes.push(b'\n');
    Ok(bytes)
}
struct OutputLogger {
    sender: Option<worker::Sender>,
    filter: Filter,
    format: Format,
    timezone: Timezone,
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
            let result =
                json_line(level, record, SystemTime::now(), self.timezone).and_then(|bytes| {
                    let mut writer = self.writer.0.lock().unwrap();
                    let result = writer
                        .output
                        .write_all(&bytes)
                        .and_then(|()| writer.output.flush());
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
        let output = open_output(args)?;
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
            .with_timer(args.log_timezone)
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
                args.log_timezone,
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
                timezone: args.log_timezone,
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
    fn file_allows_reading_and_excludes_other_writers() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../temp/logging-directory-work/sharing");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!(
            "{}-{}.log",
            env!("CARGO_PKG_NAME"),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut file = open_log_file(&path, true).unwrap();
        file.write_all(b"visible").unwrap();
        file.flush().unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"visible");
        assert!(open_log_file(&path, false).is_err());
        assert!(open_log_file(&path, true).is_err());
        #[cfg(windows)]
        assert!(OpenOptions::new().append(true).open(&path).is_err());
        drop(file);
        let file = open_log_file(&path, false).unwrap();
        drop(file);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn flush_occurs_per_record_and_errors_propagate() {
        struct CountFlush(Arc<std::sync::atomic::AtomicUsize>, bool);
        impl Write for CountFlush {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if self.1 {
                    Err(io::Error::other("flush failed"))
                } else {
                    Ok(())
                }
            }
        }
        for fails in [false, true] {
            let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let writer = SharedWriter(Arc::new(Mutex::new(Writer {
                output: Box::new(CountFlush(count.clone(), fails)),
                error: None,
            })));
            let mut worker =
                worker::Worker::start(writer.clone(), 8, worker::Overflow::Wait, Timezone::Utc)
                    .unwrap();
            for _ in 0..3 {
                worker.sender().send(LogLevel::Info, LogRecord::new("test"));
            }
            let result = worker.finish(std::time::Duration::from_secs(2));
            assert_eq!(result.is_err(), fails);
            assert_eq!(
                count.load(std::sync::atomic::Ordering::SeqCst),
                if fails { 1 } else { 4 }
            );
            count.store(0, std::sync::atomic::Ordering::SeqCst);
            assert_eq!(writer.clone().write_all(b"text").is_err(), fails);
            assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert_eq!(writer.0.lock().unwrap().error.is_some(), fails);
        }
    }

    #[test]
    fn timezone_and_directory_outputs() {
        let time = UNIX_EPOCH + std::time::Duration::from_millis(123);
        let local = Timezone::Local.timestamp(time);
        let parsed = chrono::DateTime::parse_from_rfc3339(&local).unwrap();
        let expected: chrono::DateTime<chrono::Local> = time.into();
        assert_eq!(parsed.timestamp_millis(), 123);
        assert_eq!(
            parsed.offset().local_minus_utc(),
            expected.offset().local_minus_utc()
        );
        for timezone in [Timezone::Utc, Timezone::Local] {
            let bytes = json_line(LogLevel::Info, LogRecord::new("test"), time, timezone).unwrap();
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["timestamp"], timezone.timestamp(time));
            let mut text = String::new();
            use tracing_subscriber::fmt::time::FormatTime;
            timezone
                .format_time(&mut tracing_subscriber::fmt::format::Writer::new(&mut text))
                .unwrap();
            chrono::DateTime::parse_from_rfc3339(&text).unwrap();
        }
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../temp/logging-directory-work/tests")
            .join(format!(
                "{}-{}",
                env!("CARGO_PKG_NAME"),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        for format in [Format::Json, Format::Text] {
            for timezone in [Timezone::Utc, Timezone::Local] {
                let dir = root.join(format!("{format:?}-{timezone:?}"));
                let args = LogArgs {
                    log_dir: Some(dir.clone()),
                    log_format: format,
                    log_timezone: timezone,
                    ..Default::default()
                };
                for _ in 0..2 {
                    let mut output = open_output(&args).unwrap();
                    output.write_all(b"test").unwrap();
                    output.flush().unwrap();
                }
                let files: Vec<_> = std::fs::read_dir(&dir)
                    .unwrap()
                    .map(Result::unwrap)
                    .collect();
                assert_eq!(files.len(), 2);
                for file in files {
                    let name = file.file_name().into_string().unwrap();
                    assert!(name.starts_with(env!("CARGO_PKG_NAME")));
                    assert!(name.ends_with(if format == Format::Json {
                        ".jsonl"
                    } else {
                        ".log"
                    }));
                    assert!(!name.contains(':'));
                    let stamp = name
                        .strip_prefix(concat!(env!("CARGO_PKG_NAME"), "-"))
                        .unwrap()
                        .split('.')
                        .next()
                        .unwrap();
                    let parsed =
                        chrono::DateTime::parse_from_str(&stamp[..20], "%Y%m%dT%H%M%S%z").unwrap();
                    if timezone == Timezone::Utc {
                        assert_eq!(parsed.offset().local_minus_utc(), 0);
                    }
                    assert_eq!(std::fs::read(file.path()).unwrap(), b"test");
                }
                let invalid = LogArgs {
                    log_file: Some(root.join("both.log")),
                    ..args
                };
                assert!(open_output(&invalid).is_err());
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

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
            Timezone::Utc,
        )
        .unwrap();
        assert_eq!(bytes.iter().filter(|&&c| c == b'\n').count(), 1);
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["fields"]["id"], 2);
        assert_eq!(v["fields"]["ok"], true);
        assert_eq!(v["fields"]["ratio"], 0.5);
        assert_eq!(v["fields"]["negative"], -2);
        assert_eq!(v["timestamp"], "1970-01-01T00:00:00.123+00:00");
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
            timezone: Timezone::Utc,
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
        let bytes = json_line(
            LogLevel::Error,
            LogRecord::new("empty"),
            UNIX_EPOCH,
            Timezone::Utc,
        )
        .unwrap();
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
