use audio2face3d::logging::{LogLevel, Logger};
pub struct StderrLogger {
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
impl StderrLogger {
    pub fn from_env() -> Result<Self, &'static str> {
        let value = match std::env::var("RUST_LOG") {
            Ok(v) => v,
            Err(std::env::VarError::NotPresent) => "info".into(),
            Err(_) => return Err("RUST_LOG must be Unicode"),
        };
        Self::parse(&value)
    }
    fn parse(value: &str) -> Result<Self, &'static str> {
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
                Some((target, value)) => (target, level(value).ok_or("invalid RUST_LOG level")?),
                None => (item, LogLevel::Trace),
            };
            if target.is_empty()
                || !target
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"_:-".contains(&c))
            {
                return Err(
                    "RUST_LOG supports levels and targets; span/field expressions are unsupported",
                );
            }
            result.targets.push((target.into(), threshold));
        }
        Ok(result)
    }
}
impl Logger for StderrLogger {
    fn log_level(&self) -> LogLevel {
        self.targets
            .iter()
            .map(|(_, v)| *v)
            .fold(self.default, LogLevel::min)
    }
    fn write_log(&self, level: LogLevel, message: String) {
        let target = message.split_whitespace().next().unwrap_or("");
        let threshold = self
            .targets
            .iter()
            .filter(|(prefix, _)| target.starts_with(prefix))
            .max_by_key(|(prefix, _)| prefix.len())
            .map_or(self.default, |(_, v)| *v);
        if level != LogLevel::Off && level >= threshold {
            eprintln!("[{level:?}] {message}");
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn level_target_and_unsupported_syntax() {
        let logger = StderrLogger::parse("error,audio2face3d=debug").unwrap();
        assert_eq!(logger.default, LogLevel::Error);
        assert_eq!(logger.log_level(), LogLevel::Debug);
        assert!(StderrLogger::parse("crate[span{field=value}]=trace").is_err());
        assert!(StderrLogger::parse("crate=not-a-level").is_err());
    }
}
