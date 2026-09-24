//! Bounded GUI sink and explicit host fanout for the existing logging interface.
use audio2face3d::logging::{LogLevel, LogRecord, LogValue, Logger};
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
    time::SystemTime,
};

#[derive(Clone, Debug)]
pub struct Entry {
    pub time: SystemTime,
    pub level: LogLevel,
    pub record: LogRecord,
}
pub struct GuiLogger {
    sender: SyncSender<Entry>,
    dropped: Arc<AtomicU64>,
    level: LogLevel,
}
pub struct LogBuffer {
    receiver: Receiver<Entry>,
    pub entries: VecDeque<Entry>,
    dropped: Arc<AtomicU64>,
    limit: usize,
}
pub fn channel(queue: usize, history: usize, level: LogLevel) -> (Arc<GuiLogger>, LogBuffer) {
    let (sender, receiver) = mpsc::sync_channel(queue.max(1));
    let dropped = Arc::new(AtomicU64::new(0));
    (
        Arc::new(GuiLogger {
            sender,
            dropped: dropped.clone(),
            level,
        }),
        LogBuffer {
            receiver,
            entries: VecDeque::new(),
            dropped,
            limit: history.max(1),
        },
    )
}
fn shorten(value: &mut String, limit: usize) {
    if value.len() > limit {
        let mut end = limit;
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        value.truncate(end);
        value.push('…');
        value.shrink_to_fit();
    }
}
impl Logger for GuiLogger {
    fn log_level(&self) -> LogLevel {
        self.level
    }
    fn write_log(&self, level: LogLevel, mut record: LogRecord) {
        if level == LogLevel::Off || level < self.level {
            return;
        }
        // Bound a single oversized record as well as the number of records.
        shorten(&mut record.message, 4096);
        record.fields.truncate(16);
        record.fields.shrink_to_fit();
        for (key, value) in &mut record.fields {
            shorten(key, 128);
            if let LogValue::String(value) = value {
                shorten(value, 256);
            }
        }
        if self
            .sender
            .try_send(Entry {
                time: SystemTime::now(),
                level,
                record,
            })
            .is_err()
        {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}
impl LogBuffer {
    pub fn drain(&mut self) {
        for entry in self.receiver.try_iter().take(2048) {
            if self.entries.len() == self.limit {
                self.entries.pop_front();
            }
            self.entries.push_back(entry);
        }
    }
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

/// Host sinks run on the producer thread; enqueue if an engine needs affinity.
pub struct FanoutLogger {
    sinks: Vec<Arc<dyn Logger>>,
}
impl FanoutLogger {
    pub fn new(sinks: Vec<Arc<dyn Logger>>) -> Self {
        Self { sinks }
    }
}
impl Logger for FanoutLogger {
    fn log_level(&self) -> LogLevel {
        self.sinks
            .iter()
            .map(|s| s.log_level())
            .min()
            .unwrap_or(LogLevel::Off)
    }
    fn write_log(&self, level: LogLevel, record: LogRecord) {
        for sink in &self.sinks {
            if level != LogLevel::Off && level >= sink.log_level() {
                sink.write_log(level, record.clone());
            }
        }
    }
}

#[cfg(feature = "ui-egui")]
pub struct LogView {
    pub filter: String,
    pub source: String,
    pub level: LogLevel,
    pub follow: bool,
}
#[cfg(feature = "ui-egui")]
impl Default for LogView {
    fn default() -> Self {
        Self {
            filter: String::new(),
            source: String::new(),
            level: LogLevel::Info,
            follow: true,
        }
    }
}
#[cfg(feature = "ui-egui")]
impl LogView {
    pub fn show(&mut self, ui: &mut egui::Ui, buffer: &LogBuffer) {
        ui.horizontal(|ui| {
            ui.label("Filter");
            ui.text_edit_singleline(&mut self.filter);
            ui.label("Source");
            ui.text_edit_singleline(&mut self.source);
            egui::ComboBox::from_id_salt("log-level")
                .selected_text(format!("{:?}", self.level))
                .show_ui(ui, |ui| {
                    for level in [
                        LogLevel::Trace,
                        LogLevel::Debug,
                        LogLevel::Info,
                        LogLevel::Warn,
                        LogLevel::Error,
                    ] {
                        ui.selectable_value(&mut self.level, level, format!("{level:?}"));
                    }
                });
            ui.checkbox(&mut self.follow, "Follow");
            ui.label(format!("Dropped: {}", buffer.dropped()));
        });
        let filter = self.filter.to_lowercase();
        let source = self.source.to_lowercase();
        let lines: Vec<_> = buffer
            .entries
            .iter()
            .filter(|e| e.level >= self.level)
            .filter_map(|entry| {
                let fields = format!("{:?}", entry.record.fields);
                if !fields.to_lowercase().contains(&source) {
                    return None;
                }
                let message = format!(
                    "{}.{:03} {:?} {} {}",
                    entry
                        .time
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs(),
                    entry
                        .time
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .subsec_millis(),
                    entry.level,
                    entry.record.message,
                    fields
                );
                message.to_lowercase().contains(&filter).then_some(message)
            })
            .collect();
        if ui.button("Copy filtered logs").clicked() {
            ui.ctx().copy_text(lines.join("\n"));
        }
        egui::ScrollArea::vertical()
            .max_height(160.)
            .stick_to_bottom(self.follow)
            .show_rows(
                ui,
                ui.text_style_height(&egui::TextStyle::Monospace),
                lines.len(),
                |ui, range| {
                    for i in range {
                        ui.monospace(&lines[i]);
                    }
                },
            );
    }
}
