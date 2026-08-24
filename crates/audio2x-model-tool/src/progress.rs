use hf_hub::progress::{DownloadEvent, FileProgress, FileStatus, ProgressEvent, ProgressHandler};
use std::collections::HashMap;
use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;
use std::time::{Duration, Instant};

const TERMINAL_INTERVAL: Duration = Duration::from_millis(200);
const LOG_INTERVAL: Duration = Duration::from_secs(5);

pub(crate) struct TerminalDownloadProgress {
    state: Mutex<State>,
    interactive: bool,
    output_enabled: bool,
}

struct State {
    total_files: usize,
    total_bytes: u64,
    aggregate_bytes: u64,
    bytes_per_sec: Option<f64>,
    files: HashMap<String, FileProgress>,
    last_render: Instant,
    previous_width: usize,
    started: bool,
    complete: bool,
}

impl TerminalDownloadProgress {
    pub(crate) fn new() -> Self {
        Self::with_settings(io::stderr().is_terminal(), true)
    }

    fn with_settings(interactive: bool, output_enabled: bool) -> Self {
        Self {
            state: Mutex::new(State {
                total_files: 0,
                total_bytes: 0,
                aggregate_bytes: 0,
                bytes_per_sec: None,
                files: HashMap::new(),
                last_render: Instant::now(),
                previous_width: 0,
                started: false,
                complete: false,
            }),
            interactive,
            output_enabled,
        }
    }

    pub(crate) fn failed(&self) {
        if let Ok(mut state) = self.state.lock()
            && state.started
            && !state.complete
        {
            state.render(
                self.interactive,
                self.output_enabled,
                "Download failed",
                true,
            );
        }
    }
}

impl ProgressHandler for TerminalDownloadProgress {
    fn on_progress(&self, event: &ProgressEvent) {
        let ProgressEvent::Download(event) = event else {
            return;
        };
        let state = match event {
            DownloadEvent::Start { .. } | DownloadEvent::Complete => self.state.lock().ok(),
            _ => self.state.try_lock().ok(),
        };
        let Some(mut state) = state else {
            return;
        };
        match event {
            DownloadEvent::Start {
                total_files,
                total_bytes,
            } => {
                state.total_files = *total_files;
                state.total_bytes = *total_bytes;
                state.started = true;
                state.render(self.interactive, self.output_enabled, "Downloading", false);
            }
            DownloadEvent::Progress { files } => {
                for file in files {
                    state.files.insert(file.filename.clone(), file.clone());
                }
                state.render_if_due(self.interactive, self.output_enabled);
            }
            DownloadEvent::AggregateProgress {
                bytes_completed,
                bytes_per_sec,
                ..
            } => {
                state.aggregate_bytes = *bytes_completed;
                state.bytes_per_sec = *bytes_per_sec;
                state.render_if_due(self.interactive, self.output_enabled);
            }
            DownloadEvent::Complete => {
                state.complete = true;
                state.aggregate_bytes = state.total_bytes;
                state.render(self.interactive, self.output_enabled, "Downloaded", true);
            }
        }
    }
}

impl State {
    fn render_if_due(&mut self, interactive: bool, output_enabled: bool) {
        let interval = if interactive {
            TERMINAL_INTERVAL
        } else {
            LOG_INTERVAL
        };
        if self.last_render.elapsed() >= interval {
            self.render(interactive, output_enabled, "Downloading", false);
        }
    }

    fn render(&mut self, interactive: bool, output_enabled: bool, label: &str, final_line: bool) {
        if !output_enabled {
            self.last_render = Instant::now();
            return;
        }
        let line = self.line(label);
        let mut stderr = io::stderr().lock();
        if interactive {
            let width = self.previous_width.max(line.len());
            let _ = write!(stderr, "\r{line:<width$}");
            if final_line {
                let _ = writeln!(stderr);
            }
            self.previous_width = if final_line { 0 } else { width };
        } else {
            let _ = writeln!(stderr, "{line}");
        }
        let _ = stderr.flush();
        self.last_render = Instant::now();
    }

    fn line(&self, label: &str) -> String {
        let file_bytes = self
            .files
            .values()
            .map(|file| file.bytes_completed)
            .sum::<u64>();
        let completed_bytes = file_bytes.max(self.aggregate_bytes).min(self.total_bytes);
        let completed_files = self
            .files
            .values()
            .filter(|file| file.status == FileStatus::Complete)
            .count();
        let percent = if self.total_bytes == 0 {
            0.0
        } else {
            completed_bytes as f64 * 100.0 / self.total_bytes as f64
        };
        let rate = self
            .bytes_per_sec
            .map(|value| format!(", {}/s", human_bytes(value as u64)))
            .unwrap_or_default();
        format!(
            "{label}: {percent:5.1}% {}/{} ({completed_files}/{} files){rate}",
            human_bytes(completed_bytes),
            human_bytes(self.total_bytes),
            self.total_files
        )
    }
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_binary_sizes() {
        assert_eq!(human_bytes(999), "999 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(5 * 1024 * 1024), "5.0 MiB");
    }

    #[test]
    fn accumulates_file_progress_for_a_snapshot() {
        let progress = TerminalDownloadProgress::with_settings(false, false);
        progress.on_progress(&ProgressEvent::Download(DownloadEvent::Start {
            total_files: 2,
            total_bytes: 100,
        }));
        progress.on_progress(&ProgressEvent::Download(DownloadEvent::Progress {
            files: vec![FileProgress {
                filename: "network.onnx".into(),
                bytes_completed: 40,
                total_bytes: 80,
                status: FileStatus::InProgress,
            }],
        }));
        let state = progress.state.lock().unwrap();
        assert_eq!(state.total_files, 2);
        assert!(state.line("Downloading").contains("40.0%"));
        assert!(state.line("Downloading").contains("0/2 files"));
    }
}
