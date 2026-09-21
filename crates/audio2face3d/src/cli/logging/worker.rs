use super::{LogLevel, LogRecord, LogValue, SharedWriter, Timezone, json_line};
use std::{
    io,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, SyncSender},
    },
    thread::JoinHandle,
    time::{Duration, SystemTime},
};
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
pub enum Overflow {
    #[default]
    Drop,
    Wait,
}
const MAX_RECORD_BYTES: usize = 64 * 1024;
struct Entry {
    level: LogLevel,
    record: LogRecord,
    received: SystemTime,
}
#[derive(Clone)]
pub(super) struct Sender {
    tx: Arc<Mutex<Option<SyncSender<Entry>>>>,
    dropped: Arc<AtomicU64>,
    overflow: Overflow,
}
impl Sender {
    pub fn send(&self, level: LogLevel, record: LogRecord) {
        let received = SystemTime::now();
        let bytes = record
            .message
            .capacity()
            .saturating_add(
                record
                    .fields
                    .capacity()
                    .saturating_mul(std::mem::size_of::<(String, LogValue)>()),
            )
            .saturating_add(record.fields.iter().fold(0_usize, |sum, (key, value)| {
                sum.saturating_add(key.capacity())
                    .saturating_add(match value {
                        LogValue::String(s) => s.capacity(),
                        _ => 0,
                    })
            }));
        if bytes > MAX_RECORD_BYTES {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let tx = self.tx.lock().unwrap().clone();
        let Some(tx) = tx else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        let entry = Entry {
            level,
            record,
            received,
        };
        let failed = match self.overflow {
            Overflow::Drop => tx.try_send(entry).is_err(),
            Overflow::Wait => tx.send(entry).is_err(),
        };
        if failed {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}
pub(super) struct Worker {
    sender: Sender,
    done: mpsc::Receiver<io::Result<()>>,
    thread: Option<JoinHandle<()>>,
}
impl Worker {
    pub fn start(
        writer: SharedWriter,
        capacity: usize,
        overflow: Overflow,
        timezone: Timezone,
    ) -> io::Result<Self> {
        let (tx, rx) = mpsc::sync_channel::<Entry>(capacity);
        let (done_tx, done) = mpsc::channel();
        let sender = Sender {
            tx: Arc::new(Mutex::new(Some(tx))),
            dropped: Arc::new(AtomicU64::new(0)),
            overflow,
        };
        let thread = std::thread::Builder::new()
            .name("a2f-jsonl".into())
            .spawn(move || {
                let result = (|| {
                    for entry in rx {
                        let bytes = json_line(entry.level, entry.record, entry.received, timezone)?;
                        let mut output = writer.0.lock().unwrap();
                        output.output.write_all(&bytes)?;
                        output.output.flush()?;
                    }
                    writer.0.lock().unwrap().output.flush()
                })();
                let _ = done_tx.send(result);
            })?;
        Ok(Self {
            sender,
            done,
            thread: Some(thread),
        })
    }
    pub fn sender(&self) -> Sender {
        self.sender.clone()
    }
    pub fn finish(&mut self, timeout: Duration) -> io::Result<u64> {
        self.sender.tx.lock().unwrap().take();
        let result = self
            .done
            .recv_timeout(timeout)
            .map_err(|error| io::Error::other(format!("log worker did not finish: {error}")))?;
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| io::Error::other("log worker panicked"))?;
        }
        result?;
        Ok(self.sender.dropped.load(Ordering::Relaxed))
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.sender.tx.lock().unwrap().take();
    }
}
#[cfg(test)]
mod tests {
    use super::super::Writer;
    use super::*;
    use std::io::Write;
    struct Blocked {
        started: Option<mpsc::Sender<()>>,
        release: mpsc::Receiver<()>,
    }
    impl Write for Blocked {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if let Some(tx) = self.started.take() {
                tx.send(()).unwrap();
                self.release.recv().unwrap();
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn bounded_queue_drops_and_shutdown_has_deadline() {
        let (started_tx, started) = mpsc::channel();
        let (release, rx) = mpsc::channel();
        let writer = SharedWriter(Arc::new(Mutex::new(Writer {
            output: Box::new(Blocked {
                started: Some(started_tx),
                release: rx,
            }),
            error: None,
        })));
        let mut worker = Worker::start(writer, 1, Overflow::Drop, Timezone::Utc).unwrap();
        let sender = worker.sender();
        sender.send(LogLevel::Info, LogRecord::new("first"));
        started.recv().unwrap();
        sender.send(LogLevel::Info, LogRecord::new("queued"));
        sender.send(LogLevel::Info, LogRecord::new("full"));
        sender.send(
            LogLevel::Info,
            LogRecord::new("x".repeat(MAX_RECORD_BYTES + 1)),
        );
        assert_eq!(sender.dropped.load(Ordering::Relaxed), 2);
        assert!(worker.finish(Duration::from_millis(5)).is_err());
        release.send(()).unwrap();
        assert_eq!(worker.finish(Duration::from_secs(2)).unwrap(), 2);
    }
    struct Capture(Arc<Mutex<Vec<u8>>>);
    impl Write for Capture {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn wait_policy_drains_all_records_and_preserves_fields() {
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let writer = SharedWriter(Arc::new(Mutex::new(Writer {
            output: Box::new(Capture(bytes.clone())),
            error: None,
        })));
        let mut worker = Worker::start(writer, 1, Overflow::Wait, Timezone::Utc).unwrap();
        let sender = worker.sender();
        for id in 0..100_u64 {
            sender.send(LogLevel::Info, LogRecord::new("record").field("id", id));
        }
        assert_eq!(worker.finish(Duration::from_secs(2)).unwrap(), 0);
        let text = String::from_utf8(bytes.lock().unwrap().clone()).unwrap();
        assert_eq!(text.lines().count(), 100);
        for (id, line) in text.lines().enumerate() {
            let value: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(value["fields"]["id"], id);
        }
    }
}

#[cfg(test)]
mod performance {
    use super::super::Writer;
    use super::*;
    use std::{io::Write, time::Instant};
    struct Delay(Duration);
    impl Write for Delay {
        fn write(&mut self, b: &[u8]) -> io::Result<usize> {
            std::thread::sleep(self.0);
            Ok(b.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    #[ignore = "manual output scheduling measurement"]
    fn compare_sync_and_worker() {
        for delay in [Duration::ZERO, Duration::from_millis(1)] {
            let count = 100;
            let start = Instant::now();
            let mut output = Delay(delay);
            for id in 0..count {
                output
                    .write_all(
                        &json_line(
                            LogLevel::Info,
                            LogRecord::new("short").field("id", id as u64),
                            SystemTime::now(),
                            Timezone::Utc,
                        )
                        .unwrap(),
                    )
                    .unwrap();
            }
            let sync = start.elapsed();
            let writer = SharedWriter(Arc::new(Mutex::new(Writer {
                output: Box::new(Delay(delay)),
                error: None,
            })));
            let mut worker = Worker::start(writer, count, Overflow::Wait, Timezone::Utc).unwrap();
            let sender = worker.sender();
            let start = Instant::now();
            for id in 0..count {
                sender.send(
                    LogLevel::Info,
                    LogRecord::new("short").field("id", id as u64),
                );
            }
            let producer = start.elapsed();
            assert_eq!(worker.finish(Duration::from_secs(5)).unwrap(), 0);
            eprintln!(
                "delay={delay:?} count={count} sync={sync:?} worker_producer={producer:?} worker_total={:?}",
                start.elapsed()
            );
        }
        let calls = std::cell::Cell::new(0);
        let start = Instant::now();
        use audio2face3d::logging::Logger;
        for _ in 0..100_000 {
            audio2face3d::logging::NoopLogger.log(LogLevel::Info, || {
                calls.set(calls.get() + 1);
                LogRecord::new("disabled")
            });
        }
        assert_eq!(calls.get(), 0);
        eprintln!("disabled_100000={:?}; generated=0", start.elapsed());
    }
    struct Fail;
    impl Write for Fail {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("worker output failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    #[test]
    fn worker_failure_is_returned_to_owner() {
        let writer = SharedWriter(Arc::new(Mutex::new(Writer {
            output: Box::new(Fail),
            error: None,
        })));
        let mut worker = Worker::start(writer, 1, Overflow::Wait, Timezone::Utc).unwrap();
        worker
            .sender()
            .send(LogLevel::Error, LogRecord::new("failure"));
        assert!(
            worker
                .finish(Duration::from_secs(1))
                .unwrap_err()
                .to_string()
                .contains("worker output failure")
        );
    }
}
