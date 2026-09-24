use audio2face3d::logging::{LogLevel, LogRecord, Logger};
use audio2face3d_gui::logging::{FanoutLogger, channel};
#[test]
fn saturation_is_nonblocking_and_history_is_bounded() {
    let (logger, mut buffer) = channel(2, 1, LogLevel::Info);
    for i in 0..4 {
        logger.write_log(LogLevel::Info, LogRecord::new(format!("{i}")));
    }
    assert_eq!(buffer.dropped(), 2);
    buffer.drain();
    assert_eq!(buffer.entries.len(), 1);
    assert_eq!(buffer.entries[0].record.message, "1");
}
#[test]
fn fanout_preserves_host_record_and_filters_each_sink() {
    let (first, mut a) = channel(8, 8, LogLevel::Debug);
    let (second, mut b) = channel(8, 8, LogLevel::Error);
    let logger = FanoutLogger::new(vec![first, second]);
    logger.write_log(
        LogLevel::Info,
        LogRecord::new("info").field("source", "test"),
    );
    logger.write_log(LogLevel::Error, LogRecord::new("error"));
    a.drain();
    b.drain();
    assert_eq!(a.entries.len(), 2);
    assert_eq!(b.entries.len(), 1);
    assert_eq!(b.entries[0].record.message, "error");
}
