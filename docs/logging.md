# Logging

[Client library](library.md) · [Packages and features](features.md)

## Library interface

Both libraries send their own diagnostics through the `Logger` in `Audio2Face3DContext`, including server RPC diagnostics. They do not emit tracing events or spans, configure a subscriber, or require tracing types in their public API. An unconfigured Context uses `NoopLogger`, including constructors without an explicit Context. Native SDK messages emitted outside Rust are separate.

`Logger` uses standard Rust types. `log(level, closure)` checks `log_level()` before invoking the `FnOnce` closure, so disabled messages and fields are never constructed. `Off` never emits a record. `write_log(level, record)` receives an owned `LogRecord`; direct callers of `write_log` are responsible for filtering.

```rust
use std::sync::Arc;
use audio2face3d::{Audio2Face3DContext, logging::{LogLevel, LogRecord, Logger}};

struct ApplicationLogger;
impl Logger for ApplicationLogger {
    fn log_level(&self) -> LogLevel { LogLevel::Info }
    fn write_log(&self, level: LogLevel, record: LogRecord) {
        eprintln!("{level:?}: {} {:?}", record.message, record.fields);
    }
}

let logger: Arc<dyn Logger> = Arc::new(ApplicationLogger);
logger.log(LogLevel::Info, || {
    LogRecord::new("request completed")
        .field("rpc_id", 42_u64)
        .field("elapsed_ms", 12.5_f64)
        .field("success", true)
});
let context = Audio2Face3DContext::builder().logger(logger).build();
```

`LogValue` holds `String`, `i64`, `u64`, `f64`, or `bool`. A record with no fields does not allocate a field buffer. `field(key, value)` replaces a previous value for the same key. Build expensive messages and fields inside the closure. Records can be moved into an application-owned worker queue without copying their strings.

Context clones share application resources. Internal scope propagation preserves the Logger through standard threads, future poll/drop and native cleanup. RPC IDs are request-local fields, not mutable state on a shared Context. Context destruction does not flush a custom writer or wait for its worker; the embedding application owns that shutdown policy. For server cleanup timeouts, await `CleanupCompletion` before stopping the writer.

Third-party crates such as tonic and h2 can still use tracing internally. Applications may collect those diagnostics with their own subscriber. They are not automatically forwarded into the custom Logger. There is no reverse tracing-to-Logger bridge.

## Executable output

Both executables accept these global options:

| Option | Behavior |
|---|---|
| `--log-format text` | Default. Forward Logger records to tracing and render text. Fields are a single debug-formatted attribute. |
| `--log-format json` | Write application Logger records as typed JSONL through a bounded worker. |
| `--log-file PATH` | Append application logs to the file; otherwise use stderr. Opening a file unsuccessfully fails startup. |
| `--log-queue-capacity N` | JSONL queue capacity, default 1024, range 1–65535. |
| `--log-overflow drop` | Default. Do not wait for queue space; report dropped records at shutdown. |
| `--log-overflow wait` | Wait for JSONL queue space, which can delay inference. |

For example:

```sh
audio2face3d-server --model models/mark/model.json --log-format json --log-file server.jsonl
```

JSONL records have `timestamp_unix_ms`, `level`, `message`, and a `fields` object. Time is recorded on the producer before enqueueing. Numeric and boolean fields retain their JSON types; non-finite floats become strings such as `NaN`, `inf`, and `-inf`. Messages containing newlines are escaped onto one physical line. Help, command results and progress messages are not written into the log file.

```json
{"timestamp_unix_ms":1789985567488,"level":"info","message":"completed","fields":{"rpc_id":2,"source":"audio2face3d_server::service"}}
```

In JSON mode, dependency tracing diagnostics use stderr separately and do not enter the application JSONL file. When no file is specified, both outputs share stderr; specify `--log-file` when a consumer needs a stream containing only JSONL. Text forwarding does not provide arbitrary fields as independently typed tracing fields; choose the custom JSONL output for that requirement.

`RUST_LOG` accepts levels and target prefixes, for example `info,audio2face3d_server=debug`. Its default is `info`. Unsupported span/field expressions fail startup. The minimum configured level gates record construction; target filtering uses the optional `source` field after construction. Records without `source` use the default level. Dependency tracing events use their own targets. A source-specific rejection can therefore happen after generating a record. `RUST_LOG=off` disables both outputs.

The JSONL worker serializes and writes on a standard thread. The producer still constructs the message and fields. Records exceeding 64 KiB of owned payload capacity are dropped even in wait mode; the queue capacity and record limit bound retained payload, not total process memory. The default drop policy is intentionally lossy under load. The CLI reports the dropped count on stderr. A slow or stalled writer can block producers in wait mode.

After inference/server cleanup, the CLI closes the queue, drains pending records and flushes. Writer errors fail the command. The worker has a five-second shutdown deadline; expiry is reported as an error and remaining output is not guaranteed. It does not forcibly interrupt a blocked OS write. Text output remains synchronous. No performance guarantee is made for arbitrary writers.

## Migration

The former `write_log(LogLevel, String)` signature is replaced by `write_log(LogLevel, LogRecord)`. Change a sink to use `record.message` and `record.fields`. A message-only closure can return `LogRecord::new(message)` or `message.into()`. Closures can now move captured values because they implement `FnOnce`.

The old library `tracing` feature and compatibility behavior that inherited the caller's subscriber are removed. Select `cli` for executable tracing output, or inject your own Logger into a library Context. There is no implicit subscriber installation in a Logger implementation. The executables explicitly own subscriber initialization and report initialization conflicts.
