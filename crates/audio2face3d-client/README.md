# audio2face3d-client

Runtime-independent session control for Audio2Face inference, using the Rust-owned types in `audio2face3d-types`.

Choose a constructor, then use the same Client/Session API for direct inference and gRPC inference. Public request and output data are Rust-owned types; generated protocol types remain inside the transport adapter.

## Features and initialization

| Feature | Available mode | Requirements |
| --- | --- | --- |
| none (default) | Shared API types only | Standard library and shared types |
| `direct` | `Client::direct(DirectConfig)`, diagnostic mock | No async runtime required |
| `runtime` (includes direct) | Native Regression and optional Audio2Emotion | CUDA, TensorRT, model files; no async runtime required |
| `server` | `Client::server(ServerConfig)` | A caller-owned, driven Tokio runtime with I/O and time enabled |

Features can be combined. Native initialization with `runtime,server` still works without creating or entering Tokio.

Native initialization (await this on any standard-Future executor):

```rust
use audio2face3d_client::{Client, DirectConfig, InferenceConfig, BackendKind};
let client = Client::direct(DirectConfig {
    engine: InferenceConfig {
        backend: BackendKind::Regression,
        model: Some("models/mark/model.json".into()),
        // emotion_model: Some("models/emotion/model.json".into()),
        ..Default::default()
    },
    max_executions: 2,
    ..Default::default()
}).await?;
```

Remote initialization:

```rust
use audio2face3d_client::{Client, ServerConfig};
let mut config = ServerConfig::new("http://127.0.0.1:52000");
config.runtime = Some(runtime.handle().clone());
let client = Client::server(config).await?;
```

Without an explicit Handle, server initialization uses Tokio's current Handle or returns `RuntimeUnavailable`. Connection failures are returned by initialization; RPC failures are reported by each Session. A current-thread runtime must continue to be driven while common Futures are waiting. Keep the selected runtime alive until `client.shutdown().await` completes. Dropping that runtime aborts its tasks and terminates affected requests with an error; a transport failure may be observed first.

After either constructor, use `client.start(options)?.split()`, drive input sends and output receives concurrently, await input finish, consume output through Completed, and await closed. [Rust integration tests](tests/modes.rs) use exactly the same application function for both modes.

## Session contract

- Cloned `Client` handles share request limits, buffer budgets and backend resources. `start` synchronously registers a local request; it does not wait for model loading, an execution slot or a response header.
- `Session::split` returns `Input`, `Output` and cloneable `Control`. Input and output can progress independently.
- `Input::send` transfers an owned chunk into the bounded input queue. Successful completion means local acceptance. `try_send` is nonblocking and returns the original chunk on rejection.
- `Input::finish` consumes the input handle and places an ordered end-of-input barrier after accepted chunks. Await the returned Future. Dropping it before polling cancels the request.
- `Output::recv` delivers queued events, exactly one `Completed` after successful backend cleanup and output drainage, then `None`. `ProcessingFinished` is not the successful response terminal.
- Failures discard undelivered data and bypass full queues. `recv` returns the terminal error once, then `None`. Errors include request identity and partial delivery counters; retained terminal error text is capped at 4096 UTF-8 bytes.
- `Control::cancel` is immediate and idempotent. `closed` waits for backend cleanup, independently of output consumption. Its cached result does not change if a deadline or cancellation later discards buffered output.
- Dropping unfinished input or unread output cancels the request. Dropping a control clone does not. Dropping the last Client starts shutdown.
- `shutdown` immediately rejects new requests and cancels active requests, even if its Future is never polled. Awaiting it waits for request and adapter cleanup. Polling a common Future never joins a worker thread.

## Runtime and ownership

Common control uses standard-library synchronization, `Future` and `Waker`, without Tokio, generated protocol types, native inference libraries or async traits. One deadline thread per Client advances timeouts even when application Futures are not being polled. Server initialization selects the Tokio runtime required by its transport. Direct uses one standard control thread per Client to drive bounded session Futures, with the shared inference layer owning native workers. Waiting requests do not create threads.

PCM and curve vectors move through the queues without copying their elements. Shared curve layouts retain their `Arc` identity. Dropping a pending send removes its unaccepted chunk; accepted chunks remain ordered. Sending and receiving should be driven concurrently to allow bounded queues to make progress.

## Limits

`Limits` controls active local request count, per-request settings, queue item counts, queue storage, individual chunks/events and total retained buffer storage. A queue at its item or byte limit applies backpressure. The global storage budget is a hard admission bound: insufficient budget returns `LimitExceeded` instead of retaining an unbounded pending buffer. `try_send` reports `QueueFull` without consuming the rejected chunk when its queue is full.

Each input/output endpoint can stage one pending operation outside its queue; these buffers, request settings and backend input handoffs remain charged to the global budget. `Vec` and `String` spare capacity is counted. Shared layouts are conservatively charged per event, and map nodes use a conservative per-entry allowance. These are retained-storage accounting units, not an exact allocator or process-memory ceiling; control structures, allocator overhead and buffers already returned to the application are excluded.

A request slot remains occupied until backend cleanup and either output consumption through `Completed` or cancellation/failure. This bounds successful but unread responses as well as requests awaiting execution. Execution-engine scheduling remains the responsibility of the shared inference layer.

## Validation

Permanent tests are Rust unit/integration tests. A private backend and standard-Waker executor cover bounded queues, ownership, dropped Futures and handles, cancellation races, deadlines, worker panic, cleanup ordering and shutdown. Run `cargo test -p audio2face3d-client --features direct,server`. Adapter tests additionally cover actual TCP disconnection, trailers, missing terminal messages, explicit/stopped Tokio runtimes, FIFO execution capacity and mode comparison. Native tests are explicitly ignored by default and require model/SDK configuration.

## Adapter behavior

DirectConfig selects the engine/model, optional emotion model, device, execution capacity and queue policy. All Client clones share FIFO admission. The same queue algorithm handles one or multiple execution slots. A permit is retained until native cleanup completes. Loaded runtime pooling across utterances is not implemented. `DirectConfig::default()` deliberately uses the diagnostic mock; select Regression as above for native inference.

Server success requires a response header, a final SUCCESS status and clean gRPC termination. Intermediate SUCCESS and ProcessingFinished do not end the response. Disconnected utterances are never automatically retried; later requests may reconnect. `closed` acknowledges local RPC cleanup, not remote GPU completion.

The server adapter also bounds encoded/decoded gRPC message size through `max_message_bytes`. A decoded packet, protocol conversion storage, transport buffers and native model memory are additional to the common queue accounting. Backpressure stops response reads; cancellation and deadlines remain independent of application polling. A remote error cannot be observed until its response bytes/trailers can be read.
