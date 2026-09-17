# audio2face3d-client

Runtime-independent session control for Audio2Face inference, using the Rust-owned types in `audio2face3d-types`.

This crate currently implements the common Client/Session layer. The `direct` and `server` features reserve the adapter boundaries; their constructors and inference/transport implementations are not yet available. They are planned for stages 06 and 05 respectively. The private test backend is not a public inference mode.

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

Common control uses standard-library synchronization, `Future` and `Waker`, without Tokio, generated protocol types, native inference libraries or async traits. One deadline thread per Client advances timeouts even when application Futures are not being polled. Future server initialization will select the Tokio runtime required by its transport; direct initialization will preserve runtime independence.

PCM and curve vectors move through the queues without copying their elements. Shared curve layouts retain their `Arc` identity. Dropping a pending send removes its unaccepted chunk; accepted chunks remain ordered. Sending and receiving should be driven concurrently to allow bounded queues to make progress.

## Limits

`Limits` controls active local request count, per-request settings, queue item counts, queue storage, individual chunks/events and total retained buffer storage. A queue at its item or byte limit applies backpressure. The global storage budget is a hard admission bound: insufficient budget returns `LimitExceeded` instead of retaining an unbounded pending buffer. `try_send` reports `QueueFull` without consuming the rejected chunk when its queue is full.

Each input/output endpoint can stage one pending operation outside its queue; these buffers, request settings and backend input handoffs remain charged to the global budget. `Vec` and `String` spare capacity is counted. Shared layouts are conservatively charged per event, and map nodes use a conservative per-entry allowance. These are retained-storage accounting units, not an exact allocator or process-memory ceiling; control structures, allocator overhead and buffers already returned to the application are excluded.

A request slot remains occupied until backend cleanup and either output consumption through `Completed` or cancellation/failure. This bounds successful but unread responses as well as requests awaiting execution. Execution-engine scheduling remains the responsibility of the shared inference layer.

## Validation

Permanent tests are Rust unit/integration tests. A private backend and standard-Waker executor cover bounded queues, ownership, dropped Futures and handles, cancellation races, deadlines, worker panic, cleanup ordering and shutdown. Run `cargo test -p audio2face3d-client`; both feature flags currently exercise the same common layer. Adapter behavior will receive separate tests when implemented.
