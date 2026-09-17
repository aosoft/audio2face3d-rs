# Audio2Face-3D inference

Shared Regression inference, optional Audio2Emotion processing, PCM resampling,
and execution admission for the workspace. Inputs, outputs, and errors use
`audio2face3d-types`; this crate does not expose protocol messages.

`runtime` enables the native model engine (CUDA/TensorRT), not an async runtime.
Default features are empty and support the diagnostic mock without native SDKs.
Neither configuration depends on Tokio, tonic, prost, or async-trait. Rust Edition
2024 and Rust 1.91 or newer are required.

## Engine operations

`Factory::prepare(Config)` validates settings and, for Regression, prepares one
native engine. `Factory::start(RequestOptions)` creates an utterance backend.
The unused prepared engine is consumed once when the request has no custom
parameters. Each subsequent utterance loads its own engine; engines are not
pooled or reset for reuse. `release_prepared()` awaits destruction of any unused
prepared engine without affecting active utterances.

`Backend` uses ordinary trait methods returning `EngineFuture`:

- `push(InputChunk)` accepts owned PCM and sparse emotion keyframes.
- `next_frame(&Cancellation)` returns an `OutputBatch` or `None` when currently
  available output is exhausted. Before finish, `None` means more input is needed.
- `finish()` closes input and flushes its resampler. Continue draining output.
- `close()` drains and destroys native resources; await it after success or error.

Operations on an utterance are sequential. Dropping an operation Future does not
cancel native work already submitted: request cancellation, then await `close()`
before releasing its admission permit. Dropping the backend also schedules native
cleanup on its owner worker, but does not itself wait for that cleanup.

Input is PCM16 mono at 16/44.1/48 kHz. Output is 16 kHz PCM with 30 Hz frames and
52 named curves. Native weights are copied from borrowed callbacks into owned
buffers. The existing resampling, quantization, reordering and clamping algorithms
are preserved; buffer-copy optimization is separate work.

`RequestOptions::timeout` is for the owning session to enforce, not a timer started
by this engine. This crate is an engine layer; the split input/output Client/Session
API and transport-backed inference are separate layers.

## Execution admission and runtime ownership

`Admission::new(active, queued, timeout)` maintains FIFO waiters and uses any free
slot for all positive active capacities. `queued` limits only waiting requests;
a zero timeout means no queue deadline. `acquire()` registers immediately and
returns a standard Future. A full queue yields `QueueFull`.

Cancellation, acquire-Future drop, queue expiry and `Admission::close()` remove
waiting requests. Cancellation and expiry also progress while callers stop polling.
Dropping a reserved but unconsumed acquire returns its slot. Delivered RAII permits
remain held until their owners release them, including after queue shutdown.
Keep the permit until native cleanup and output consumption/discard both finish.

There is one optional deadline thread per admission queue, never one per waiting
request. Regression has one standard control worker per initialized engine. That
worker owns model construction, execution waits and destruction. Native JobRunner
threads remain separate, so waiting for their work cannot consume their execution
capacity. Future polling observes completion using standard Mutex/Waker primitives;
it does not enter a Tokio runtime or join a native thread.

The low-level workspace crate may use Tokio in its own dev-tests. This crate has
no such dependency, including in its tests.

## Verification

```powershell
cargo test --locked -p audio2face3d-inference
cargo clippy --locked -p audio2face3d-inference --all-targets -- -D warnings
```

Native tests additionally require the SDK environment and prepared model engines:

```powershell
$env:PATH="$env:CUDA_PATH/bin;$env:TENSORRT_ROOT_DIR/bin;$env:PATH"
$env:A2F_MODEL=Resolve-Path models/mark/model.json
$env:A2E_MODEL=Resolve-Path models/emotion/model.json
cargo test --locked -p audio2face3d-inference --features runtime --lib
cargo test --locked -p audio2face3d-inference --features runtime --lib -- --ignored --nocapture
```

These native tests use standard Future waiting throughout, with no async runtime.
They cover model lifecycle, real inference with capacities 1/2, cancellation and
recovery, and typed Audio2Emotion metadata. Capacity 4 is exercised in queue/mock
tests. Short tests do not establish sustained production throughput or quality.
