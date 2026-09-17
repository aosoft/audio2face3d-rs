# Audio2Face-3D gRPC server

Generate facial animation from audio through the ACE
`A2FControllerService/ProcessAudioStream` bidirectional RPC. The server uses
the workspace's [shared inference layer](../../README.md) for Regression inference and the
host BlendShape solver, returning 52 face curves together with audio.
Audio2Emotion classifier inference is optional.

This crate is built from the workspace and is not published to crates.io.
A separate mock backend is available for development diagnostics.

## Build and run Regression inference

Run commands from the workspace root. Requirements:

- Rust 1.91 or newer (Edition 2024); protoc is bundled by the build dependency.
- A CUDA-capable GPU, CUDA and TensorRT. The tested native configuration is
  Windows x64/MSVC, CUDA 12.9 and TensorRT 10.16.1.
- A 16 kHz Regression model and its TensorRT engines. Follow the workspace
  instructions for [model acquisition](../../README.md#explicit-model-acquisition)
  and [engine generation](../../README.md#tensorrt-engine-generation).
  Model files and native SDKs are separate downloads.

Set `CUDA_PATH` and `TENSORRT_ROOT_DIR` to your installations, then add their
`bin` directories to `PATH`. The default `native` feature provides Regression inference. Enable `cli` to build the executable:

```powershell
$env:PATH = "$env:CUDA_PATH\bin;$env:TENSORRT_ROOT_DIR\bin;$env:PATH"
cargo build --release --locked -p audio2face3d-server --features cli
.\target\release\audio2face3d-server.exe --backend regression --model models/mark/model.json
```

The model path above assumes Mark was prepared under `models/mark`.
The default endpoint is `127.0.0.1:52000`. Transport is plaintext HTTP/2, without
TLS. API-key authentication is opt-in. The default bind address is loopback.

`--device` selects the CUDA device (default 0). `--emotion-model` optionally
adds an Audio2Emotion classifier descriptor. Without it, the emotion
post-processing path uses configured/preferred emotions without classifier
inference. Each concurrent RPC owns its runtime; model loading and GPU
memory requirements must be included in the deployment budget. Start with
`--max-streams 1`, especially with Audio2Emotion.

Use `RUST_LOG=debug` for detailed logs. Ctrl+C changes health to NOT_SERVING,
cancels active work and starts cleanup. Each shutdown stage has its own timeout.
After a timeout the executable keeps the runtime alive until cleanup completes,
then returns a process error; the timeout does not force-release native resources.

## Library, logging and authentication

The library never creates a runtime, reads environment variables, installs a
global logger or handles process signals. The embedding application binds a
Tokio listener, builds a `Server`, and supplies its stop future:

```rust,no_run
use std::{future::Future, sync::Arc};
use audio2face3d::{Audio2Face3DContext, logging::Logger};
use audio2face3d_server::{
    auth::Authenticator, Server, ServerConfig, ServerError, ShutdownReport,
};

async fn run<A: Authenticator>(
    config: ServerConfig,
    logger: Arc<dyn Logger>,
    verifier: Option<A>,
    stop: impl Future<Output = ()> + Send + 'static,
) -> Result<ShutdownReport, Box<dyn std::error::Error + Send + Sync>> {
    let context = Audio2Face3DContext::builder().logger(logger).build();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:52000").await?;
    let server = Server::builder(config)
        .context(context)
        .authentication(verifier)
        .build()?;
    match server.serve(listener, stop).await {
        Ok(report) => Ok(report),
        Err(ServerError::ShutdownTimeout { completion, .. }) => {
            completion.await?;
            Err("shutdown exceeded its deadline; cleanup completed".into())
        }
        Err(error) => Err(error.into()),
    }
}
```

Use `serve(listener, async { /* wait for application stop */ })`, where the
future returns `()`. The returned `ShutdownReport` records server observations,
not client receipt or playback. A `ShutdownTimeout` error owns a
`CleanupCompletion`: await it before flushing your writer or dropping the
runtime. Dropping the handle does not stop the supervisor. Dropping the serve
future requests shutdown but does not synchronously join workers.

Configure a shared `Audio2Face3DContext` with `Arc<dyn Logger>`; the default is
`NoopLogger`. The standard-library-only `Logger` trait exposes `log_level`,
`write_log`, and lazy `log(level, closure)`. Values below the threshold are not
formatted. Runtime tasks, inference workers and cleanup retain their context.
The executable writes synchronously to stderr; `RUST_LOG` supports levels and
target directives, not span/field expressions. Target-specific rejection occurs
in the writer after the global minimum level check.

The library supplies no credential database or single-key comparison policy.
Inject a synchronous `Fn(AuthRequest) -> AuthResult` with
`builder.authentication(Some(verifier))`. Its concrete type is preserved in
`Server<A>`; no boxed authenticator is required. Runtime `Option<A>` is supported.
`without_authentication()` returns to the default marker type.
`async_authenticator` accepts closures returning an owned Send future. A custom
`Authenticator` can use its associated borrowed future to avoid copying keys.
Credentials are available only through explicit `SecretApiKey::expose()`.
Principals must be safe identifiers, never keys or key prefixes. Custom
verifiers must not log credentials or include them in panic payloads.

For the executable, `--api-key KEY` overrides `AUDIO2FACE3D_API_KEY`. If neither
is set, authentication is disabled. Empty or malformed selected values fail
startup. The executable's private verifier compares one key using `subtle`;
the library only validates and dispatches Bearer credentials. Help, version and
descriptor export do not read the authentication environment. Argument errors
are deliberately generic to avoid echoing secrets.

Health is public by default. `--health-auth same-as-inference`, or the
`HealthAuth::SameAsInference` builder setting, applies the same verifier to
Check and Watch and requires authentication to be configured. Each Watch
authenticates once and releases its auth slot. Health never uses an inference
slot. Unknown RPC methods remain UNIMPLEMENTED.

Authentication permits at most 64 simultaneous verifications, each with a
5-second timeout, before entering the inference FIFO. Rejections do not read
audio or start inference. Invalid credentials return UNAUTHENTICATED; denied
access returns PERMISSION_DENIED; verifier outages/timeouts return UNAVAILABLE;
full auth capacity returns RESOURCE_EXHAUSTED. HTTP/2 header lists are bounded
to 16 KiB, credentials to 4 KiB. RPC deadlines continue through queued work and
response streaming; cancellation retains native ownership until cleanup.

## Streaming contract

- Send a controller header, PCM chunks, then `EndOfAudio`. The server does
  not wait for the client to half-close after that marker.
- Input is PCM16 little-endian, mono, at 16,000, 44,100 or 48,000 Hz. Chunks
  must end on sample boundaries; an utterance needs at least one sample.
- Output contains a header with 52 names, 30 Hz animation frames with
  corresponding 16 kHz PCM, an end event and final SUCCESS status, then EOF.
  Audio and curve timestamps refer to the same sample position. The final
  partial frame is preserved. 16 kHz input PCM is returned unchanged;
  44.1/48 kHz input is resampled.
- Regression supports the implemented face float parameters, BlendShape
  multipliers/offsets/clamping, and emotion settings/keyframes. Unknown
  parameters and unsupported forms are rejected. Mock ignores these settings.
- Standard gRPC health Check/Watch supports the empty service name and
  `nvidia_ace.services.a2f_controller.v1.A2FControllerService`.
- Invalid input returns INVALID_ARGUMENT; a full request queue or duration
  limit returns RESOURCE_EXHAUSTED; request-wait/idle/output-queue timeout
  returns DEADLINE_EXCEEDED.
  Failed streams do not emit SUCCESS.

Use `--help` for all options. Defaults are one active stream, 64 waiting
requests, a 1 MiB message limit, 600 seconds of audio, 16 queued output messages, 30 seconds of input
idle time, 10 seconds of output queue wait, and 5 seconds of shutdown wait.
Output queue timeout is not a deadline for actual client playback.

Requests exceeding `--max-streams` wait FIFO for an execution slot. With
`--max-streams 1`, requests execute sequentially. Ordering is the order in
which the server registers waiters, not client wall-clock send timestamps.
`--request-queue-capacity` bounds waiting requests separately from active
streams. `--request-queue-timeout-ms` defaults to 0 (no server-imposed wait
limit); client deadlines still apply. Cancellation removes a waiting request,
and shutdown wakes waiters with UNAVAILABLE.

Waiting does not start inference or read the audio stream into an application
buffer; HTTP/2 flow control can block uploads until execution starts. Input
idle timeout starts after admission. An execution slot is retained until the
worker has cleaned up and its response stream has drained or been cancelled.
This queues RPC processing, not playback on a remote device. The request queue
is in memory and is not preserved across server restarts.

## Development: mock backend and curve diagnostics

The mock backend is for protocol and face-mapping diagnostics without model
inference. It requires neither CUDA nor TensorRT:

```powershell
cargo run --locked -p audio2face3d-server --no-default-features --features cli,mock -- --backend mock
```

Mock is the default backend only for a mock-only build. The default Cargo
feature is native, and native builds default to Regression even when mock is
also enabled. Regression requires an explicit model; failures never fall back to mock.

The default mock pattern emits a one-second triangular `JawOpen` pulse and returns
the supplied PCM. Existing presets remain available through `--mock-pattern`:
`jaw-open-pulse`, `eye-blink-left`, `eye-blink-right`, `mouth-smile-left`,
`mouth-smile-right`.

Select any of the [52 case-sensitive ACE names](../audio2face3d/src/inference/animation.rs) with
`--mock-curve`. It overrides the preset's curve selection. Add `--mock-value`
to hold that curve at a finite weight in [0, 1] while audio is streamed;
without it, the selected curve pulses. Other weights are zero unless a jaw baseline is specified.

```powershell
cargo run --locked -p audio2face3d-server --no-default-features --features cli,mock -- --backend mock --mock-curve EyeBlinkLeft --mock-value 1
# Neutral reference, with all 52 weights zero:
cargo run --locked -p audio2face3d-server --no-default-features --features cli,mock -- --backend mock --mock-curve JawOpen --mock-value 0
```

For combined diagnostics, `--mock-jaw-open 0.5` holds JawOpen at 0.5 while
the selected curve varies or holds its own value. This helps inspect
MouthClose and TongueOut against an open-jaw reference. Set `--mock-value 0`
for the reference, then 0.5 or 1 for comparison. The baseline accepts finite
values in [0, 1] and cannot be combined with selecting JawOpen itself.

```powershell
cargo run --locked -p audio2face3d-server --no-default-features --features cli,mock -- --backend mock --mock-curve TongueOut --mock-value 1 --mock-jaw-open 0.5
```

These options are for mock only. Mock weights do not depend on the supplied
audio or emotion and do not test lip synchronization or model quality.

## Verification and scope

```powershell
cargo fmt --all --check
cargo check --locked -p audio2face3d-server
cargo test --locked -p audio2face3d-server --no-default-features --features mock
cargo clippy --locked -p audio2face3d-server --no-default-features --features cli,mock --all-targets -- -D warnings
cargo test --release --locked -p audio2face3d-server --features cli
```

The protocol tests include all 52 isolated curves, PCM/time preservation,
invalid input, cancellation, limits and shutdown. Enabling `native` in
cargo test checks native compilation and available tests; it does not by
itself prove real-model inference. Test inference with a prepared model and
an actual client.

GetConfigs, TLS/mTLS, Diffusion serving, the GPU BlendShape solver, extended
tongue output and model pooling are not implemented by this server. It emits
52 face curves, not the additional tongue curves or head rotation channels.
Long-running production stability and perceptual lip-sync quality are not
established by the short integration tests.

Inference, resampling and FIFO admission use the shared inference module. Native
work runs on standard control workers; gRPC transport remains on Tokio.

The NVIDIA protocol definitions are maintained in
[protocol module](../audio2face3d/proto). They retain their
upstream notices and are covered by [LICENSE-APACHE](LICENSE-APACHE).
See the workspace license for the Rust implementation.
