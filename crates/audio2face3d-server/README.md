# Audio2Face-3D gRPC server

Generate facial animation from audio through the ACE
`A2FControllerService/ProcessAudioStream` bidirectional RPC. The server uses
this workspace's Rust Audio2Face-3D runtime for Regression inference and the
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
`bin` directories to `PATH`. Build with the `runtime` feature and explicitly
select the `regression` backend:

```powershell
$env:PATH = "$env:CUDA_PATH\bin;$env:TENSORRT_ROOT_DIR\bin;$env:PATH"
cargo build --release --locked -p audio2face3d-server --features runtime
.\target\release\audio2face3d-server.exe --backend regression --model models/mark/model.json
```

The model path above assumes Mark was prepared under `models/mark`.
The default endpoint is `127.0.0.1:52000`. Transport is plaintext HTTP/2, without
server authentication or TLS. The default bind address is loopback.

`--device` selects the CUDA device (default 0). `--emotion-model` optionally
adds an Audio2Emotion classifier descriptor. Without it, the emotion
post-processing path uses configured/preferred emotions without classifier
inference. Each concurrent RPC owns its runtime; model loading and GPU
memory requirements must be included in the deployment budget. Start with
`--max-streams 1`, especially with Audio2Emotion.

Use `RUST_LOG=debug` for detailed logs. Ctrl+C changes health to NOT_SERVING,
cancels active work and waits for bounded shutdown. Startup failures and
shutdown timeouts result in a process error.

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
- Invalid input returns INVALID_ARGUMENT; concurrency/duration limits return
  RESOURCE_EXHAUSTED; idle/output-queue timeout returns DEADLINE_EXCEEDED.
  Failed streams do not emit SUCCESS.

Use `--help` for all options. Defaults are one active stream, a 1 MiB message
limit, 600 seconds of audio, 16 queued output messages, 30 seconds of input
idle time, 10 seconds of output queue wait, and 5 seconds of shutdown wait.
Output queue timeout is not a deadline for actual client playback.

## Development: mock backend and curve diagnostics

The mock backend is for protocol and face-mapping diagnostics without model
inference. It requires neither CUDA nor TensorRT:

```powershell
cargo run --locked -p audio2face3d-server -- --backend mock
```

For compatibility, the CLI currently selects mock when `--backend` is
omitted, and the default Cargo features do not include the inference runtime.
Use `--features runtime` and `--backend regression` as shown above for real
inference.

The default mock pattern emits a one-second triangular `JawOpen` pulse and returns
the supplied PCM. Existing presets remain available through `--mock-pattern`:
`jaw-open-pulse`, `eye-blink-left`, `eye-blink-right`, `mouth-smile-left`,
`mouth-smile-right`.

Select any of the [52 case-sensitive ACE names](src/animation.rs) with
`--mock-curve`. It overrides the preset's curve selection. Add `--mock-value`
to hold that curve at a finite weight in [0, 1] while audio is streamed;
without it, the selected curve pulses. Other weights are zero unless a jaw baseline is specified.

```powershell
cargo run --locked -p audio2face3d-server -- --backend mock --mock-curve EyeBlinkLeft --mock-value 1
# Neutral reference, with all 52 weights zero:
cargo run --locked -p audio2face3d-server -- --backend mock --mock-curve JawOpen --mock-value 0
```

For combined diagnostics, `--mock-jaw-open 0.5` holds JawOpen at 0.5 while
the selected curve varies or holds its own value. This helps inspect
MouthClose and TongueOut against an open-jaw reference. Set `--mock-value 0`
for the reference, then 0.5 or 1 for comparison. The baseline accepts finite
values in [0, 1] and cannot be combined with selecting JawOpen itself.

```powershell
cargo run --locked -p audio2face3d-server -- --backend mock --mock-curve TongueOut --mock-value 1 --mock-jaw-open 0.5
```

These options are for mock only. Mock weights do not depend on the supplied
audio or emotion and do not test lip synchronization or model quality.

## Verification and scope

```powershell
cargo fmt --all --check
cargo check --locked -p audio2face3d-server
cargo test --locked -p audio2face3d-server
cargo clippy --locked -p audio2face3d-server --all-targets -- -D warnings
cargo test --release --locked -p audio2face3d-server --features runtime
```

The protocol tests include all 52 isolated curves, PCM/time preservation,
invalid input, cancellation, limits and shutdown. Enabling `runtime` in
cargo test checks native compilation and available tests; it does not by
itself prove real-model inference. Test inference with a prepared model and
an actual client.

GetConfigs, TLS/mTLS, Diffusion serving, the GPU BlendShape solver, extended
tongue output and model pooling are not implemented by this server. It emits
52 face curves, not the additional tongue curves or head rotation channels.
Long-running production stability and perceptual lip-sync quality are not
established by the short integration tests.

The bundled NVIDIA protocol definitions retain their upstream notices and
are covered by [LICENSE-APACHE](LICENSE-APACHE). See the workspace license
for the Rust implementation.