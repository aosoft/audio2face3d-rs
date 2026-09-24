# Audio2Face-3D GUI

The Windows desktop application inspects local or remote inference with a
procedural 52-channel head, raw/interpolated values, bars, audio playback,
channel timelines and bounded logs. The head is diagnostic geometry, not a
production character or a reconstruction of the training head.

## Run a packaged application

Keep `audio2face3d-gui.exe` and `assets/default-head.glb` together. Double-click
the executable from any working directory. It resolves the standard head next
to the executable; it never searches a developer's checkout. An optional first
argument overrides the head path. `Open head` also loads a compatible GLB.
Missing/invalid assets produce an error with the attempted path; the window
remains usable to select another file. Installed Meiryo/Yu Gothic fonts are used
as a fallback for Japanese filenames/errors on Windows when available.

The gRPC package needs a graphics/audio device but no CUDA/TensorRT installation.
The local+gRPC package additionally needs the separately obtained model JSON,
cached TensorRT engine, NVIDIA driver, CUDA and TensorRT runtime. Configure
`platform.toml` using the startup options below. The window's `CUDA root override`
and `TensorRT root override` fields can replace the configured SDK group.
No models or NVIDIA SDK binaries are redistributed with this application.
Windows x64 is the tested desktop target; other desktop platforms are unvalidated.

1. Select a WAV with `Browse WAV`.
2. Choose `Grpc` and set the endpoint (default `http://127.0.0.1:52000`) and API
   key if required, or `Local` and select the model JSON and GPU index.
3. Click `Infer WAV`. By default, wait for `Completed` then press `Play`.
   `Cancel` stops the current request; the next request is enabled after cleanup.
4. Enable `Play while inferring` before starting to play as results arrive.
5. Select channels in the right panel to show their graphs. Drag the timeline
   cursor to seek within received data; use zoom/scroll controls and loop/pause.
   Bars clip to the display range; numeric/raw values preserve received data.
6. Drag the head to rotate, use the wheel to zoom, or choose a camera preset.
   `Manual` stops audio and exposes individual sliders to inspect morphs.
   `Sync demo` supplies a short synthetic clip without loading an inference model.
7. Expand `Logs` to filter/copy records and toggle following new messages.

WAV accepts mono/stereo PCM16/24/32 and float32, 8..192 kHz, at most 600 seconds.
Stereo is averaged and a 128-tap windowed-sinc low-pass resampler converts to
mono PCM16 at 16 kHz. Nonfinite samples and malformed input are rejected.
Playback follows the **returned** audio and timed curves. The device's scheduled
audio timestamp drives the head, values and cursor, including queued audio latency.

Streaming sends 100 ms chunks, initially up to 500 ms ahead, then at real-time
pace. It starts/restarts after 100 ms of contiguous audio and curves are ready.
If inference falls behind, audio outputs silence and media time freezes until
ready. This is observable as buffering and an underrun count. The separate `audio busy` count records callbacks that emitted
silence because the shared player was locked. Normal short clips
flush after completion; failed/cancelled partial results can be inspected by
seeking but do not play automatically. Streaming is not a real-time guarantee.
Input sent, result completion, resource release and playback end are distinct.
Pause/seek do not cancel inference; future, unreceived time is not seekable.
Results are limited to 256 MiB and 10 minutes; exceeding a limit marks the result
failed. Logs retain 10,000 records with a 2,048-record nonblocking queue and a
visible dropped count. Closing waits for cancellation/shutdown; native model
initialization already in progress may take time before it can finish cleanup.

## Startup options and platform.toml

The standard executable uses the same `PlatformArgs` resolver as the inference
and server CLIs. File selection is: `--platform-config FILE`, then
`AUDIO2FACE3D_PLATFORM_CONFIG`, then `platform.toml` in the working directory,
then the user configuration directory, then default runtime discovery.
Only one file is selected; missing explicit files or invalid files are errors.
Paths inside TOML are relative to that file. Command-line paths are relative to
the working directory. The executable does not guess the source checkout from
its installation directory. The head asset remains relative to the executable.

```powershell
./audio2face3d-gui.exe --platform-config C:/config/platform.toml --mode local --model C:/models/mark/model.json --wav C:/audio/voice.wav
./audio2face3d-gui.exe --help
```

`--head FILE` (or the legacy positional GLB), `--mode local|grpc|mock`,
`--model JSON`, `--wav WAV`, `--endpoint URL`, and `--device INDEX` populate
controls. `mock` requires the development feature. `--infer` also starts inference;
without it no inference begins automatically. `--play-while-inferring` enables
paced streaming playback. Existing shared flags `--cuda-root`, `--tensorrt-root`,
`--cuda-library-dir`, `--tensorrt-library-dir` and `--runtime-search` override the
selected file with the same rules as the server CLI.

Runtime roots, directory lists and search policy are retained in the request and
passed into the inference context. The Local panel displays the startup runtime
settings. Blank GUI override fields preserve them; a nonempty root replaces that
SDK's configured root/directory list while retaining the other SDK and policy.
Embedded hosts provide `startup::Options` or `Request::runtime` explicitly;
constructing a library request never reads process arguments or config files.

## Build and package from source

Rust 2024 / Rust 1.91+, Windows x64 MSVC. From the checkout:

```powershell
cargo run -p audio2face3d-gui --no-default-features --features desktop,grpc -- crates/audio2face3d-gui/assets/default-head.glb
./ci/package-gui.ps1 -Mode grpc
./ci/package-gui.ps1 -Mode local-grpc -PlatformConfig platform.toml
```

The packaging script builds release mode with the lockfile and creates a **new**
`temp/gui-dist-grpc` or `temp/gui-dist-local-grpc` directory. Override
`-OutputDirectory` for another location. It refuses an existing destination.
It copies the executable, standard GLB, usage notes, license texts, dependency
inventory/notices (including fonts), and the corresponding MPL-covered SVD source.
Keep the whole directory when redistributing. No asset generation runs at build
or application startup. A graphics driver and system audio output are required.

For local builds, use the runtime/build setup documented in the repository's
`docs/platform.md`. CUDA 12.9 was validated with the MSVC **14.42** developer
environment; selecting that compiler while retaining newer MSVC include paths
is insufficient. Run the corresponding `vcvars64.bat -vcvars_ver=14.42` first.
No developer SDK path is compiled into the application configuration.

## Regenerate or replace the head

```powershell
cargo run -p audio2face3d-headgen --features cli -- --config crates/audio2face3d-headgen/presets/default-head.json --output crates/audio2face3d-gui/assets/default-head.glb
```

The versioned preset/tool produces deterministic bytes: 2,134 vertices in 18
parts, 4,016 triangles, 52 active channels, 860,580 bytes. The head surface alone
has 610 vertices. The generated mesh and deltas are original MIT-licensed work.
External heads must follow `docs/gui-contract.md`: embedded GLB with indexed
triangles, position/normal morph deltas, target names, metadata and plain materials.
Arbitrary production glTF files with skins, transforms or textures are unsupported.
The NVIDIA-style `MouthClose` pose independently lowers the chin with closed lips;
it is not the inverse of `JawOpen`. Combined weights are additive without hidden
clamping or corrective mixing. Inspect extreme combinations as diagnostics.

## Library boundaries and engine embedding

- `audio2face3d-gui-core`: CPU model types, validation and optional GLB read/write.
- `audio2face3d-headgen`: CPU generator library with optional `cli`.
- `audio2face3d-gui`: playback/session/logging library; `render-wgpu` and `ui-egui`
  are optional. `desktop` adds the standard CPAL/eframe host. `local` and `grpc`
  independently enable inference. `mock` and `capture` are development features.

All default feature sets are empty. A host can feed `Clip`, call `Player` with
its audio callback and scheduled audible timestamp, and obtain one `Snapshot`
for the head/values/timeline. `HeadRenderer` accepts a host-owned wgpu device,
queue and render target; the host submits commands and owns texture lifetime.
No global tracing subscriber is installed. Pass the existing `Logger` interface
or `FanoutLogger` to inference. GUI log callbacks enqueue; engine loggers needing
thread affinity must enqueue on the receiving side too.

Unreal editor panels, C ABI, native texture sharing and RHI synchronization are
future E1/E2 work. A wgpu texture is not assumed to be directly shareable with an
engine device. ACE wire-compatible gRPC tests are separate from Unreal plugin
playback verification.

## Validation

`ci/run-tier.ps1 portable` includes CPU model/GLB/generator/playback/log/WAV/Mock
tests and desktop gRPC compilation. Hardware tests are intentionally separate:

```powershell
cargo test -p audio2face3d-gui --features render-wgpu --test gpu -- --ignored --nocapture
cargo run -p audio2face3d-gui --features desktop --example audio_probe
cargo run --release -p audio2face3d-gui --features desktop,local,grpc --example stream_probe -- local input.wav models/mark/model.json
```

The stream probe requires a long enough input to start audible playback before
input completion. `infer_probe` compares ordinary results and accepts `local`,
`grpc` or explicit development `mock`. Capture helpers live under `examples`.
Recorded hardware results and the distinction between fresh context, cached
engine and untested prewarmed clients are in `docs/gui-progress.md`.
