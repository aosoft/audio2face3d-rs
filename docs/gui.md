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
compatible TensorRT engine, NVIDIA driver, CUDA Runtime with cuBLAS/cuBLASLt/cuRAND,
and TensorRT runtime libraries, including their dependencies. `cudart` alone is
insufficient. Running the prebuilt application requires neither `nvcc`, SDK headers,
nor Visual Studio/Build Tools; the compiled CUDA PTX is embedded in the binary.
Ordinary OS/runtime prerequisites still apply. A full CUDA Toolkit installation
is not required for normal inference.

Configure `platform.toml` using the startup options below. A deployment file may
contain only `[runtime]` with `cuda-library-dirs` and `tensorrt-library-dirs`;
`[build-cuda]` is unnecessary. Relative directories are resolved against that TOML.
Runtime paths are resolved at startup and cannot be edited in the Inference panel.
See [Deploying a prebuilt application](platform.md#deploying-a-prebuilt-application)
for the dependency table and configuration example. When moving to another GPU or
TensorRT version, check engine compatibility; regenerating an engine requires
`trtexec`, separately from running inference. Launch the executable directly on the
destination machine: `cargo run` is a build-and-run command, not a deployment command.
No models or NVIDIA SDK binaries are redistributed with this application.
Windows x64 is the tested desktop target. CI checks macOS gRPC-only compilation; macOS GUI/audio behavior still requires device testing.

1. Select a WAV with `Browse` and choose `Play while inferring` as needed.
2. Choose `gRPC` and set the endpoint (default `http://127.0.0.1:52000`) and API
   key in its group, or `Local` and select the model JSON. GPU selection uses
   the startup `--device` option.
3. Press `Initialize & Start` above the seek slider. With `Play while inferring` off,
   inference completes into memory and playback starts automatically. `Pause`
   pauses playback; `Play` resumes without running inference again. Editing any
   inference setting while paused invalidates the result and restores
   `Initialize & Start`. While inference prepares the results, the button displays
   `Abort Initializing`; clicking it cancels the operation.
4. With `Play while inferring` on, every `Initialize & Start` starts a fresh
   inference session from the beginning. Before playback begins, the button shows
   `Abort Initializing`. Once playback starts it becomes `Stop`, which cancels
   inference and stops audio. Seeking and looping are disabled in this mode.
   The timeline follows playback as results arrive. Inference settings are editable
   only while the button shows `Initialize & Start` or `Play`; cancellation must
   finish releasing resources before settings can be edited again.
5. Select channels in `Channels` to show their graphs. Drag the timeline
   cursor to seek within completed offline results; use zoom/scroll controls and loop/pause.
   Pressing the time ruler or a track seeks immediately on mouse-down; dragging
   continues seeking. The ruler shows seconds with zoom-dependent major/minor
   ticks, and one yellow playhead spans the ruler and all visible channel rows.
   The top seek bar represents the entire clip. The bar directly above the ruler
   represents the visible time range: its thumb width is proportional to the
   visible fraction, and dragging it pans without seeking. During playback or a
   seek, the viewport stays still until the playhead leaves it, then recenters
   on the playhead (clamped at the clip endpoints). Paused manual panning is retained.
   Bars clip to the display range; numeric/raw values preserve received data.
   Channels fill multiple columns according to the available width, with one
   compact row per channel. Hover over a numeric value to see its raw value.
   Name fields fit the longest channel name rather than stretching with the
   window. Numeric values and bars align within each column; gutters and vertical
   separators distinguish neighboring columns.
   In Channels only, names and numeric values turn dark gray when the current
   value is exactly zero, including in Manual mode. Timeline colors stay unchanged.
   The head occupies a narrow resizable panel on the left; the remaining width
   is reserved for channels. Manual controls use the same column layout.
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
silence because the shared player was locked. Channel/timeline drawing runs
outside the player lock; only snapshots and graph-data preparation hold it.
`audio_gui_probe` exercises real audio under channel/timeline drawing load and
reports source underruns separately from audio lock contention. Normal short clips
flush after completion; failed/cancelled partial results can be inspected by
seeking but do not play automatically. Streaming is not a real-time guarantee.
Input sent, result completion, resource release and playback end are distinct.
Pause/seek do not cancel inference; future, unreceived time is not seekable.
Seek updates the playhead, head and channel values immediately. Audio stream
teardown/setup runs on a worker and coalesces rapid seeks to the latest position;
stale callbacks are silenced and cannot advance the new media clock. Seeking
while paused does not open an audio device. Device failures arrive asynchronously
through the audio error slot and pause playback.
Results are limited to 256 MiB and 10 minutes; exceeding a limit marks the result
failed. Logs retain 10,000 records with a 2,048-record nonblocking queue and a
visible dropped count. Closing waits for cancellation/shutdown; native model
initialization already in progress may take time before it can finish cleanup.

## GUI configuration

Copy [`gui.example.toml`](../gui.example.toml) to `gui.toml` in the working
directory or next to the executable. Configuration selection is:

1. The file explicitly selected with `--config PATH` (relative to the working directory).
2. `gui.toml` in the working directory.
3. `gui.toml` next to the executable.
4. Built-in defaults if neither automatic location contains a file.

Only one file is read; files are not merged. Missing explicit files, invalid files,
unknown keys and access errors are errors, without falling back to another file.
Paths inside the selected file remain relative to that file's directory.
GUI edits do not write back to TOML. Thus `cargo run` from the repository root
uses the root `gui.toml` without `--config`, even though the executable is under
`target/debug` or `target/release`.

Local `gui.toml` and `platform.toml` files are ignored by Git; examples are tracked.

```powershell
./audio2face3d-gui.exe --config C:/config/gui.toml
./audio2face3d-gui.exe --config C:/config/gui.toml --infer=false --play-while-inferring=false
```

The file configures `head`, `platform-config`, `[inference]` (`mode`, `wav`,
`play-while-inferring`, `auto-start`), `[local]` (`model`, `device`), and `[grpc]`
(`endpoint`, `api-key`). Unspecified values use application defaults. Explicit CLI
values override file values, including `--device 0`, `--infer=false` and
`--play-while-inferring=false`. Validation for automatic inference happens after
merging, so the WAV/model can come from either source. Without auto-start, files
are selected for later use and inference does not begin.

All paths written in a GUI TOML are relative to that TOML's directory. A referenced
`platform.toml` resolves its own paths relative to its own directory, independently
of the GUI file. Absolute paths are preserved. CLI paths are relative to the working
directory. Omitting `head` retains the executable-relative standard head fallback.

CUDA/TensorRT settings are not duplicated in GUI TOML. Set `platform-config` to
reference the same file used by the other CLIs. See
[`platform.example.toml`](../platform.example.toml) for a shared configuration template.

## Startup options and platform.toml

The standard executable uses the same `PlatformArgs` resolver as the inference
and server CLIs. File selection is: `--platform-config FILE`, then GUI TOML `platform-config`, then
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
`--model JSON`, `--wav WAV`, `--endpoint URL`, `--api-key KEY`, and `--device INDEX` populate
controls. `mock` requires the development feature. `--infer` also starts inference;
otherwise the GUI TOML auto-start setting applies (false by default). `--play-while-inferring` enables
paced streaming playback. Existing shared flags `--cuda-root`, `--tensorrt-root`,
`--cuda-library-dir`, `--tensorrt-library-dir` and `--runtime-search` override the
selected file with the same rules as the server CLI.

Runtime roots, directory lists and search policy are retained in the request and
passed into the inference context. To change them, update the configuration and
restart the application.
Embedded hosts provide `startup::Options` or `Request::runtime` explicitly;
constructing a library request never reads process arguments or config files.

## Build and package from source

Rust 2024 / Rust 1.91+, Windows x64 MSVC. From the checkout:

```powershell
cargo run -p audio2face3d-gui --no-default-features --features standalone-app,grpc -- crates/audio2face3d-gui/assets/default-head.glb
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
`docs/platform.md`. CUDA 12.9 was validated with MSVC **14.42.34433**. Configure
`[build-cuda.windows]` with the Visual Studio root and exact toolset version;
the native build initializes matching compiler, header and library paths automatically.
The `standalone-app,grpc` build needs no CUDA/TensorRT or `build-cuda` settings.
No developer SDK path is compiled into the application configuration.

## Convert or replace the head

Use the CPU-only OBJ converter described in [headgen.md](headgen.md):

```powershell
cargo run -p audio2face3d-headgen -- convert --config crates/audio2face3d-headgen/presets/ict-facekit.toml --input-root C:/Data/ICT-FaceKit/FaceXModel --output temp/ict-facekit.glb
```

Open the resulting GLB with **Open head** or `--head`. The head's unsupported
channels are displayed separately; inference values and timeline tracks stay intact.
The original bundled mannequin remains a diagnostic fallback. It is not ICT data.
Third-party inputs and converted assets are obtained and managed by the user;
no automatic downloads or package replacement occur.

## Library boundaries and engine embedding

- `audio2face3d-gui-core`: CPU model types, validation and optional GLB read/write.
- `audio2face3d-headgen`: CPU OBJ conversion library and standard CLI (no feature flag).
- `audio2face3d-gui`: playback/session/logging library; `render-wgpu` and `ui-egui`
  are optional. `standalone-app` adds the standard CPAL/eframe host. `local` and `grpc`
  independently enable inference. `mock` and `capture` are development features.

All default feature sets are empty. A host can feed `Clip`, call `Player` with
its audio callback and scheduled audible timestamp, and obtain one `Snapshot`
for the head/values/timeline. `HeadRenderer` accepts a host-owned wgpu device,
queue and render target; the host submits commands and owns texture lifetime.
No global tracing subscriber is installed. Pass the existing `Logger` interface
or `FanoutLogger` to inference. GUI log callbacks enqueue; engine loggers needing
thread affinity must enqueue on the receiving side too.

Unreal editor panels, C ABI, native texture sharing and RHI synchronization are
future integration work. A wgpu texture is not assumed to be directly shareable with an
engine device. ACE wire-compatible gRPC tests are separate from Unreal plugin
playback verification.

## Validation

`ci/run-tier.ps1 portable` includes CPU model/GLB/converter/playback/log/WAV/Mock
tests and desktop gRPC compilation. Hardware tests are intentionally separate:

```powershell
cargo test -p audio2face3d-gui --features render-wgpu --test gpu -- --ignored --nocapture
cargo run -p audio2face3d-gui --features standalone-app --example audio_probe
cargo run --release -p audio2face3d-gui --features standalone-app,local,grpc --example stream_probe -- local input.wav models/mark/model.json
```

The stream probe requires a long enough input to start audible playback before
input completion. `infer_probe` compares ordinary results and accepts `local`,
`grpc` or explicit development `mock`. Capture helpers live under `examples`.
When recording performance, distinguish a fresh inference context with cached
engine files from engine generation and retained, prewarmed contexts. Portable
tests do not establish GPU performance, acoustic latency or Unreal ACE plugin
playback compatibility; validate those separately on the target system.
