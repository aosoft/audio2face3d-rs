# Audio2Face-3D for Rust

Rust implementation of the NVIDIA Audio2Face-3D and Audio2Emotion runtime pipelines. TensorRT engines and model data are not bundled or downloaded during builds.

The workspace is organized as two publishable packages:

| Package | Purpose |
|---|---|
| `audio2face3d` | Library containing the common, CUDA, TensorRT, animation, emotion, and unified pipeline modules |
| `audio2face3d-cli` | Single `audio2face3d` executable for model management, samples, and benchmarks |

The Rust modules correspond to the original SDK as follows:

| NVIDIA SDK component | Rust module |
|---|---|
| `audio2x-sdk` | `audio2face3d` facade and pipeline |
| `audio2x-common` schemas and accumulators | `audio2face3d::common` |
| `audio2x-common` CUDA support | `audio2face3d::cuda` |
| `audio2x-common` TensorRT support | `audio2face3d::tensorrt` |
| `audio2face-sdk` | `audio2face3d::audio2face` |
| `audio2emotion-sdk` | `audio2face3d::audio2emotion` |

## Release support contract

The required release target is Windows x86-64 with MSVC, CUDA 12.9.x,
TensorRT 10.16.1.x, an SM 8.6 GPU, and both FP32 and FP16 engines. Linux
x86-64 with the same CUDA and TensorRT families is currently build-only and
experimental. The minimum supported Rust version is 1.87.

| Feature | Contents | Required tier |
|---|---|---|
| default | CPU animation and emotion processing | `portable` |
| `animation` | geometry, BlendShape, Teeth, and interactive APIs | `portable` |
| `emotion` | Audio2Emotion post-processing | `portable` |
| `cuda` | CUDA buffers, streams, solvers, and lifetime tests | `cuda-lifetime` |
| `tensorrt` | real TensorRT model execution | `tensorrt-model` |

The machine-readable contract is `release/release-baseline.json`. Its five CI
tiers deliberately separate portable tests, CUDA ownership tests, real-model
tests, original-SDK reference parity, and package/release checks:

```powershell
./ci/run-tier.ps1 portable
./ci/run-tier.ps1 cuda-lifetime
./ci/run-tier.ps1 tensorrt-model
./ci/run-tier.ps1 reference-parity
./ci/run-tier.ps1 release
```

Native tiers require the environment variables listed in the release contract;
machine-local SDK/model/audio paths and generated captures are never committed.
Audit the contract, pinned revisions, licenses, artifact sizes/hashes, benchmark
definitions, API policy, and safety invariants with:

```sh
cargo run -p audio2face3d-cli -- release audit
```

### Inference-free Audio2Emotion

`PostProcessEmotionExecutorFactory` creates the original post-process-only
executor without loading a TensorRT engine. Its `load` method returns a
runtime-independent Future. Audio samples define the duration but are not
read; each frame feeds zero classifier output plus the configured or
accumulated preferred emotion into the post-processor.

```rust,no_run
use audio2face3d::audio2emotion::EmotionExecutor;
use audio2face3d::audio2emotion::post_process::{
    PostProcessEmotionExecutorCreationParameters, PostProcessEmotionExecutorFactory,
};

async fn run(parameters: PostProcessEmotionExecutorCreationParameters) -> audio2face3d::Result<()> {
let mut executor = PostProcessEmotionExecutorFactory::load(parameters).await?;
let mut callback = |_result| std::ops::ControlFlow::Continue(());
executor.execute(&mut callback)?.await?;
Ok(())
}
```

## Component composition

`GeometryExecutorBundleFactory` wraps the model-specific Regression or
Diffusion factory in a closed, non-generic owning bundle. `load` returns a
runtime-independent Future and retains model kind, track resources, CUDA
stream, and output validation in the completed executor:

```rust,no_run
use audio2face3d::audio2face::{
    GeometryExecutorBundle, GeometryExecutorBundleCreationParameters,
    GeometryExecutorBundleFactory,
};

async fn load(
    parameters: GeometryExecutorBundleCreationParameters,
) -> audio2face3d::Result<GeometryExecutorBundle> {
    GeometryExecutorBundleFactory::load(parameters).await
}
```

After loading, accumulate and close the shared track accumulators. Match the
closed bundle once to call `GeometryExecutor::execute`, copy any required
`DeviceComponentResults` with `copy_to` inside the callback, and await the
returned `Execution`.

The closed bundle exposes `executor()`, `cuda_stream()`,
`audio_accumulator()`, and `emotion_accumulator()` accessors. Geometry can be
consumed exactly once by `try_into_host_blendshape` or
`try_into_device_blendshape`; failed transfers return the original bundle in a
`TransferError`.

## Interactive and GPU callback quick start

Interactive execution keeps inference, post-processing, and BlendShape caches
inside one owner. Callback device views are valid only for the callback and
cannot be retained by safe Rust:

```rust,no_run
use audio2face3d::audio2face::GeometryInteractiveExecutor;
use audio2face3d::audio2face::regression::{
    RegressionGeometryInteractiveExecutorCreationParameters,
    RegressionGeometryInteractiveExecutorFactory,
};

async fn run(
    parameters: RegressionGeometryInteractiveExecutorCreationParameters,
) -> audio2face3d::Result<()> {
let mut executor = RegressionGeometryInteractiveExecutorFactory::load(parameters).await?;
let mut callback = |_result| std::ops::ControlFlow::Continue(());
executor.compute_frame(0, &mut callback).await?;
Ok(())
}
```

Use `compute_all_frames` for an ordered temporal pass. Use `compute_frame` for
random access and replay; invalidating geometry automatically invalidates the
dependent cache. Host/device BlendShape conversion is exposed by the owning
geometry bundle and all device views require an explicit copy on their stream.

## Standalone teeth animation

`AnimatorTeeth` is the host-side semantic name for the teeth transform and
matches the original `IAnimatorTeeth` parameter and column-major transform
contract. Create it through `create_animator_teeth`; the lower-level
`JawTransform` helper remains an implementation detail. GPU solver work is
returned through the owning device executor and its stream-scoped results.

## Runtime setup

CUDA 12 and TensorRT 10 must be installed separately. The initial validated versions are CUDA 12.9 and TensorRT 10.16.1.11.

On Windows, set `CUDA_PATH` and `TENSORRT_ROOT_DIR`, then add their `bin` directories to `PATH`:

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
$env:TENSORRT_ROOT_DIR = 'C:\SDK\TensorRT-10.16.1.11'
$env:PATH = "$env:CUDA_PATH\bin;$env:TENSORRT_ROOT_DIR\bin;$env:PATH"
```

On Linux, set the roots and loader path:

```sh
export CUDA_PATH=/usr/local/cuda-12.9
export TENSORRT_ROOT_DIR=/opt/TensorRT-10.16.1.11
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$TENSORRT_ROOT_DIR/lib:$LD_LIBRARY_PATH"
```

Check runtime discovery before loading a model:

```sh
cargo run -p audio2face3d-cli -- doctor
```

The model tool uses `clap` for argument parsing. Top-level help, command-specific options, accepted values, defaults, and the package version are available directly from the CLI:

```sh
cargo run -p audio2face3d-cli -- --help
cargo run -p audio2face3d-cli -- model engine --help
cargo run -p audio2face3d-cli -- --version
```

## Explicit model acquisition

First accept the applicable NVIDIA model license and configure a Hugging Face access token. Downloads are explicit and never happen from `build.rs` or model loading:

```sh
export HF_TOKEN=...
cargo run -p audio2face3d-cli -- model list
cargo run -p audio2face3d-cli -- model download mark
cargo run -p audio2face3d-cli -- model download all
```

The built-in catalog covers `diffusion`, `claire`, `james`, `mark`, and `emotion`. It pins the repository and full commit revision and installs under `./models/<preset>` by default. Pass a different output root and token environment after the preset when needed. Arbitrary immutable revisions remain available through `download-revision <owner/repository> <revision> <output> [token-env]`.

When an output already exists, the tool verifies its repository, revision, required files, and actual `network.onnx` SHA-256 against `.audio2x-source.json`. A matching snapshot is skipped without network access. A mismatch is preserved and reported; pass `--force` to download, validate, and safely replace it:

```sh
cargo run -p audio2face3d-cli -- model download diffusion --force
```

The dedicated Rust tool uses the Hugging Face Hub API directly; it does not launch Python or the `hf` CLI. An absent token is reported before any network request. HTTP 401/403, gated-repository, revision, and rate-limit failures are reported as structured download errors.

During a download the tool reports overall bytes, percentage, completed files, and transfer rate. Interactive terminals reuse one line; redirected output emits periodic log lines instead.

Each download is staged in a sibling temporary directory, checked for the required Audio2Face-3D model files, and atomically installed. The tool writes the existing `.audio2x-source.json` provenance format with the repository, immutable revision, and `network.onnx` SHA-256.

## TensorRT engine generation

Generate an environment-specific TensorRT engine from a downloaded preset with the same Rust tool. It expands the optimization profiles in the model's `trt_info.json` and invokes `TRTEXEC` or `trtexec` directly:

```sh
cargo run -p audio2face3d-cli -- model engine mark
cargo run -p audio2face3d-cli -- model engine mark --precision=fp16
cargo run -p audio2face3d-cli -- model engine mark --precision=fp32 --device=0
cargo run -p audio2face3d-cli -- model engine emotion --max-batch=32
```

`default` matches the original SDK's standard build: FP32 is available and TensorRT may use TF32. `fp16` enables mixed FP16/FP32 execution. Explicit `fp32` disables TF32. The generated artifacts are:

| Precision | Engine | TensorRT metadata | Runtime descriptor |
|---|---|---|---|
| `default` | `network.trt` | existing `trt_info.json` | existing `model.json` |
| `fp16` | `network_fp16.trt` | `trt_info_fp16.json` | `model_fp16.json` |
| `fp32` | `network_fp32.trt` | `trt_info_fp32.json` | `model_fp32.json` |

The FP16 names match the original SDK. ONNX is only an engine-build input; runtime samples load the generated model descriptor and its referenced `.trt` file.

Engine generation starts with the distributed TensorRT profile, including Audio2Emotion's `MAX_BATCH_SIZE=128`. If TensorRT explicitly reports that available device memory cannot support any tactic, an unspecified maximum automatically retries at half the batch size until it succeeds (`128 -> 64 -> 32 -> ... -> 1`). Other build failures stop immediately. `--max-batch=N` disables fallback and strictly uses the requested value; `OPT_BATCH_SIZE` is clamped when it exceeds that value. The runtime reads the generated engine profile and accepts up to the selected maximum. The tool passes `--skipInference` because this command builds an engine rather than benchmarking it.

The tool records the ONNX and engine hashes, expanded arguments, automatic/explicit batch policy, attempted and selected batch sizes, device, `trtexec` binary hash/version, CUDA toolkit, and GPU/driver identity in a precision-specific `.audio2x-engine*.json` sidecar. A matching existing engine is verified and skipped. A mismatch is preserved and reported; use `--force` to stage, validate, and replace the selected precision's complete artifact set with rollback on installation failure.

An engine created by an earlier tool version with `--max-batch=N` remains verifiable with the same explicit option. Use `--force` once, without `--max-batch`, to replace it with an automatically selected profile.

Download/verification and engine generation can be combined. For `prepare`, `--force` replaces both a mismatched downloaded snapshot and the selected engine artifacts, while preserving both until their respective replacements validate:

```sh
cargo run -p audio2face3d-cli -- model prepare mark --precision=fp16
cargo run -p audio2face3d-cli -- model prepare all --device=0
```

Engine generation can take several minutes per model. `trtexec` output is streamed to the console. Set an explicit executable when it is not on `PATH`:

```powershell
$env:TRTEXEC = 'C:\SDK\TensorRT-10.16.1.11\bin\trtexec.exe'
```

## Samples

The samples accept a resolved `model.json`, track count, and optional number of zero-valued audio samples:

```sh
cargo run -p audio2face3d-cli --features runtime -- run regression ./models/mark/model.json 1 16000
cargo run -p audio2face3d-cli --features runtime -- run regression ./models/mark/model_fp16.json 1 16000
cargo run -p audio2face3d-cli --features runtime -- run diffusion ./models/diffusion/model.json 1 16000
cargo run -p audio2face3d-cli --features runtime -- run emotion ./models/emotion/model.json 1 16000
```

The `audio2face3d` facade resolves model-relative paths, constructs per-track accumulators, owns the selected TensorRT backend, exposes callback metadata, and validates track parameter updates.

## Benchmarks

The benchmark command separates build, descriptor-cache, warm-up, steady-state inference, post-process, and end-to-end phases. It reports P50/P95/P99 nanoseconds and peak process GPU memory when `nvidia-smi` exposes it. On Windows WDDM, where per-process accounting can be unavailable, it labels and reports device-wide used memory instead:

```sh
cargo run --release -p audio2face3d-cli --features runtime -- benchmark ./models/mark/model.json 1 fp32 100
cargo run --release -p audio2face3d-cli --features runtime -- benchmark ./models/mark/model_fp16.json 2 fp16 100
cargo run --release -p audio2face3d-cli --features runtime -- benchmark ./models/mark/model.json 1 fp32 100 --scope blendshape-cpu
cargo run --release -p audio2face3d-cli --features runtime -- benchmark ./models/mark/model.json 1 fp32 100 --scope blendshape-gpu
cargo run --release -p audio2face3d-cli --features runtime -- benchmark ./models/mark/model.json 1 fp32 100 --scope interactive-gpu-replay
cargo run --release -p audio2face3d-cli --features runtime -- benchmark ./models/mark/model.json 1 fp32 100 --output reference/compatible_test/benchmarks/mark.json
```

JSON output records the model and engine SHA-256, execution environment,
P50/P95/P99 latency, throughput, and peak GPU memory. Compare performance
against the tracked hardware baseline independently from reference-value
parity:

```sh
cargo run -p audio2face3d-cli -- release benchmark-compare reference/benchmark-baseline.json reference/compatible_test/benchmarks/mark.json
```

For geometry models, the isolated post-process phase currently measures the device-result consumption boundary; full animator and blendshape timing must be reported separately from raw TensorRT inference. Compare C++ and Rust only with identical model/engine, batch, precision, GPU, driver, CUDA, and TensorRT versions.

## Safety, errors, and versioning

CUDA streams, events, buffers, GPU solvers, and TensorRT sessions are
thread-affine (`!Send`/`!Sync`). Device views borrow their allocations, and
asynchronous fences retain all borrowed resources until queued CUDA work is
observed complete. Device identity is checked before composition. Interactive
cache eviction, invalidation, and destruction synchronize before releasing
referenced buffers. TensorRT execution contexts require mutable access and
cannot be enqueued concurrently.

Every TensorRT C++ shim entry catches C++ exceptions before returning through
the C ABI. Public operations report malformed input, schema, device, CUDA, and
TensorRT failures through `audio2face3d::Result`; panics are not part of the
public error contract. Compile-fail tests fix the view/fence lifetime and
thread-affinity rules, while `release_stress` covers long input, maximum track
limits, reset/replay, queued work, and drop behavior.

This project follows SemVer 2.0.0. Before 1.0, minor releases may change the
public API; patch releases remain compatible. Removing a public feature is a
breaking change, and every release requires a public-API and feature review.

## License

The source code in this repository is available under the [MIT License](LICENSE).
It includes work derived from the NVIDIA Audio2Face-3D SDK and retains the
applicable NVIDIA copyright and license notice.

The MIT License does not apply to NVIDIA model files or externally installed
CUDA and TensorRT SDK binaries, headers, and components. Those are distributed
separately and remain subject to their respective NVIDIA license terms. In
particular, Audio2Face-3D models use the NVIDIA Open Model License, while
Audio2Emotion models use their model-specific license agreement.
