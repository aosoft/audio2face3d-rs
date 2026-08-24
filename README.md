# audio2face-3d-rs

Rust 2024 implementation of the NVIDIA Audio2Face-3D and Audio2Emotion runtime pipelines. TensorRT engines and model data are not bundled or downloaded during builds.

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
cargo run -p audio2x-model-tool -- doctor
```

## Explicit model acquisition

First accept the applicable NVIDIA model license and configure a Hugging Face access token. Downloads are explicit and never happen from `build.rs` or model loading:

```sh
export HF_TOKEN=...
cargo run -p audio2x-model-tool -- list
cargo run -p audio2x-model-tool -- download mark
cargo run -p audio2x-model-tool -- download all
```

The built-in catalog covers `diffusion`, `claire`, `james`, `mark`, and `emotion`. It pins the repository and full commit revision and installs under `./models/<preset>` by default. Pass a different output root and token environment after the preset when needed. Arbitrary immutable revisions remain available through `download-revision <owner/repository> <revision> <output> [token-env]`.

When an output already exists, the tool verifies its repository, revision, required files, and actual `network.onnx` SHA-256 against `.audio2x-source.json`. A matching snapshot is skipped without network access. A mismatch is preserved and reported; pass `--force` to download, validate, and safely replace it:

```sh
cargo run -p audio2x-model-tool -- download diffusion --force
```

The dedicated Rust tool uses the Hugging Face Hub API directly; it does not launch Python or the `hf` CLI. An absent token is reported before any network request. HTTP 401/403, gated-repository, revision, and rate-limit failures are reported as structured download errors.

During a download the tool reports overall bytes, percentage, completed files, and transfer rate. Interactive terminals reuse one line; redirected output emits periodic log lines instead.

Each download is staged in a sibling temporary directory, checked for the Audio2X model files, and atomically installed. The tool writes `.audio2x-source.json` with the repository, immutable revision, and `network.onnx` SHA-256.

## TensorRT engine generation

Generate an environment-specific TensorRT engine from a downloaded preset with the same Rust tool. It expands the optimization profiles in the model's `trt_info.json` and invokes `TRTEXEC` or `trtexec` directly:

```sh
cargo run -p audio2x-model-tool -- engine mark
cargo run -p audio2x-model-tool -- engine mark --precision=fp16
cargo run -p audio2x-model-tool -- engine mark --precision=fp32 --device=0
cargo run -p audio2x-model-tool -- engine emotion --max-batch=32
```

`default` matches the original SDK's standard build: FP32 is available and TensorRT may use TF32. `fp16` enables mixed FP16/FP32 execution. Explicit `fp32` disables TF32. The generated artifacts are:

| Precision | Engine | TensorRT metadata | Runtime descriptor |
|---|---|---|---|
| `default` | `network.trt` | existing `trt_info.json` | existing `model.json` |
| `fp16` | `network_fp16.trt` | `trt_info_fp16.json` | `model_fp16.json` |
| `fp32` | `network_fp32.trt` | `trt_info_fp32.json` | `model_fp32.json` |

The FP16 names match the original SDK. ONNX is only an engine-build input; runtime samples load the generated model descriptor and its referenced `.trt` file.

Engine generation starts with the distributed TensorRT profile, including Audio2Emotion's `MAX_BATCH_SIZE=128`. If TensorRT explicitly reports that available device memory cannot support any tactic, an unspecified maximum automatically retries at half the batch size until it succeeds (`128 -> 64 -> 32 -> ... -> 1`). Other build failures stop immediately. `--max-batch=N` disables fallback and strictly uses the requested value; `OPT_BATCH_SIZE` is clamped when it exceeds that value. The runtime reads the generated engine profile and accepts up to the selected maximum. The tool passes `--skipInference` because this command builds an engine rather than benchmarking it.

The tool records the ONNX and engine hashes, expanded arguments, automatic/explicit batch policy, attempted and selected batch sizes, device, `trtexec` binary hash/version, CUDA toolkit, and GPU/driver identity in a precision-specific `.audio2x-engine*.json` sidecar. A matching existing engine is verified and skipped. A mismatch is preserved and reported; use `--replace` to stage, validate, and replace the selected precision's complete artifact set with rollback on installation failure.

An engine created by an earlier tool version with `--max-batch=N` remains verifiable with the same explicit option. Use `--replace` once, without `--max-batch`, to replace it with an automatically selected profile.

Download/verification and engine generation can be combined. `--force` applies only to the downloaded snapshot and `--replace` only to the selected engine artifacts:

```sh
cargo run -p audio2x-model-tool -- prepare mark --precision=fp16
cargo run -p audio2x-model-tool -- prepare all --device=0
```

Engine generation can take several minutes per model. `trtexec` output is streamed to the console. Set an explicit executable when it is not on `PATH`:

```powershell
$env:TRTEXEC = 'C:\SDK\TensorRT-10.16.1.11\bin\trtexec.exe'
```

## Samples

The samples accept a resolved `model.json`, track count, and optional number of zero-valued audio samples:

```sh
cargo run -p audio2x --all-features --bin audio2x-regression -- ./models/mark/model.json 1 16000
cargo run -p audio2x --all-features --bin audio2x-regression -- ./models/mark/model_fp16.json 1 16000
cargo run -p audio2x --all-features --bin audio2x-diffusion -- ./models/diffusion/model.json 1 16000
cargo run -p audio2x --all-features --bin audio2x-emotion -- ./models/emotion/model.json 1 16000
```

The `audio2x` facade resolves model-relative paths, constructs per-track accumulators, owns the selected TensorRT backend, exposes callback metadata, and validates track parameter updates.

## Benchmarks

The benchmark command separates build, descriptor-cache, warm-up, steady-state inference, post-process, and end-to-end phases. It reports P50/P95/P99 nanoseconds and peak process GPU memory when `nvidia-smi` exposes it. On Windows WDDM, where per-process accounting can be unavailable, it labels and reports device-wide used memory instead:

```sh
cargo run --release -p audio2x --all-features --bin audio2x-benchmark -- ./models/mark/model.json 1 fp32 100
cargo run --release -p audio2x --all-features --bin audio2x-benchmark -- ./models/mark/model_fp16.json 2 fp16 100
```

For geometry models, the isolated post-process phase currently measures the device-result consumption boundary; full animator and blendshape timing must be reported separately from raw TensorRT inference. Compare C++ and Rust only with identical model/engine, batch, precision, GPU, driver, CUDA, and TensorRT versions.
