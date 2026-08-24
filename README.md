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

The dedicated Rust tool uses the Hugging Face Hub API directly; it does not launch Python or the `hf` CLI. An absent token is reported before any network request. HTTP 401/403, gated-repository, revision, and rate-limit failures are reported as structured download errors.

Each download is staged in a sibling temporary directory, checked for the Audio2X model files, and atomically installed. The tool writes `.audio2x-source.json` with the repository, immutable revision, and `network.onnx` SHA-256. The downloaded descriptor already names `network.trt`; generate that environment-specific engine with `audio2x-engine` and the profiles in the downloaded `trt_info.json` before running a sample.

## Samples

The samples accept a resolved `model.json`, track count, and optional number of zero-valued audio samples:

```sh
cargo run -p audio2x --all-features --bin audio2x-regression -- ./models/mark/model.json 1 16000
cargo run -p audio2x --all-features --bin audio2x-diffusion -- ./models/diffusion/model.json 1 16000
cargo run -p audio2x --all-features --bin audio2x-emotion -- ./models/emotion/model.json 1 16000
```

The `audio2x` facade resolves model-relative paths, constructs per-track accumulators, owns the selected TensorRT backend, exposes callback metadata, and validates track parameter updates.

## Benchmarks

The benchmark command separates build, descriptor-cache, warm-up, steady-state inference, post-process, and end-to-end phases. It reports P50/P95/P99 nanoseconds and peak process GPU memory when `nvidia-smi` exposes it. On Windows WDDM, where per-process accounting can be unavailable, it labels and reports device-wide used memory instead:

```sh
cargo run --release -p audio2x --all-features --bin audio2x-benchmark -- ./models/mark/model.json 1 fp32 100
cargo run --release -p audio2x --all-features --bin audio2x-benchmark -- ./models/mark/model.json 2 fp16 100 ./models/mark/network-fp16.trt
```

For geometry models, the isolated post-process phase currently measures the device-result consumption boundary; full animator and blendshape timing must be reported separately from raw TensorRT inference. Compare C++ and Rust only with identical model/engine, batch, precision, GPU, driver, CUDA, and TensorRT versions.
