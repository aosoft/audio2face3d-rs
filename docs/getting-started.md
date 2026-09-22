# Getting started

[Project overview](../README.md) · [Documentation index](../README.md#documentation)

Run these commands from the workspace root. Native inference requires separately installed SDKs and prepared models.

## Install from this checkout

The packages are not yet published to crates.io. After configuring the SDK locations below, install the executables from source:

```powershell
cargo install --path crates/audio2face3d --locked --features cli,native
cargo install --path crates/audio2face3d-server --locked --features cli
```

For a portable diagnostic server, use `--no-default-features --features cli,mock` when installing `audio2face3d-server`. Cargo features are selected at build time. For server startup and library embedding, see the [server guide](server.md).

## Runtime setup

CUDA 12 and TensorRT 10 must be installed separately. The initial validated versions are CUDA 12.9 and TensorRT 10.16.1.11.

Building the native features also requires a host C++ compiler for the TensorRT
shim and CUDA compilation. On Windows, install the MSVC C++ build tools;
the Visual Studio IDE itself is not required. The default portable features
do not compile the CUDA kernels or TensorRT shim.

Optionally configure both build and runtime with one `platform.toml`. Without a selected or discovered file, the existing SDK environment variables and runtime search paths remain available. See [platform configuration](platform.md) for the complete file format, search precedence, and library embedding.

For example, after preparing the file described there:

```powershell
cargo run -p audio2face3d --features cli -- --platform-config platform.toml doctor
cargo run -p audio2face3d --features cli,native -- --platform-config platform.toml doctor --load --device 0
```

The samples below automatically read `platform.toml` from this checkout; add `--platform-config platform.toml` after Cargo's `--` separator to select another runtime file with any command. Runtime minor differences warn, major differences fail, and patch/build differences are allowed. Required API or engine incompatibilities still fail.

The model tool uses `clap` for argument parsing. Top-level help, command-specific options, accepted values, defaults, and the package version are available directly from the CLI:

```sh
cargo run -p audio2face3d --features cli -- --help
cargo run -p audio2face3d --features cli -- model engine --help
cargo run -p audio2face3d --features cli -- --version
```

## Explicit model acquisition

First obtain access on the official Hugging Face model page, then configure
your own access token. Downloads are explicit and never happen from `build.rs`
or model loading:

PowerShell:

```powershell
$env:HF_TOKEN = '...'
```

Bash/sh:

```sh
export HF_TOKEN='...'
```

Then list and download models:

```sh
cargo run -p audio2face3d --features cli -- model list
cargo run -p audio2face3d --features cli -- model download mark
cargo run -p audio2face3d --features cli -- model download all
```

The built-in catalog covers `diffusion`, `claire`, `james`, `mark`, and `emotion`.
Each user obtains model snapshots directly from NVIDIA's official Hugging Face
repositories using their own token; this project does not redistribute models.
The catalog pins the repository and full commit revision and installs under
`./models/<preset>` by default. Pass a different output root and token environment
after the preset when needed. Arbitrary immutable revisions remain available
through `download-revision <owner/repository> <revision> <output> [token-env]`.

When an output already exists, the tool verifies its repository, revision, required files, and actual `network.onnx` SHA-256 against `.audio2x-source.json`. A matching snapshot is skipped without network access. A mismatch is preserved and reported; pass `--force` to download, validate, and safely replace it:

```sh
cargo run -p audio2face3d --features cli -- model download diffusion --force
```

The dedicated Rust tool uses the Hugging Face Hub API directly; it does not launch Python or the `hf` CLI. An absent token is reported before any network request. HTTP 401/403, gated-repository, revision, and rate-limit failures are reported as structured download errors.

During a download the tool reports overall bytes, percentage, completed files, and transfer rate. Interactive terminals reuse one line; redirected output emits periodic log lines instead.

Each download is staged in a sibling temporary directory, checked for the required Audio2Face-3D model files, and atomically installed. The tool writes the existing `.audio2x-source.json` provenance format with the repository, immutable revision, and `network.onnx` SHA-256.

## TensorRT engine generation

Generate an environment-specific TensorRT engine from a downloaded preset with the same Rust tool. It expands the optimization profiles in the model's `trt_info.json` and invokes `TRTEXEC` or `trtexec` directly:

```sh
cargo run -p audio2face3d --features cli -- model engine mark
cargo run -p audio2face3d --features cli -- model engine mark --precision=fp16
cargo run -p audio2face3d --features cli -- model engine mark --precision=fp32 --device=0
cargo run -p audio2face3d --features cli -- model engine emotion --max-batch=32
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
cargo run -p audio2face3d --features cli -- model prepare mark --precision=fp16
cargo run -p audio2face3d --features cli -- model prepare all --device=0
```

Engine generation can take several minutes per model. `trtexec` output is streamed to the console. It is found under the configured TensorRT location; its child process receives the selected SDK search paths. An optional explicit executable override is also available:

```powershell
$env:TRTEXEC = 'C:\SDK\TensorRT-10.16.1.11\bin\trtexec.exe'
```

## Samples

The samples accept a resolved `model.json`, track count, and optional number of zero-valued audio samples:

```sh
cargo run -p audio2face3d --features cli,native -- run regression ./models/mark/model.json 1 16000
cargo run -p audio2face3d --features cli,native -- run regression ./models/mark/model_fp16.json 1 16000
cargo run -p audio2face3d --features cli,native -- run diffusion ./models/diffusion/model.json 1 16000
cargo run -p audio2face3d --features cli,native -- run emotion ./models/emotion/model.json 1 16000
```

The `audio2face3d` facade resolves model-relative paths, constructs per-track accumulators, owns the selected TensorRT backend, exposes callback metadata, and validates track parameter updates.
