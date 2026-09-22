# Development and validation

[Project overview](../README.md) · [Documentation index](../README.md#documentation)

Support targets, test tiers, reference parity, and benchmark procedures.

## Release support contract

The required release target is Windows x86-64 with MSVC, CUDA 12.9.x,
TensorRT 10.16.1.x, an SM 8.6 GPU, and both FP32 and FP16 engines. Linux
x86-64 with the same CUDA and TensorRT families is currently build-only and
experimental. The minimum supported Rust version is 1.91.

The CI tiers separate portable tests, CUDA ownership tests, real-model tests,
and original-SDK reference parity:

```powershell
./ci/run-tier.ps1 portable
./ci/run-tier.ps1 cuda-lifetime
./ci/run-tier.ps1 tensorrt-model
./ci/run-tier.ps1 reference-parity
```

Native tiers accept `-PlatformConfig platform.toml`. Runtime SDK locations must resolve to roots (from common settings or `[runtime]`) for legacy test adapters; only child processes receive their SDK environment. Existing environment-based runs still work. See [platform configuration](platform.md). Real-model tests require `AUDIO2FACE3D_TEST_FACADE_MODELS`. See the
[reference guide](../reference/README.md) for original-SDK comparison setup.
Machine-local SDK/model/audio paths and generated captures are not committed.
The `release` tier verifies native documentation and publishable packages.
The optional `release audit` command requires an explicit `--baseline` path
and the corresponding `--workspace`; it is not part of CI or publication.

## Validation

Public API and feature changes are reviewed in pull request diffs.
Portable CI checks formatting, builds, linting, tests, and documentation across
the portable feature configurations. It runs automatically
on pull requests and can also be run manually. Other workflows remain manual-only.
CUDA/TensorRT validation requires CUDA and TensorRT
installations; original-SDK reference comparison additionally requires the
Audio2Face-3D SDK checkout.
Passing portable checks does not imply model-runtime validation.

## CI build cache

Both portable CI jobs use GitHub's `actions/cache` to preserve Cargo registry/git
dependencies and `target` build outputs. Incremental compilation artifacts and
generated API documentation are excluded to limit cache size.

Keys separate operating system, architecture, job, and pinned Rust version.
They also hash Cargo manifests, the lockfile, Cargo configuration, and portable
CI definitions. When these files change, a fallback can restore the same
OS/architecture/job/Rust cache; Cargo still checks whether artifacts can be reused.
Update the Rust version in the cache key when changing the job's toolchain.

All checks and tests run even on a cache hit. The first successful run populates
the cache; compare subsequent runs to measure the benefit. Cache availability
follows GitHub's branch scope: a PR cache is not shared with unrelated PRs.
Run the workflow manually on `main` to seed a base-branch cache for future PRs.

## Benchmarks

The benchmark command separates build, descriptor-cache, warm-up, steady-state inference, post-process, and end-to-end phases. It reports P50/P95/P99 nanoseconds and peak process GPU memory when `nvidia-smi` exposes it. On Windows WDDM, where per-process accounting can be unavailable, it labels and reports device-wide used memory instead:

```sh
cargo run --release -p audio2face3d --features cli,native -- benchmark ./models/mark/model.json 1 fp32 100
cargo run --release -p audio2face3d --features cli,native -- benchmark ./models/mark/model_fp16.json 2 fp16 100
cargo run --release -p audio2face3d --features cli,native -- benchmark ./models/mark/model.json 1 fp32 100 --scope blendshape-cpu
cargo run --release -p audio2face3d --features cli,native -- benchmark ./models/mark/model.json 1 fp32 100 --scope blendshape-gpu
cargo run --release -p audio2face3d --features cli,native -- benchmark ./models/mark/model.json 1 fp32 100 --scope interactive-gpu-replay
cargo run --release -p audio2face3d --features cli,native -- benchmark ./models/mark/model.json 1 fp32 100 --output temp/benchmarks/mark.json
```

JSON output records the model and engine SHA-256, execution environment,
P50/P95/P99 latency, throughput, and peak GPU memory. Compare performance
against the tracked hardware baseline independently from reference-value
parity:

```sh
cargo run -p audio2face3d --features cli -- release benchmark-compare reference/benchmark-baseline.json temp/benchmarks/mark.json
```

For geometry models, the isolated post-process phase currently measures the device-result consumption boundary; full animator and blendshape timing must be reported separately from raw TensorRT inference. Compare C++ and Rust only with identical model/engine, batch, precision, GPU, driver, CUDA, and TensorRT versions.
