# Reference baseline

The proprietary model artifacts stay in the local NVIDIA SDK checkout because
their licenses differ from this project's source license. This directory keeps
only schemas, provenance, tolerance profiles, runner source, and orchestration;
it does not redistribute a model, TensorRT engine, or user-provided recording.

Both migrated reference validations are ignored tests. A normal workspace test
compiles them but does not access the SDK checkout, TensorRT engine, or GPU.

Validate `reference/artifacts.json` against a local SDK checkout explicitly:

```powershell
$env:AUDIO2FACE_SDK_ROOT = '<Audio2Face-3D-SDK checkout>'
cargo test -p audio2face3d --features cli --test reference_artifacts `
  reference_artifact_manifest_matches_sdk_checkout -- --ignored --exact
```

The test checks the recorded SDK revision and every fixture's size and SHA-256.
The checkout path is supplied only through the environment and is not stored in
the manifest.

Model download and TensorRT engine generation are owned by the main Rust model
tool. For example, generate the Mark FP16 engine with the original `_fp16`
suffix convention as follows:

```powershell
cargo run -p audio2face3d --features cli -- model engine mark --precision fp16
```

The three cases are:

- `regression`: Mark regression sample
- `diffusion`: multi-identity diffusion sample
- `a2e`: Audio2Emotion classifier sample

ONNX, configuration, input, golden, and intermediate NPZ/BIN artifacts from
both sample and generated test-data directories are hashed. TensorRT engine
hashes are recorded but are environment-specific and must be validated
numerically after regeneration.

Validate a Rust-generated engine against the matching C++ tensor fixture by
pointing the TensorRT test at both artifacts:

```powershell
$env:AUDIO2FACE3D_REFERENCE_ENGINE = '<Rust-generated-engine>'
$env:AUDIO2FACE3D_REFERENCE_TENSORS = '<Audio2Face-3D-SDK>\_data\generated\audio2x-common\tests\data\test_data_inference.bin'
cargo test -p audio2face3d --features tensorrt `
  --test reference_inference cpp_fixture_matches_rust_tensor_rt_engine `
  -- --ignored --exact
```

The engine must be generated from the fixture's corresponding
`test_data_inference_network.onnx`. The test derives its single dynamic input
dimension from each fixture tensor, runs inference, and compares every output
element with an absolute tolerance of `1e-3`.

## Release benchmark baseline

`benchmark-baseline.json` fixes the validated hardware/software environment,
required Regression, Diffusion, Audio2Emotion, CPU/GPU BlendShape, and
interactive replay workloads, plus the initial measured cases. Capture commands
use only environment variables for local model descriptors. Generated benchmark
JSON belongs below `temp/benchmarks/` and is ignored.

The benchmark comparator checks P50/P95/P99 latency, throughput, and peak GPU
memory without mixing performance failure with the numeric artifact comparator:

```powershell
cargo run -p audio2face3d --features cli -- release benchmark-compare `
  reference/benchmark-baseline.json `
  temp/benchmarks/regression.json
```

The default limits are 15 percent for latency and throughput and 10 percent for
memory. Candidate and baseline engine hashes must match. Recapture a baseline
when the GPU, driver, CUDA, TensorRT, precision, track count, or model engine
changes; do not compare unlike environments.

## Original SDK compatibility harness

`artifact-schema.json` is shared by the original C++ SDK runner and the Rust
runner. Each capture directory contains `artifact.json` plus a contiguous
`values.f32le` blob. Every record pins its layer, component, callback metadata,
shape, byte range, and SHA-256. `tolerances.json` separates FP32 and FP16 limits
and can override them per component.

Generated captures go under `temp/reference-comparison/` (override with `-OutputRoot`); the runner is built under `temp/reference-tools/`. Both are ignored by Git. The existing input location is retained. Place the evaluation
audio at the fixed path below before running the harness:

```powershell
New-Item -ItemType Directory -Force reference\compatible_test | Out-Null
Copy-Item '<evaluation-audio.wav>' reference\compatible_test\input.wav
```

`input.wav` must be mono PCM16 at 16 kHz. The harness decodes it once into
the case's `fixture/samples.f32le`; both runners consume that same sample
sequence. Set `AUDIO2FACE3D_REFERENCE_WAV_SHA256` when the source WAV digest
must be pinned, and set `AUDIO2FACE3D_REFERENCE_WAV_LICENSE` to record a license
label other than the default `user-provided-not-for-redistribution`.

Use `ci/run-tier.ps1 reference-parity -PlatformConfig platform.toml` to select SDKs without parent environment changes. Direct invocation of this legacy harness still reads SDK roots from its environment. Original SDK and input-license parameters remain separate. See [platform configuration](../docs/platform.md).

```powershell
$env:AUDIO2FACE_SDK_ROOT = '<Audio2Face-3D-SDK checkout>'
$env:TENSORRT_ROOT_DIR = '<TensorRT installation>'
$env:CUDA_PATH = '<CUDA Toolkit installation>'
```

The default model is resolved below `AUDIO2FACE_SDK_ROOT`. An alternative local
descriptor can be selected through `AUDIO2FACE3D_REFERENCE_MODEL`.

Build the original SDK runner as an isolated MSVC target. The script selects an
installed MSVC and Windows SDK, then links with `/INCREMENTAL:NO`, `/LTCG:OFF`,
and `/DEBUG:NONE`. This avoids the aggregate SDK link state that produced
`LNK1000`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File reference\build-cpp-runner.ps1
```

Run a complete capture and comparison with one command. `-Execution` accepts
`standard`, `interactive-random`, `interactive-all`,
`interactive-blendshape-random`, `interactive-blendshape-all`,
`blendshape-cpu`, `blendshape-gpu`, or `teeth-standalone`; Audio2Emotion currently uses
`standard`. The standalone teeth case compares the original
`IMultiTrackAnimatorTeeth` with the Rust multi-track CUDA animator using the
same model neutral jaw, deterministic deltas, per-track parameters, and padded
input/output rows with non-zero offsets.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File reference\run-sdk-compatibility.ps1 `
  -Pipeline regression `
  -Execution standard
```

The command intentionally exits unsuccessfully when parity is outside the
profile and still writes `comparison.json`, including the first mismatching
layer/component/index and the observed maximum error up to that point. The
local directory layout is:

```text
reference/compatible_test/input.wav     # existing user-supplied input
temp/reference-tools/                  # C++ runner and object
temp/reference-comparison/
   <pipeline>-<execution>-<precision>-tracks<N>-seed<N>/
     fixture/{fixture.json,samples.f32le}
     cpp/{artifact.json,values.f32le}
     rust/{artifact.json,values.f32le}
     comparison.json
```

None of these local inputs, binaries, captures, or comparison results are
committed. Only the runner source, schema, case catalog, and tolerance profile
remain tracked.

The case catalog is `cases.json`. Standard Regression, Diffusion, and
Audio2Emotion, Regression/Diffusion interactive random/all-frame execution,
interactive GPU BlendShape random/all-frame execution, and
CPU/GPU BlendShape and standalone Teeth capture use the same artifact contract.
The older ignored TensorRT fixture test remains the low-level binding/inference
check.

## API regression assessment

The legacy 19-case API assessment explicitly retains its existing `reference/compatible_test` output layout. New standalone compatibility runs use `temp` as described above.

Before changing execution behavior, preserve the local `results` directory
under a new ignored `reference/compatible_test/<baseline>/results` directory.
Record the source revision and capture conditions separately: a copied local
capture is not proof of a pre-migration revision.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File reference/run-api-regression.ps1 `
  -BaselineDirectory reference/compatible_test/baseline `
  -ReportDirectory reference/compatible_test/api-report
powershell -NoProfile -ExecutionPolicy Bypass -File reference/compare-api-baseline.ps1 `
  -BaselineDirectory reference/compatible_test/baseline `
  -ReportDirectory reference/compatible_test/api-report
```

Use a new report directory for each run. The first command executes 19 FP32,
seed-zero cases, preserves their captures alongside the report, and records
structural failures separately from numerical differences. It fails when any
case is unclassified or did not run. The optional second command assesses
numerical stability against matching baseline inputs using the unchanged
tolerances. Its assessment does not turn an SDK comparison failure into a pass.
Maximum errors cover only the records visited before comparison stops;
zero values compared is not evidence of numerical agreement.

The strict `reference-parity` tier still gates only Emotion standard (one
track) and standalone Teeth (two tracks). Emotion interactive C++ captures
are not implemented by this harness; native facade tests are separate evidence.
