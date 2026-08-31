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
cargo test -p audio2face3d-cli --test reference_artifacts `
  reference_artifact_manifest_matches_sdk_checkout -- --ignored --exact
```

The test checks the recorded SDK revision and every fixture's size and SHA-256.
The checkout path is supplied only through the environment and is not stored in
the manifest.

Model download and TensorRT engine generation are owned by the main Rust model
tool. For example, generate the Mark FP16 engine with the original `_fp16`
suffix convention as follows:

```powershell
cargo run -p audio2face3d-cli -- model engine mark --precision fp16
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

## Original SDK compatibility harness

`artifact-schema.json` is shared by the original C++ SDK runner and the Rust
runner. Each capture directory contains `artifact.json` plus a contiguous
`values.f32le` blob. Every record pins its layer, component, callback metadata,
shape, byte range, and SHA-256. `tolerances.json` separates FP32 and FP16 limits
and can override them per component.

All machine-local input and generated output belongs under
`reference/compatible_test/`, which is ignored by Git. Place the evaluation
audio at the fixed path below before running the harness:

```powershell
New-Item -ItemType Directory -Force reference\compatible_test | Out-Null
Copy-Item '<evaluation-audio.wav>' reference\compatible_test\input.wav
```

`input.wav` must be mono PCM16 at 16 kHz. The harness decodes it once into
`compatible_test/fixture/samples.f32le`; both runners consume that same sample
sequence. Set `AUDIO2FACE3D_REFERENCE_WAV_SHA256` when the source WAV digest
must be pinned, and set `AUDIO2FACE3D_REFERENCE_WAV_LICENSE` to record a license
label other than the default `user-provided-not-for-redistribution`.

Configure every external installation through environment variables. No local
checkout or installation path is accepted as a script argument or stored in a
tracked file.

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
`standard`, `interactive-random`, `interactive-all`, `blendshape-cpu`, or
`blendshape-gpu`; Audio2Emotion currently uses `standard`.

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
reference/compatible_test/       # ignored as a whole
├── input.wav                     # user-supplied fixed input name
├── fixture/
│   ├── fixture.json
│   └── samples.f32le
├── tools/                        # C++ runner binary and object
└── results/
    └── <pipeline>-<execution>-<precision>-tracks<N>-seed<N>/
        ├── cpp/{artifact.json,values.f32le}
        ├── rust/{artifact.json,values.f32le}
        └── comparison.json
```

None of these local inputs, binaries, captures, or comparison results are
committed. Only the runner source, schema, case catalog, and tolerance profile
remain tracked.

The case catalog is `cases.json`. Standard Regression, Diffusion, and
Audio2Emotion, Regression/Diffusion interactive random/all-frame execution, and
CPU/GPU BlendShape capture use the same artifact contract. The older ignored
TensorRT fixture test remains the low-level binding/inference check.
