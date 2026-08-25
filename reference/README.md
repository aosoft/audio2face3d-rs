# Reference baseline

The reference artifacts stay in the local NVIDIA SDK checkout because model and
SDK licenses differ from this project's source license. This directory records
their revisions, hashes, and benchmark results without redistributing them.
Executable validation lives in the crate that owns the behavior being tested;
`reference/` is data and documentation only.

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
