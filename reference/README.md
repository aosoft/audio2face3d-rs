# Reference baseline

The reference artifacts stay in the local NVIDIA SDK checkout because model and
SDK licenses differ from this project's source license. This directory records
their revisions and hashes without redistributing them.

Generate `reference/artifacts.json` from the repository root:

```powershell
cargo run --manifest-path reference/Cargo.toml -- `
  --sdk-root <Audio2Face-3D-SDK checkout>
```

The checkout location is required through `--sdk-root` or the
`AUDIO2FACE_SDK_ROOT` environment variable. No local path is written to the
manifest. Generation is atomic: the final JSON replaces the previous manifest
only after every file has been read and hashed.

Generate the optional FP16 engine and descriptors without the original Python
generator by setting `TRTEXEC` and passing the generated model directory:

```powershell
$env:TRTEXEC = '<TensorRT-root>\bin\trtexec.exe'
cargo run --manifest-path reference/Cargo.toml -- `
  --sdk-root <Audio2Face-3D-SDK-checkout> `
  --generate-fp16 <generated-Mark-model-directory>
```

This reads the SDK's `trt_info.json`, adds `--fp16`, expands its batch-profile
defaults, builds `network_fp16.trt`, and writes `trt_info_fp16.json` and
`model_fp16.json`. Existing FP16 artifacts are never overwritten.

The three cases are:

- `regression`: Mark regression sample
- `diffusion`: multi-identity diffusion sample
- `a2e`: Audio2Emotion classifier sample

ONNX, configuration, input, golden, and intermediate NPZ/BIN artifacts from
both sample and generated test-data directories are hashed. TensorRT engine
hashes are recorded but are environment-specific and must be validated
numerically after regeneration.
