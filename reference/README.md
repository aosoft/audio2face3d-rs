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

The three cases are:

- `regression`: Mark regression sample
- `diffusion`: multi-identity diffusion sample
- `a2e`: Audio2Emotion classifier sample

ONNX, configuration, input, golden, and intermediate NPZ/BIN artifacts from
both sample and generated test-data directories are hashed. TensorRT engine
hashes are recorded but are environment-specific and must be validated
numerically after regeneration.
