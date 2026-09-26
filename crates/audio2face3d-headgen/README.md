# audio2face3d-headgen

CPU-only library and CLI that converts matching neutral/expression OBJ files to
head GLB data. No CUDA/TensorRT or feature selection is required.

Run from the repository root (the output directory must already exist):

```sh
cargo run -p audio2face3d-headgen -- convert --config converter.toml --input-root path/to/obj-files --output head.glb
```

See the [head conversion guide](../../docs/headgen.md) for configuration,
library usage and the ICT-FaceKit reference preset.
