# audio2face3d-gui

Head models, playback, inference and UI library with a desktop executable.
Defaults enable `standalone-app,grpc`. From the repository root:

```sh
cargo run -p audio2face3d-gui
```

Add `--features local` for native inference. Configure `gui.toml` and, for local
inference, `platform.toml`. Library users disable default features; model-only
users enable just `gltf-read` or `gltf-write` as needed.
See the [GUI guide](../../docs/gui.md).
