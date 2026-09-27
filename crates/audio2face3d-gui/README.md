# audio2face3d-gui

Head models, playback, inference and UI library with a desktop executable.
By default, the application supports gRPC inference only (`standalone-app,grpc,obj2morph,emotion`). From the repository root:

```sh
cargo run -p audio2face3d-gui
```

Local inference requires `--features local` (gRPC remains available):

```sh
cargo run -p audio2face3d-gui --features local
```

Configure `gui.toml` and, for local
inference, `platform.toml`. Library users disable default features; model-only
users enable just `gltf-read` or `gltf-write` as needed.
See the [GUI guide](../../docs/gui.md).

The default `obj2morph` feature adds head conversion without opening a window.
Create `models/heads` first, then run:

```sh
cargo run -p audio2face3d-gui -- obj2morph --config crates/audio2face3d-gui/presets/ict-facekit.toml --input-root /path/to/ICT-FaceKit/FaceXModel --output models/heads/ict-facekit.glb
```

After conversion, uncomment `head = "models/heads/ict-facekit.glb"` in your
`gui.toml` copied from `gui.example.toml`. Paths are relative to the TOML file.

For a GUI without conversion, use `--no-default-features --features standalone-app,grpc`.
See the [head conversion guide](../../docs/headgen.md).

Emotion controls are enabled by default. Local emotion inference is optional: select an Audio2Emotion model to enable it. See the [GUI guide](../../docs/gui.md#emotion) for configuration and feature selection.
