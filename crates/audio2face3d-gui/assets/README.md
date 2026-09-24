# Standard debug head

`default-head.glb` is original procedural geometry, licensed under the repository's
MIT license. It contains no NVIDIA head geometry or third-party textures.

Regenerate explicitly from the repository root:

```sh
cargo run -p audio2face3d-headgen --features cli -- --config crates/audio2face3d-headgen/presets/default-head.json --output crates/audio2face3d-gui/assets/default-head.glb
```

Generator 0.1.0, `debug_face_52_v1`: 610 head-body vertices, 2,134 total vertices,
4,016 triangles, 18 independent mesh parts. All 52 channels have nonzero deltas;
parts share names where needed. Normal deltas are recomputed from posed geometry.
This is a diagnostic mannequin, not a production facial rig. Side views expose
the deliberately simplified eyelid/lip construction. Extreme simultaneous
weights can create unrealistic expressions. See `docs/gui-contract.md` for
MouthClose semantics and character-relative left/right.

Visual validation tool (requires a GPU):

```sh
cargo run -p audio2face3d-gui --features render-wgpu --example head_atlas -- crates/audio2face3d-gui/assets/default-head.glb temp/head-atlas
```

Each contact-sheet row follows its adjacent text file. Columns: front at 0,
front at 0.5, front at 1, three-quarter at 1. `capture_head` accepts named values
such as `JawOpen=0.7 MouthSmileLeft=0.5 yaw=0.6` for combination checks.
