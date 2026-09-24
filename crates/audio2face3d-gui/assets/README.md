# Standard debug head

`default-head.glb` is original procedural geometry, licensed under the repository's
MIT license. It contains no NVIDIA head geometry or third-party textures.

The asset was produced by the former procedural generator (0.1.0), then
re-exported with `audio2face_rs_tester_v1`: 610 head-body vertices, 2,134 total vertices,
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

The procedural generator and its JSON preset have been removed. Its source is
available in Git history (before the OBJ conversion migration). There is no
current regeneration command for this original asset. The new converter accepts
user-provided neutral/expression OBJ files; see `docs/headgen.md`.
ICT source data, derived GLBs and diagnostic images are not bundled here.
