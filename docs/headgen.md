# OBJ head conversion

`audio2face3d-headgen` is a CPU-only library and executable. No CUDA, TensorRT,
Blender, Python, or Cargo feature selection is required. The tool does not fetch
models. Users provide input data and decide how their converted models are used.

```powershell
cargo run -p audio2face3d-headgen -- inspect --config crates/audio2face3d-headgen/presets/ict-facekit.toml --input-root C:/Data/ICT-FaceKit/FaceXModel --report temp/ict-inspect.json
cargo run -p audio2face3d-headgen -- convert --config crates/audio2face3d-headgen/presets/ict-facekit.toml --input-root C:/Data/ICT-FaceKit/FaceXModel --output temp/ict-facekit.glb
```

Output directories must already exist. Convert also writes `ict-facekit.report.json`.
`--report` overrides that path; `--force` explicitly permits replacing regular output
files. Inputs, config and output paths cannot collide. Both outputs are staged and
synced before either is replaced. Two-file atomicity is not guaranteed: a report
commit failure identifies the already-committed GLB and its hash. Inspect performs
all shape validation and exact GLB size measurement but writes no GLB.

## Input contract

Use UTF-8/ASCII OBJ files, with `v x y z`, optional `vt`, `vn`, `o`, `g`, `s`,
`usemtl`, `mtllib`, and triangle/quad `f` records. All four face-index syntaxes and
negative indices are accepted. Unsupported instructions, nonfinite coordinates,
homogeneous/color vertices, invalid references, self-intersections and degenerate
retained triangles are errors. MTL and textures are never opened.

Neutral and every expression must have identical vertex counts and the exact same
ordered face vertex indices. This is checked before exclusions or triangulation.
Coincident vertices remain distinct. UV/normal indices may differ. Matching topology
cannot prove semantic correspondence; inspect the actual poses visually.

All configured input paths use forward slashes relative to `--input-root`. Absolute
paths, parent traversal and links outside that root are rejected. Only explicitly
listed files are read, including on ICT datasets containing identity models.

## Configuration

See `crates/audio2face3d-headgen/presets/ict-facekit.toml` for all required sections.
Unknown fields, duplicate keys and unknown enum values are rejected. The keys in
`targets` and `unsupported_channels` must partition the canonical 52 channel names.
An output channel lists one or more unique expression files; their neutral-relative
deltas are **summed**, not averaged. No hidden gains, retargeting or corrective shapes
are applied. Every output channel must retain nonzero displacement somewhere.

`transform` declares distinct signed up/forward axes, fitted height and target center.
A right-handed axis map and one scale/translation are computed from retained Neutral
bounds, then shared by every expression. Output uses +Y up, +Z forward.
`geometry.split_by` is `material` or `object_group_material`. Exclusions name Neutral
materials. Opaque RGBA colors come from `materials`, not MTL. Shared-vertex normals
are calculated across materials before parts are split; hard edges require separate
source vertices. Composite normals are recalculated from the summed pose.
`expected` provides optional strict dataset counts. `reference` is descriptive
provenance, not a license or permission determination.

Input limits: 128 MiB/file, 2 GiB total, 1 MiB/line, 100,000 source vertices,
600,000 triangulated indices and 104 expression files. Output limits: 64 MiB GLB
and decoded geometry, 100,000 split vertices, 600,000 indices, 64 parts and 52 targets
per part. GPU vec4 storage estimates are reported separately; the renderer checks
device limits. The converter never silently reduces geometry or raises limits.

## Library and diagnostics

`Config::parse`, `inspect(&Config, &Path)` and `convert(&Config, &Path)` return typed
errors. Convert returns `Conversion { model, report }`; the library writes no files.
The executable owns output persistence and process exit codes: 0 success, 2 config/
CLI, 3 input/topology/shape, 4 I/O/output/limits. Expected unsupported channels are
not errors. Reports contain source/config hashes, relative paths, transform,
material exclusions, channel displacement metrics, reversed-normal candidates,
CPU/GPU size estimates and output hashes. Reversal warnings may be legitimate large
rotations and require visual review. Same inputs/config/tool/platform produce the
same GLB bytes; no timestamps, random identifiers or absolute paths are embedded.

## ICT reference preset and visual review

The preset maps 53 expression files to 51 channels; BrowInnerUp and CheekPuff sum
left/right files. TongueOut is explicitly unsupported. Identity, pupil dilation,
additional cheek shapes and textures are not imported. EyeBlend, EyeOcclusion,
LacrimalFluid and EyeLashes are excluded because their intended rendering requires
special shading. The preset's `reference.status` records its validation status.

Use `head_atlas` for supported channels at 0 / 0.5 / 1 and a three-quarter view.
Use `capture_head` for Neutral and combinations, for example:

```powershell
cargo run -p audio2face3d-gui --features render-wgpu --example head_atlas -- temp/ict-facekit.glb temp/ict-atlas
cargo run -p audio2face3d-gui --features render-wgpu --example capture_head -- temp/ict-facekit.glb temp/ict-jaw.png JawOpen=0.7 MouthClose=0.3 yaw=0.6
```

Check eyelids, eyes, lips, oral cavity, teeth/tongue motion and left/right meanings.
Conversion success is not a visual quality certification. Keep ICT source data,
derived models, reports and screenshots outside versioned assets (for example in
ignored `temp`). Normal CI uses only independent tiny fixtures; packaging explicitly
copies the original fallback head, not conversion output.

## P08 validation and explicit pose tolerance

The ICT preset was exercised on revision `da5f95a607f5e6b37755b38d3385d7f2853732e5`.
All 53 expression inputs pass ordered topology matching and map to 51 channels.
MouthStretchRight and MouthUpperUpRight contain two oral source faces (15776 and
15781) whose adjacent vertices collapse in the posed geometry.

`geometry.degenerate_pose_triangles` defaults to `"error"`. The ICT preset explicitly
uses `"skip_normal_contribution"`, approved for this input: only zero-area **posed**
triangles are omitted when accumulating normals. Vertex positions, morph deltas
and triangle indices remain unchanged. Surrounding nondegenerate faces must still
provide a usable normal at every retained vertex. Neutral degeneracy, overflow and
undefined surrounding normals always remain errors. Reports record skipped triangle
indices and original source face numbers per output channel.

The reference conversion produces 24,953 split vertices, 48,836 triangles and eight
parts, within the existing GLB/CPU/GPU limits. CPU/GPU pose checks and actual Local
WAV playback were exercised. These are functional checks, not a quality guarantee.
The preset remains a candidate: opaque outer sclera geometry obscures the iris,
limiting gaze inspection; strong additive mouth/eyelid combinations need further
artistic review. A separate custom-model repository could adapt eye geometry/material
boundaries for opaque rendering and author combination corrections. This converter
does not insert hidden corrective weights or silently modify source geometry.
