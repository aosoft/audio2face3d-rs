# GUI implementation contract

This document specifies the GUI model format, rig semantics, playback behavior
and integration boundaries.

## Packages and dependencies

Rust 2024, workspace MSRV 1.91. `audio2face3d-gui-core` owns model data, validation
and GLB I/O; `audio2face3d-headgen` is a CPU-only OBJ conversion library/CLI;
`audio2face3d-gui` contains the playback/session library, optional renderer/UI and
a thin desktop executable. It uses `audio2face3d`, never the server package.
No build script regenerates assets. No library installs a process-global logger.

Use gltf/gltf-json 1.4.1 (`utils`, `names`, `extras`, no image/import feature),
egui/egui-wgpu/eframe 0.31.1 with wgpu 24, CPAL 0.15.3 and hound 3.5.1.
These compatible baseline versions were inspected with `cargo info`; resolved
transitive versions are recorded in Cargo.lock and checked against the MSRV.
Use eframe only in the desktop host; the renderer receives a host device/queue.
GPU morph deltas use storage buffers, supporting all 52 targets together.

Features: core `gltf-read` / `gltf-write`; headgen requires no feature flags; GUI `local`, `grpc`,
`render-wgpu`, `ui-egui`, `standalone-app`. Default features are empty. `standalone-app` selects
rendering/UI/audio/file dialogs, never native inference. `local` selects native;
`grpc` selects client-grpc and a host-owned Tokio runtime. Mock is development-only.

## GLB profile version 1

GLB 2.0 only, at most 64 MiB. Exactly one embedded BIN buffer, no external URI.
One default scene containing independent identity-transform root nodes, each
referencing one distinct mesh. One indexed TRIANGLES primitive per mesh.
No children, transforms, skins, animations, cameras, images, textures or required
extensions. Reject unsupported data instead of silently dropping it.
Positions/normals and morph position/normal deltas are non-normalized FLOAT VEC3.
Support valid accessor offsets and byte strides; sparse accessors are rejected.
Indices are unsigned 16 or 32 bit SCALAR; writer uses 16 bit when all indices fit.
Vertex arrays must agree in length; indices must be in range. All numbers finite.
Base normals must be nonzero. No more than 52 morph targets per mesh.
At most 64 meshes, 100,000 total vertices and 600,000 total indices.
Reader validates buffer/view/accessor byte bounds before constructing iterators.

Material: opaque single baseColorFactor, metallic=0, roughness=1, doubleSided=true,
no texture. Coordinate system: right handed, Y up, face toward +Z, meters.
Character's left is +X (viewer right in the default front view).
Each mesh has `extras.targetNames`, exactly matching morph target order, unique
and nonempty within the mesh. Names shared across meshes drive all those parts.
Initial weights must be zero. Target names must be canonical ACE names.

Top-level `extras.audio2face3d_preview` contains `schema_version: 1`,
`rig_profile: "audio2face_rs_tester_v1"`, and a nonempty `generator_version`.
Only this profile is supported. Legacy profile names are rejected.
The union of target names must be a nonempty subset of the 52 names below.
Every declared channel must have a position delta exceeding 1e-8 somewhere.
No individual mesh needs every target. Unsupported channels are the difference
between the canonical set and this union; GUI inference data is not filtered.

## OBJ conversion configuration version 1

See [headgen.md](headgen.md) and the checked-in ICT reference preset. Strict TOML
maps canonical channels to absolute-pose OBJ files with identical ordered topology.
Neutral-space triangulation, coordinate transformation, material splitting and
summed deltas are deterministic. Composite normals are recalculated before splitting.
The former procedural generator and JSON configuration are no longer supported.
Only the original fallback asset is bundled. Third-party models are never included
by the converter, CI, or package script.

## Rig semantics

Weights are additive displacement coefficients, not normalized across channels.
The raw received values are retained. UI may show a clipped bar but also the
original number. Interpolation is linear in media time. No implicit corrective
mixing, clamping or ARKit conversion is applied.

The original bundled fallback asset uses NVIDIA-style MouthClose: its standalone pose lowers the
jaw/chin while keeping the lips together. It is NOT an inverse JawOpen delta.
JawOpen opens lips and lowers chin. Both at 1 add chin displacement; this is an
extreme diagnostic pose, not automatic cancellation. Teeth follow jaw movement.
The profile validates names and data, not anatomical or model-training parity.
Converted models retain their source poses, including different MouthClose semantics.
The converter does not force a source pose to match the fallback's convention.
The following table describes the fallback's intended motions; it is not a guarantee
that a third-party input implements every motion identically.

| Channel(s), exact server spelling | Target motion at weight 1 |
| --- | --- |
| EyeBlinkLeft, EyeBlinkRight | Upper/lower eyelids meet over the corresponding eye |
| EyeLookDownLeft, EyeLookDownRight | Iris moves downward |
| EyeLookInLeft, EyeLookInRight | Iris moves toward nose |
| EyeLookOutLeft, EyeLookOutRight | Iris moves toward temple |
| EyeLookUpLeft, EyeLookUpRight | Iris moves upward |
| EyeSquintLeft, EyeSquintRight | Lower lid rises, aperture narrows |
| EyeWideLeft, EyeWideRight | Eyelids separate |
| JawForward | Jaw/chin moves toward +Z |
| JawLeft, JawRight | Jaw/chin moves to character left/right |
| JawOpen | Chin/lower lip descends, mouth aperture opens |
| MouthClose | Jaw lowers with lips together; NVIDIA convention above |
| MouthFunnel | Lips form an open rounded forward funnel |
| MouthPucker | Lip aperture narrows and lips move forward |
| MouthLeft, MouthRight | Lips translate to character left/right |
| MouthSmileLeft, MouthSmileRight | Corresponding mouth corner rises and widens |
| MouthFrownLeft, MouthFrownRight | Corresponding corner descends |
| MouthDimpleLeft, MouthDimpleRight | Corner pulls backward and slightly outward |
| MouthStretchLeft, MouthStretchRight | Corner moves laterally outward |
| MouthRollLower, MouthRollUpper | Specified lip rolls inward/backward |
| MouthShrugLower, MouthShrugUpper | Specified lip moves upward |
| MouthPressLeft, MouthPressRight | Corresponding upper/lower lips move together |
| MouthLowerDownLeft, MouthLowerDownRight | Corresponding lower lip descends |
| MouthUpperUpLeft, MouthUpperUpRight | Corresponding upper lip rises |
| BrowDownLeft, BrowDownRight | Corresponding brow descends |
| BrowInnerUp | Medial ends of both brows rise |
| BrowOuterUpLeft, BrowOuterUpRight | Lateral brow rises |
| CheekPuff | Both cheeks expand outward/forward |
| CheekSquintLeft, CheekSquintRight | Corresponding cheek rises |
| NoseSneerLeft, NoseSneerRight | Corresponding nose wing rises |
| TongueOut | Tongue extends through mouth toward +Z |

## Playback and resource limits

Initial WAV support: PCM16/24/32 or IEEE float32, mono/stereo, 8..192 kHz.
Reject nonfinite float samples and malformed files. Stereo is averaged; use a
documented low-pass resampler to produce signed PCM16 mono 16 kHz input.
Playback uses the returned audio/media sample positions, not input-send time.
Device callbacks provide scheduled audible timestamps; GUI uses the audible
position, including device latency. Host audio is an injectable interface.
Pause/seek invalidate queued audio by recreating the standard host stream;
media anchors are cleared on repositioning. Loop boundaries split clock anchors.

Maximum clip length 10 minutes; result memory budget 256 MiB including audio and
curves. Reject further input with a visible error on exhaustion; keep partial
results labelled failed, never auto-play them as successful clips. Bound worker
queues independently. Logs: 2,048 queued, 10,000 displayed; nonblocking enqueue,
drop newest on saturation and expose a dropped count. Host owns logger lifetime.

Streaming playback requires an initial/rebuffer threshold of 100 ms of contiguous audio AND curves
(including interpolation lookahead). Freeze media clock on underrun, output
silence until ready, then resume; never shift original result timestamps.
Normally completed short clips may play below threshold. Failed streams require
explicit user action to inspect partial data. Session and playback completion
are separate. Future seeking is rejected; received past remains seekable.

Engine-specific GPU sharing and FFI remain future work. No promise that a wgpu
texture can directly cross graphics devices or Unreal's RHI boundary.

## References

- [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
- [gltf 1.4.1](https://docs.rs/gltf/1.4.1/gltf/)
- [eframe 0.31.1](https://docs.rs/eframe/0.31.1/eframe/)
- [CPAL 0.15.3](https://docs.rs/cpal/0.15.3/cpal/)
- [NVIDIA MouthClose convention](https://github.com/NVIDIA/Audio2Face-3D-Training-Framework/blob/main/docs/preparing_animation_data.md#optional-blend-shapes-data)
