# Packages and features

[Project overview](../README.md) · [Documentation index](../README.md#documentation)

Choose Cargo features for the capabilities your application needs.

## Package boundaries

`audio2face3d` provides inference/client APIs and an optional CLI.
`audio2face3d-server` and `audio2face3d-gui` each combine reusable libraries with
an executable; there are no separate core packages. `audio2face3d-headgen` uses
only the GUI crate's model/GLB features with default features disabled.

## Application features

| Package | Feature | Purpose |
|---|---|---|
| `audio2face3d` | default (`[]`) | Shared types and client control |
| `audio2face3d` | `mock` / `native` / `client-grpc` | Diagnostic / native / remote inference |
| `audio2face3d` | `cli` | Base package executable |
| `audio2face3d-server` | default (`native,cli`) | Full native server executable |
| `audio2face3d-server` | `mock` | Diagnostic backend; disable defaults and add `cli` to run it |
| `audio2face3d-gui` | default (`standalone-app,grpc`) | Standard GUI with remote inference |
| `audio2face3d-gui` | `local` | Native inference |
| `audio2face3d-gui` | `gltf-read` / `gltf-write` | GLB I/O; disable defaults for model-only use |
| `audio2face3d-gui` | `session`, `render-wgpu`, `ui-egui` | Reusable media, rendering and UI components |

Library consumers disable defaults and select only required features. Server CLI
dependencies are gated by `cli`; GUI startup/window/audio device dependencies by
`standalone-app`. Features are additive: model-only builds must not also enable the
default GUI feature set through another dependency in the same build graph.

## Lower-level features

| Feature | Contents | Required tier |
|---|---|---|
| default | Shared Rust data and client control; no external dependencies | `portable` |
| `animation` | geometry, BlendShape, Teeth, and interactive APIs | `portable` |
| `emotion` | Audio2Emotion post-processing; also enables `animation` | `portable` |
| `cuda` | CUDA buffers, streams, solvers, and lifetime tests | `cuda-lifetime` |
| `tensorrt` | real TensorRT model execution | `tensorrt-model` |

The tiers are described in [development and validation](development.md#release-support-contract).

## SDK module mapping

The Rust modules correspond to the original SDK as follows:

| NVIDIA SDK component | Rust module |
|---|---|
| `audio2x-sdk` | `audio2face3d` facade and pipeline |
| `audio2x-common` schemas and accumulators | `audio2face3d::common` |
| `audio2x-common` CUDA support | `audio2face3d::cuda` |
| `audio2x-common` TensorRT support | `audio2face3d::tensorrt` |
| `audio2face-sdk` | `audio2face3d::audio2face` |
| `audio2emotion-sdk` | `audio2face3d::audio2emotion` |

## Previous workspace layout

The former types/client/inference/protocol packages are modules under `audio2face3d::{types,client,inference,protocol}`. The former CLI package is now the base package binary. The old `direct`/`runtime`/remote `server` feature selections become `mock` or `native`, `native`, and `client-grpc`, respectively. Wire types are separate from common Rust request/result types. `grpc-server` is a technical wire integration feature used by the server dependency; it provides shared inference plumbing but does not compile a backend.

`runtime-cli` exposes the shared native-path argument parser used by the server executable without enabling model download commands. Applications normally select `cli` instead. Native loading itself is enabled by `cuda`/`tensorrt`; no separate loader feature or shim DLL is required.

## Logging dependencies

Library logging uses a standard-library-only Logger. `mock` and `native` do not enable tracing or tracing-subscriber for logging, and direct inference does not require Tokio. The base and server packages gate direct CLI logging dependencies behind `cli`. gRPC dependencies may themselves use tracing. Cargo features are additive: `--lib --features cli` still enables CLI dependencies even when no executable is built. See [Logging](logging.md) for output configuration and migration from the removed `tracing` feature.
