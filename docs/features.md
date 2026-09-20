# Packages and features

[Project overview](../README.md) · [Documentation index](../README.md#documentation)

Choose Cargo features for the capabilities your application needs.

## Package boundaries

The workspace contains two packages: `audio2face3d` and `audio2face3d-server`. Both expose a library and an executable enabled by `cli`. The base package has no default features; `mock` enables local diagnostic inference, `native` enables CUDA/TensorRT inference, and `client-grpc` enables the remote client. Local inference does not require Tokio. The server depends on the base package and defaults to `native`; its executable defaults to Regression, which requires an explicit model. Use `--no-default-features --features cli,mock` for a diagnostic server.

## Application features

| Package | Feature | Purpose |
|---|---|---|
| `audio2face3d` | default (`[]`) | Shared data and client control without external dependencies |
| `audio2face3d` | `mock` | Local diagnostic inference |
| `audio2face3d` | `native` | CUDA/TensorRT inference, including animation and emotion |
| `audio2face3d` | `client-grpc` | Remote client using the caller’s Tokio runtime |
| `audio2face3d` | `grpc-server` | Protocol integration used by the server package |
| Both | `cli` | Enable the package executable |
| `audio2face3d-server` | default (`native`) | Native Regression backend |
| `audio2face3d-server` | `mock` | Diagnostic backend; disable defaults for a portable build |

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
