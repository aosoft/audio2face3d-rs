# Audio2Face-3D for Rust

This project is an **unofficial Rust port of NVIDIA’s Audio2Face-3D SDK**,
maintained independently and not an official NVIDIA SDK release.

It also provides a unified client library for direct and remote inference and an
embeddable gRPC server.

## Packages

| Package | Purpose |
|---|---|
| `audio2face3d` | Shared types, direct/remote client, inference engines, and optional CLI |
| `audio2face3d-server` | Embeddable gRPC server and optional server executable |

The base package has no default features. Select `native` for local CUDA/TensorRT
inference, `client-grpc` for remote inference, or `mock` for diagnostics.
Both executables require `cli`; the server enables `native` by default.
See [packages and features](docs/features.md) for the complete breakdown.

## Getting started

Rust 1.91 or newer is required. Native inference is validated on Windows x86-64/MSVC
with CUDA 12.9 and TensorRT 10.16.1; Linux native support is experimental and build-only.
Models and native SDKs are obtained separately.

1. Follow [runtime setup](docs/getting-started.md#runtime-setup).
2. [Obtain models](docs/getting-started.md#explicit-model-acquisition) and [generate TensorRT engines](docs/getting-started.md#tensorrt-engine-generation).
3. Run the [inference samples](docs/getting-started.md#samples), use the [client library](docs/library.md), or start the [gRPC server](crates/audio2face3d-server/README.md).

After preparing the SDKs and Mark model, start the native server:

```powershell
cargo run -p audio2face3d-server --features cli -- --platform-config platform.toml --model models/mark/model.json
```

## Documentation

| Guide | Contents |
|---|---|
| [Getting started](docs/getting-started.md) | SDK setup, source installation, model downloads, engine generation, CLI samples |
| [Platform configuration](docs/platform.md) | Build/runtime path selection, diagnostics, version policy, loader lifetime |
| [Packages and features](docs/features.md) | Package boundaries, feature selection, SDK module mapping |
| [Client library](docs/library.md) | Configuration builders, direct/remote execution, authentication, shared context and logging |
| [Low-level runtime APIs](docs/runtime-api.md) | Executor composition, callbacks, interactive execution, safety contracts |
| [gRPC server](crates/audio2face3d-server/README.md) | Server CLI, embedding, authentication, health and streaming contracts |
| [Development and validation](docs/development.md) | Supported environments, test tiers, benchmarks |
| [Reference comparison](reference/README.md) | Original-SDK parity setup and capture procedures |

## License

This project is an independently maintained Rust port of NVIDIA's MIT-licensed
[Audio2Face-3D-SDK](https://github.com/NVIDIA/Audio2Face-3D-SDK), not an official
NVIDIA SDK release. The upstream source and reference revision is
`1ca0f02535ed774f5dbcd724a31cd486368dc783`.

The source code in this repository is available under the [MIT License](LICENSE),
except the Eigen-derived CPU SVD implementation, which is licensed under
[MPL-2.0](LICENSE-MPL-2.0). The ACE protocol definitions are covered by [Apache-2.0](crates/audio2face3d/LICENSE-APACHE).
The library package declares `MIT AND MPL-2.0 AND Apache-2.0`.
See [third-party notices](THIRD-PARTY-NOTICES.md) for the file scope and provenance.
It includes work derived from the NVIDIA Audio2Face-3D SDK and retains the
applicable NVIDIA copyright and license notice.

Models are obtained separately. Users are responsible for determining whether
and how to use them.

Externally installed CUDA and TensorRT SDK binaries, headers, and components
remain subject to their respective NVIDIA license terms.
