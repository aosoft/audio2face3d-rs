# Native SDK configuration

[Project overview](../README.md) · [Getting started](getting-started.md)

Native inference uses explicitly loaded CUDA/TensorRT libraries. SDK-specific environment variables and PATH edits are optional. Build-time SDK selection and runtime library selection are independent; applications supply the versions they intend to use.

## Build configuration

Native builds still need CUDA headers and nvcc, TensorRT headers, and a compatible host C++ compiler. On Windows use an MSVC developer shell. The validated CUDA 12.9 build uses MSVC 14.42; the compiler environment's headers must match the chosen host compiler.

Create `native-build.toml` at this repository's root (ignored by Git):

```toml
schema_version = 1
cuda_root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
tensorrt_root = 'C:\SDK\TensorRT-10.16.1.11'
cuda_archs = ['86']
# Optional: select an installed compatible host compiler.
# cuda_host_compiler = 'C:\...\bin\Hostx64\x64\cl.exe'
```

Build configuration is selected in this order:

1. `AUDIO2FACE3D_BUILD_CONFIG`, when explicitly set.
2. This source repository's `native-build.toml`, only for its own workspace layout.
3. User configuration: `%LOCALAPPDATA%\audio2face3d\native-build.toml` on Windows; `$XDG_CONFIG_HOME/audio2face3d/native-build.toml` or `$HOME/.config/audio2face3d/native-build.toml` on Linux.
4. Legacy SDK variables, then an unambiguous installed SDK.

Registry dependencies and extracted packages do not search the consumer's current directory for a workspace configuration. Use the user configuration or explicitly select a file when building them. Relative paths inside the file are relative to that file. An invalid selected file fails the build; it does not fall back. Header versions are embedded for subsequent runtime comparison. Changing the selected build file triggers Cargo's build script again.

```powershell
cargo build --release --locked -p audio2face3d --features cli,native
cargo build --release --locked -p audio2face3d-server --features cli
```

`cuda_archs` defaults to `['86']`. Portable features require no native build configuration. The internal C++ shim is a static archive inside the Rust artifact; there is no project-specific shim DLL to deploy. NVIDIA DLLs are not ordinary executable imports. The selected NVIDIA SDK binaries remain separate dependencies.

## Runtime configuration

Create an application-owned file such as `native-runtime.toml`:

```toml
schema_version = 1
search_policy = 'explicit'
cuda_root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
tensorrt_root = 'C:\SDK\TensorRT-10.16.1.11'
```

Both executables accept the same global options:

```powershell
audio2face3d --runtime-config native-runtime.toml doctor
audio2face3d --runtime-config native-runtime.toml doctor --load --device 0
audio2face3d --runtime-config native-runtime.toml run regression models/mark/model.json 1 16000
audio2face3d-server --runtime-config native-runtime.toml --model models/mark/model.json
```

CLI flags override the corresponding file settings. `--cuda-root` and repeated `--cuda-library-dir` replace each other as a group; the same applies to TensorRT. Root and directory options for the same SDK cannot be specified together in one source. File-relative paths use the file's parent; CLI-relative paths use the working directory.

For a deployment directory layout, replace roots with `cuda_library_dirs = ['...']` and `tensorrt_library_dirs = ['...']`, or use repeated `--cuda-library-dir` / `--tensorrt-library-dir` flags. Include the selected SDK's required transitive libraries. Library paths passed directly to the Rust builder must be absolute.

`--runtime-search explicit` requires supplied locations. `discover` (the default) permits legacy `CUDA_PATH` / `TENSORRT_ROOT_DIR`, platform installation directories and existing loader search paths when no explicit location was provided. Explicit locations never fall back to another SDK. Multiple distinct candidate libraries fail rather than selecting the newest version. The CUDA driver comes from the system driver installation, not a Toolkit stub.

`doctor` inspects files without loading DLLs (`Discovered`). `doctor --load` initializes the native APIs (`Loaded`); adding `--device 0` also retains a CUDA device context (`DeviceReady`). `--json` exposes observed paths and versions. Build and runtime versions remain absent before they are observed. `--load` requires a native-enabled executable. Help, version, mock and remote operation do not initialize GPU libraries.

## Library use and lifetime

```rust,no_run
use audio2face3d::{Audio2Face3DContext, runtime::{NativeRuntimeConfig, NativeSearchPolicy}};

let native = NativeRuntimeConfig::builder()
    .cuda_root(r"C:\SDK\CUDA\v12.9")
    .tensorrt_root(r"C:\SDK\TensorRT-10.16.1.11")
    .search_policy(NativeSearchPolicy::ExplicitOnly)
    .build()?;
let context = Audio2Face3DContext::builder().native_runtime(native).build();
// Pass context to Client::direct_with_context or Server::builder(config).context(context).
# Ok::<(), audio2face3d::runtime::NativeRuntimeError>(())
```

Building or observing the Context does not load libraries. Native preparation initializes them synchronously on the existing preparation worker. `context.initialize_native()` is also available for explicit initialization when native features are enabled. `context.native_runtime_info()` returns only observations already made by that Context. Direct inference retains its runtime-independent asynchronous interface.

Context clones share resources. Process-wide native libraries are retained for process lifetime, while streams, allocations, sessions and device references are released through their normal owners. Dropping the caller's Context does not invalidate consumers. Same actual library files can be shared; a different SDK in the same process fails with `RuntimeConflict`. Hot reload and native module unloading are unsupported.

Initialization is serialized without holding the registry lock across native code or Logger callbacks. Other callers wait; same-thread reentry reports `InitializationReentered`. A failure before loading begins can be retried. A failure after loading starts is retained and reports `restart_required`; use a new process after correcting it. Cancellation of preparation still waits for ownership-safe cleanup.

On Windows, absolute `LoadLibraryExW` calls and retained `AddDllDirectory` entries allow transitive and delayed dependencies. These entries affect the process loader, but do not change PATH or the user/system environment. Already loaded NVIDIA dependencies are checked against the selected files. Applications embedding another CUDA user must coordinate process-wide SDK choices.

## Version policy

| Comparison of the same component | Result |
| --- | --- |
| CUDA Runtime or TensorRT major differs from build headers | Initialization error |
| Major matches, minor differs in either direction | One warning during shared initialization; continue |
| Only patch/build differs | No version warning or rejection |
| Driver version differs from Toolkit/Runtime major | No major-match requirement |

Warnings use the supplied Logger and obey its threshold; a default silent Context does not emit them. This policy does not guarantee all minor combinations work. Missing required symbols, native initialization failures, unsupported PTX/device operations, and incompatible serialized TensorRT engines remain errors. CUDA 12.9 / TensorRT 10.16.1.11 is the tested combination; version fixtures test policy rather than certify other NVIDIA releases.

## External tools

Engine generation resolves `trtexec` to an absolute path from the configured TensorRT location. `TRTEXEC` can still override the executable. Version probes use the same path configuration, including `nvcc` for Toolkit information.

`NativeRuntimeConfig::tool_command(NativeTool::Trtexec, None)` prepares a `std::process::Command` without starting it or initializing the parent's inference libraries. Only the child receives the required PATH (Windows) or LD_LIBRARY_PATH (Linux) prefix. Existing parent/user/system variables are unchanged. Tool execution and inference use the same selected locations; tool failures are reported normally.

## Validation scope

Windows x86-64/MSVC is the required target. Tests cover explicit-path inference, no NVIDIA load-time imports, fixture failures, concurrent initialization, resource lifetime, and an independent host with a statically linked CUDA Runtime. Linux remains experimental/build-only. Fourteen portable registry/loader, configuration and child-tool fixture tests passed on Ubuntu 22.04 under WSL; native Linux builds and GPU inference remain unverified because that environment has no CUDA/TensorRT SDK. A machine without an installed NVIDIA driver has not been tested; GPU-free startup was checked in child processes without SDK environment variables or SDK PATH entries.
