# Platform configuration

[Project overview](../README.md) · [Getting started](getting-started.md)

Use one `platform.toml` for native builds and the inference, server and GUI executables. SDK environment variables and PATH edits are optional. Keys use kebab-case; Rust fields and methods keep snake_case. There is no schema-version field.

## One file for build and runtime

Create `platform.toml` at this repository's root (ignored by Git):

```toml
cuda-root = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
tensorrt-root = 'C:\SDK\TensorRT-10.16.1.11'

[build]
# Optional: select an installed compatible host compiler.
# cuda-host-compiler = 'C:\...\bin\Hostx64\x64\cl.exe'

[runtime]
search-policy = 'explicit'
```

The common roots apply to both phases. Both sections are optional. Each phase can override roots inside its own section; build settings never select runtime libraries and runtime settings never select compiler inputs.

| Location | Accepted keys | Purpose |
| --- | --- | --- |
| Top level | `cuda-root`, `tensorrt-root` | Shared SDK locations |
| `[build]` | `cuda-root`, `tensorrt-root`, `cuda-host-compiler` | Build-only overrides and compiler options |
| `[runtime]` | `cuda-root`, `tensorrt-root`, `cuda-library-dirs`, `tensorrt-library-dirs`, `search-policy` | Runtime-only overrides and library discovery |

All file-relative paths are relative to the configuration file. Unknown keys, wrong types and empty path lists are errors, including in the other phase's section. The whole file is validated before it is used. Actual SDK files are checked only by the phase that uses them; for example, running a distributed executable does not require a build host compiler to exist.

A deployment can keep its build SDK locations and override only the runtime layout:

```toml
cuda-root = 'C:\BuildSDK\CUDA'
tensorrt-root = 'C:\BuildSDK\TensorRT'

[runtime]
cuda-library-dirs = ['deploy/cuda']
tensorrt-library-dirs = ['deploy/tensorrt']
search-policy = 'explicit'
```

A runtime library-directory list replaces that SDK's common root. A root and library-directory list for the same SDK cannot both appear inside `[runtime]`. Include required transitive libraries in the deployment. Runtime-only files can omit common/build settings; native builds still require the corresponding SDK roots.

## File selection

The GUI can also reference this file through `platform-config` in its GUI TOML.
That reference takes precedence over the environment variable but not over
`--platform-config`; its path is relative to the GUI TOML. SDK paths inside the
referenced file remain relative to the platform file itself. See [GUI configuration](gui.md#gui-configuration).
Copy [`platform.example.toml`](../platform.example.toml) as a starting point.

Only one file is read; files are not merged. Missing explicitly selected files and invalid selected files fail without falling back.

| Priority | Native build | Executable runtime |
| --- | --- | --- |
| 1 | `AUDIO2FACE3D_PLATFORM_CONFIG` | `--platform-config <FILE>` |
| 2 | This source workspace's `platform.toml` | `AUDIO2FACE3D_PLATFORM_CONFIG` |
| 3 | User `platform.toml` | `platform.toml` in the working directory |
| 4 | Legacy SDK variables / installed SDK search | User `platform.toml` |
| 5 | — | Default runtime discovery |

User configuration is `%LOCALAPPDATA%\audio2face3d\platform.toml` on Windows, or `$XDG_CONFIG_HOME/audio2face3d/platform.toml` / `$HOME/.config/audio2face3d/platform.toml` on Linux. Registry dependencies and extracted packages do not guess the consumer workspace. For `cargo install` or another workspace, use user configuration or set `AUDIO2FACE3D_PLATFORM_CONFIG` to an absolute path. Relative environment paths are resolved against each reader's working directory, which can differ between Cargo's build script and the executable.

## When no configuration is supplied

The file is optional. If no file is selected or found and no SDK flags are supplied, the existing environment-based discovery remains active:

- Native builds use `CUDA_PATH`, `TENSORRT_ROOT_DIR` and `AUDIO2FACE3D_CUDA_HOST_COMPILER`; missing SDK roots use the first installed SDK in path-name order. The usual compiler environment is still required. PATH alone does not select build SDK roots.
- Executables use the default `discover` policy: `CUDA_PATH` and `TENSORRT_ROOT_DIR` select SDK roots; otherwise `PATH` (Windows) or `LD_LIBRARY_PATH` (Linux) is searched in order, followed by installed SDK directories. Environment-selected roots take precedence over those search paths.

Omitting `--platform-config` does not disable automatic file selection: a workspace/working-directory or user `platform.toml` still takes precedence over legacy SDK variables. An empty runtime configuration also uses discovery, but a selected build configuration must supply the required SDK roots; missing build fields are not filled from environment variables. Invalid files or invalid explicit SDK locations remain errors.

When an SDK location is not explicitly configured, discovery accepts the first matching binary in search-path order, even if other versions are installed. Installed directories and filenames within one directory use path-name order for repeatability; this is not a newest-version selection. Explicit roots and library-directory lists from a file, CLI flags or library API still require a unique binary per component. Missing files, incompatible major versions and conflicts with already loaded libraries remain errors.

## Build and install

Native builds need CUDA headers and nvcc, TensorRT headers, and a compatible host C++ compiler. On Windows use an MSVC developer shell. The validated CUDA 12.9 build uses MSVC 14.42; its compiler environment's headers must match the chosen host compiler.

From this checkout, the root `platform.toml` is used automatically:

```powershell
cargo build --release --locked -p audio2face3d --features cli,native
cargo build --release --locked -p audio2face3d-server --features cli
```

To explicitly select the same file for build, installation and execution:

```powershell
$env:AUDIO2FACE3D_PLATFORM_CONFIG = (Resolve-Path ./platform.toml).Path
cargo install --path crates/audio2face3d --locked --features cli,native
cargo install --path crates/audio2face3d-server --locked --features cli
```

Portable builds need no SDK configuration. File changes trigger the native build script again. Header versions are embedded for runtime comparison. The C++ shim is a static archive inside the Rust artifact; there is no project-specific DLL to deploy and no ordinary NVIDIA DLL imports. NVIDIA SDK binaries remain separate runtime dependencies.

### CUDA kernel compilation

The build queries the selected `nvcc --list-gpu-arch` and uses the lowest generic PTX target supported by both the compiler and this project's CUDA kernels. No architecture setting or build-machine GPU detection is required.

The four kernels compile at `compute_50`, the lowest target supported by the validated CUDA 12.9 toolchain. A compiler that no longer supports that target selects its next supported generic target. The driver compiles the generated PTX for the execution GPU. See [NVIDIA's PTX compilation documentation](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-compiler-driver-nvcc/index.html#just-in-time-compilation).

This target covers the project's post-processing kernels. Supported GPUs for complete inference also depend on CUDA libraries, TensorRT and the model engine.

## CLI options

Both executables accept these global options. Individual SDK flags override runtime values from the selected file.

| Option | Meaning |
| --- | --- |
| `--platform-config <FILE>` | Read the shared file and use its common roots plus runtime section |
| `--cuda-root <PATH>`, `--tensorrt-root <PATH>` | Override a runtime SDK root |
| `--cuda-library-dir <PATH>`, `--tensorrt-library-dir <PATH>` | Override with library directories; repeat for multiple directories |
| `--runtime-search explicit` / `discover` | Override `runtime.search-policy` |

Root and directory flags replace the inherited SDK group, and are mutually exclusive for the same SDK. CLI-relative paths use the working directory.

```powershell
audio2face3d --platform-config platform.toml doctor
audio2face3d --platform-config platform.toml doctor --load --device 0
audio2face3d --platform-config platform.toml run regression models/mark/model.json 1 16000
audio2face3d-server --platform-config platform.toml --model models/mark/model.json

# From this checkout: Cargo and the executable both find the root platform.toml.
cargo run -p audio2face3d-server --features cli -- --model models/mark/model.json
```

Options after Cargo's `--` affect the running executable only. They cannot change a build that already happened. Select a custom build file through `AUDIO2FACE3D_PLATFORM_CONFIG`; `--platform-config` can independently choose another runtime file with the same format.

The default `discover` policy uses legacy `CUDA_PATH` / `TENSORRT_ROOT_DIR`, then loader search paths, then platform installation directories when no explicit location was provided. Multiple discovery candidates are allowed and the first match is used. `explicit` requires supplied locations. Explicit locations never fall back to another SDK and still reject ambiguous component binaries. The CUDA driver comes from the system installation.

`doctor` inspects files without loading DLLs (`Discovered`). `doctor --load` initializes native APIs (`Loaded`); adding `--device 0` retains a CUDA device context (`DeviceReady`). `--json` reports observed paths and versions. `--load` requires a native-enabled executable. Help, version, mock and remote operation do not initialize GPU libraries.

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

Context and library builders do not implicitly read platform.toml or AUDIO2FACE3D_PLATFORM_CONFIG; the embedding application supplies NativeRuntimeConfig. Building or observing the Context does not load libraries. Native preparation initializes them synchronously on the existing preparation worker. `context.initialize_native()` is also available for explicit initialization when native features are enabled. `context.native_runtime_info()` returns only observations already made by that Context. Direct inference retains its runtime-independent asynchronous interface.

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

`NativeRuntimeConfig::tool_command(NativeTool::Trtexec, None)` prepares a `std::process::Command` without starting it or initializing the parent's inference libraries. Only the child receives the required PATH (Windows) or LD_LIBRARY_PATH (Linux) prefix. Existing parent/user/system variables are unchanged. Unconfigured tool selection follows SDK root environment variables, then PATH, then installed locations, accepting the first executable. Explicit SDK locations retain uniqueness checks. Tool execution and inference use the same discovery policy; tool failures are reported normally.

## Validation scope

Native inference is validated on Windows x86-64/MSVC with CUDA 12.9 and TensorRT 10.16.1.11, using either explicit SDK paths or environment-based discovery.

Linux support is experimental. Tests that do not require CUDA/TensorRT have passed under WSL, but native Linux builds and GPU inference have not been verified.

Startup without SDK environment variables or SDK PATH entries has been verified on a Windows PC with an NVIDIA driver installed. Startup on a PC without an NVIDIA driver has not been tested.
