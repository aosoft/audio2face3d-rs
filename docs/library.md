# Client library and shared resources

[Project overview](../README.md) · [Documentation index](../README.md#documentation)

Configure direct or remote inference, inject shared logging, and provide remote credentials.

## Direct and remote execution

`Client::direct` and `Client::server` expose the same session types after initialization. Direct inference is independent of an async runtime; remote inference uses a caller-owned Tokio runtime. Drive input and output concurrently, await request completion, and call `Client::shutdown` before dropping the runtime. See the [streaming example](../crates/audio2face3d/examples/stream.rs).

## Configuration builders

Use consuming builders to specify optional settings and validate before starting work.
Configuration fields are private outside their package. Construct these types
through their builders; public `new`, `Default`, and struct-literal construction
are not supported. Read values through accessors. To revise a configuration,
use `into_builder()`, change the desired options, then call `build()` again.

| Configuration | Builder entry |
|---|---|
| Remote client | `client::ServerConfig::builder(endpoint)` |
| Direct client | `client::DirectConfig::builder(engine_config)` |
| Client limits | `client::Limits::builder()` |
| Inference engine | `inference::Config::builder(backend)` |
| Inference request | `types::RequestOptions::builder(input_format)` |
| Server service | `audio2face3d_server::ServerConfig::builder(backend)` |

Setters consume and return the builder. `build()` returns a validated configuration
or an error, without connecting or loading inference engines. Optional fields have
both `field(value)` and `optional_field(Option<T>)` setters, so callers can supply
runtime options or reset a value to `None`. Unset, zero and false remain distinct.
The backend is explicit at builder construction, avoiding feature-dependent defaults.
Builder support introduces no dependency or asynchronous runtime requirement.

## Remote client authentication

With `client-grpc`, use `ServerConfig::builder(endpoint).api_key(key).build()?`
when initializing `Client::server`. Omitting the key sends no
authorization header. Each inference RPC carries `authorization: Bearer <key>`;
authentication is performed per RPC, not when the transport connects.
The same setting applies to cloned clients. Create a new client to change the key.

Keys must be nonempty RFC 6750 Bearer tokens of at most 4096 bytes; invalid
values fail initialization without being echoed. Configuration `Debug` redacts
the key. Credentials are not part of the shared inference request types.
Authentication rejections currently surface as `ErrorKind::Transport`.
Use HTTPS or a trusted local transport when sending credentials.

```rust,ignore
let config = audio2face3d::client::ServerConfig::builder(endpoint)
    .api_key(api_key)
    .build()?;
let client = audio2face3d::client::Client::server(config).await?;
```

## Native runtime paths

Attach `NativeRuntimeConfig` with `.native_runtime(config)` when building `Audio2Face3DContext`. This supplies direct inference and embedded servers with the same SDK locations and shared native ownership. Configuration and diagnostics remain GPU-free until native preparation. See [platform configuration](platform.md) for an example, version checks and process lifetime rules.

## Shared context and logging

`Audio2Face3DContext` owns shared application resources. Build it with an `Arc<dyn Logger>` and pass it to `Client::direct_with_context`, `Client::server_with_context`, `inference::Factory::prepare_with_context`, a low-level `load_with_context` factory, or `Server::builder(config).context(context)`. Clones share the same resources. Context drop does not stop consumers or flush the logger.

The standard-library-only `Logger` receives an owned `LogRecord` containing a message and optional typed fields. `log(level, || record)` constructs neither message nor fields below the configured level. Both libraries use this path for their own diagnostics, including RPC events; unconfigured contexts are silent. See [Logging](logging.md) for an implementation example, JSONL output, tracing boundaries, and migration details.

## Server library

Server construction, authenticator injection, health policy, and cleanup ownership are documented in the [server guide](server.md#library-logging-and-authentication).
