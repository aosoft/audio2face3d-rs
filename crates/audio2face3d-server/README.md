# Audio2Face-3D gRPC server

An embeddable server library and optional CLI for generating facial animation
through the ACE `A2FControllerService/ProcessAudioStream` bidirectional RPC.

Part of an **unofficial, independently maintained Rust port of NVIDIA's
Audio2Face-3D SDK**, with additional client and server functionality.

- Native Regression inference through `audio2face3d`, returning 52 face curves
  with audio.
- Bounded FIFO request queuing with configurable concurrency.
- Application-provided authentication and structured logging.
- A mock backend for diagnostics without CUDA or TensorRT.

The default `native` feature requires CUDA, TensorRT, and separately obtained
models. Enable `cli` to build the executable. For a portable diagnostic build,
use `--no-default-features --features cli,mock`.

## Documentation

See the [server guide](../../docs/server.md)
for setup, CLI usage, library embedding, authentication, health, streaming
contracts, mock diagnostics, and supported scope.

- [Model and SDK setup](../../docs/getting-started.md)
- [Platform configuration](../../docs/platform.md)
- [Logging](../../docs/logging.md)

## License

See the [project license and notices](../../README.md#license).
The NVIDIA protocol definitions retain their upstream notices and are covered
by [Apache-2.0](LICENSE-APACHE).
