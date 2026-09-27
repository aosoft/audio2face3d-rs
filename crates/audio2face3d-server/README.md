# audio2face3d-server

Embeddable ACE-compatible gRPC server and executable. Defaults enable `native,cli`.
From the repository root:

```sh
cargo run -p audio2face3d-server -- --model models/mark/model.json
```

Configure native dependencies in `platform.toml`. For portable diagnostics use
`--no-default-features --features cli,mock`. Library users disable default features
and select a backend. See the [server guide](../../docs/server.md).
