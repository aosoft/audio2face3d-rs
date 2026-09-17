# audio2face3d-protocol

Internal NVIDIA ACE wire definitions and consuming conversions to
`audio2face3d-types`. Default features generate messages and adapters only.
Enable `client` for gRPC client bindings or `server` for server bindings;
these features enable the tonic transport dependencies.

The nine vendored schemas retain their original bytes and notices. See
[LICENSE-APACHE](LICENSE-APACHE) for the upstream schema license. The Rust
adapter implementation uses the workspace MIT license.

Converters move owned PCM and curve vectors. They preserve optional request
containers and explicit zero/false. `EncodedRequest::timeout` must be applied
by the transport driver; it is not an audio-header field. Packet conversion
alone does not establish a successful or continuous stream: the driver must
validate cross-packet ordering and normal final status/trailers.

Unsupported camera/joint data is rejected. Unknown optional metadata generates
a diagnostic; known emotion metadata is decoded and validated. Domain types
remain independent of generated messages and asynchronous runtimes.
