#[cfg(feature = "cuda")]
#[path = "build_cuda.rs"]
mod build_cuda;
#[cfg(feature = "cuda")]
#[path = "build_native.rs"]
mod build_native;
#[cfg(any(feature = "client-grpc", feature = "grpc-server"))]
#[path = "build_protocol.rs"]
mod build_protocol;
#[cfg(feature = "tensorrt")]
#[path = "build_tensorrt.rs"]
mod build_tensorrt;
fn main() {
    #[cfg(feature = "cuda")]
    let native = build_native::resolve(cfg!(feature = "tensorrt"))
        .unwrap_or_else(|error| panic!("native SDK configuration: {error}"));
    #[cfg(feature = "cuda")]
    build_native::emit_versions(&native).expect("native SDK version headers");
    #[cfg(feature = "cuda")]
    build_cuda::build(&native);
    #[cfg(feature = "tensorrt")]
    build_tensorrt::build(&native);
    #[cfg(any(feature = "client-grpc", feature = "grpc-server"))]
    build_protocol::build().expect("ACE protocol generation failed");
}
