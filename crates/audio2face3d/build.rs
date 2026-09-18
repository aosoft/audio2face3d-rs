#[cfg(feature = "cuda")]
#[path = "build_cuda.rs"]
mod build_cuda;
#[cfg(any(feature = "client-grpc", feature = "grpc-server"))]
#[path = "build_protocol.rs"]
mod build_protocol;
#[cfg(feature = "tensorrt")]
#[path = "build_tensorrt.rs"]
mod build_tensorrt;
fn main() {
    #[cfg(feature = "cuda")]
    build_cuda::build();
    #[cfg(feature = "tensorrt")]
    build_tensorrt::build();
    #[cfg(any(feature = "client-grpc", feature = "grpc-server"))]
    build_protocol::build().expect("ACE protocol generation failed");
}
