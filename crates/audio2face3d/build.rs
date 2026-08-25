#[path = "build_cuda.rs"]
mod build_cuda;
#[path = "build_tensorrt.rs"]
mod build_tensorrt;

fn main() {
    build_cuda::build();
    build_tensorrt::build();
}
