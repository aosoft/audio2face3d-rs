use audio2face3d::cuda::CudaStream;

fn require_send<T: Send>() {}

fn main() {
    require_send::<CudaStream>();
}
