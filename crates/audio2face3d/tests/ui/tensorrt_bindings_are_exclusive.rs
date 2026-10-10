use audio2face3d::cuda::CudaStream;
use audio2face3d::tensorrt::{DeviceBindings, TensorRtSession};

fn invalid(first: &mut TensorRtSession, second: &mut TensorRtSession, bindings: &mut DeviceBindings<'_>, stream: &CudaStream) {
    let pending = first.enqueue(0, bindings, stream).unwrap();
    let _conflicting = second.enqueue(0, bindings, stream).unwrap();
    pending.synchronize().unwrap();
}

fn main() {}
