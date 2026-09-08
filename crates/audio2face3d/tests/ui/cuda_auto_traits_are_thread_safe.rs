use audio2face3d::cuda::{
    CublasHandle, CudaEvent, CudaFunction, CudaModule, CudaStream, CudaStreamRef, CurandHandle,
    DeviceBuffer, DeviceView, GpuDevice,
};

fn require_send<T: Send>() {}
fn require_sync<T: Sync>() {}

fn main() {
    require_send::<CudaStream>();
    require_sync::<CudaStream>();
    require_send::<CudaEvent>();
    require_sync::<CudaEvent>();
    require_send::<CudaModule>();
    require_sync::<CudaModule>();
    require_send::<CudaFunction<'static>>();
    require_sync::<CudaFunction<'static>>();
    require_send::<GpuDevice>();
    require_sync::<GpuDevice>();
    require_send::<DeviceBuffer<f32>>();
    require_sync::<DeviceBuffer<f32>>();
    require_send::<DeviceView<'static, f32>>();
    require_sync::<DeviceView<'static, f32>>();
    require_send::<CudaStreamRef<'static>>();
    require_sync::<CudaStreamRef<'static>>();
    require_send::<CublasHandle>();
    require_sync::<CublasHandle>();
    require_send::<CurandHandle>();
}
