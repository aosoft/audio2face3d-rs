use audio2x_cuda::{DeviceView, GpuDevice};

fn invalid_view() -> DeviceView<'static, f32> {
    let device = GpuDevice::new(0).unwrap();
    let buffer = device.allocate::<f32>(4).unwrap();
    buffer.view()
}

fn main() {
    let _ = invalid_view();
}
