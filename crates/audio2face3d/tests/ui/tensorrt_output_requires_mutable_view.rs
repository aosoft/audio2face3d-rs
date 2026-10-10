use audio2face3d::cuda::DeviceBuffer;
use audio2face3d::tensorrt::OutputBinding;

fn invalid(buffer: &DeviceBuffer<f32>) {
    let _output = OutputBinding::from_view(buffer.view());
}

fn main() {}
