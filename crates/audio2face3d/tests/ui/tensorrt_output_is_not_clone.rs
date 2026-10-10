use audio2face3d::cuda::DeviceBuffer;
use audio2face3d::tensorrt::OutputBinding;

fn invalid(buffer: &mut DeviceBuffer<f32>) {
    let output = OutputBinding::from_view(buffer.view_mut());
    let _duplicate = output.clone();
}

fn main() {}
