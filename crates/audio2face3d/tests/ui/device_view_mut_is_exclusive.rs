use audio2face3d::cuda::DeviceBuffer;

fn invalid(buffer: &mut DeviceBuffer<f32>) {
    let output = buffer.view_mut();
    let input = buffer.view();
    println!("{} {}", output.len(), input.len());
}

fn main() {}
