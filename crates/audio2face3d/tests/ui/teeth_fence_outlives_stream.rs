use audio2face3d::animation::{
    GpuMultiTrackTeethAnimator, GpuMultiTrackTeethFence, GpuTeethInputBatch,
    GpuTeethOutputBatch,
};
use audio2face3d::cuda::{DeviceBuffer, GpuDevice};

fn invalid_fence(
    device: &std::sync::Arc<GpuDevice>,
    animator: &'static mut GpuMultiTrackTeethAnimator,
    input: &'static DeviceBuffer<f32>,
    output: &'static mut DeviceBuffer<f32>,
) -> GpuMultiTrackTeethFence<'static> {
    let stream = device.create_stream().unwrap();
    let input_info = animator.input_batch_info();
    let output_info = animator.output_batch_info();
    animator
        .compute(
            GpuTeethInputBatch::new(input, input_info),
            GpuTeethOutputBatch::new(output, output_info),
            &stream,
        )
        .unwrap()
}

fn main() {}
