use audio2face3d::animation::{InteractiveGpuBlendshapeLayer, RegressionGeometry};
use audio2face3d::cuda::DeviceView;

fn escape_view<'a>(
    layer: &'a mut InteractiveGpuBlendshapeLayer,
    geometry: &RegressionGeometry,
) -> Option<DeviceView<'a, f32>> {
    let mut escaped = None;
    layer
        .compute_frame(0, 1, geometry, |output| {
            escaped = output.skin_weights;
            true
        })
        .unwrap();
    escaped
}

fn main() {}
