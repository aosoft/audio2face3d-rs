use audio2face3d::cuda::{DeviceElement, DeviceView, DeviceViewMut};

fn require_element<T: DeviceElement>() {}
fn require_thread_safe<T: Send + Sync>() {}

fn main() {
    require_element::<u8>();
    require_element::<u32>();
    require_element::<u64>();
    require_element::<i32>();
    require_element::<f32>();
    require_thread_safe::<DeviceView<'static, f32>>();
    require_thread_safe::<DeviceViewMut<'static, f32>>();
}
