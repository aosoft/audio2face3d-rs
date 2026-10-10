use audio2face3d::cuda::{DeviceElement, DeviceView};
use std::num::NonZeroU32;

fn require_element<T: DeviceElement>() {}

fn main() {
    require_element::<String>();
    require_element::<bool>();
    require_element::<NonZeroU32>();
    require_element::<&'static u32>();
    let _: Option<DeviceView<'static, String>> = None;
}
