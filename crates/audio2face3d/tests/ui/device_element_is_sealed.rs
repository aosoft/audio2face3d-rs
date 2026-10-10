use audio2face3d::cuda::DeviceElement;

#[derive(Clone, Copy)]
struct Unchecked(u32);

impl DeviceElement for Unchecked {}

fn main() {}
