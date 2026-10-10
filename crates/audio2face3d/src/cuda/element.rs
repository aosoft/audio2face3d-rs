//! Element types whose representations may be transferred by CUDA.

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i32 {}
    impl Sealed for f32 {}
}

/// An audited, plain numeric representation usable in device memory.
///
/// Every bit pattern, including all-zero bytes, is a valid value. Implementors
/// have no padding, references, or owned resources. This trait is sealed so
/// downstream crates cannot introduce representations that violate these rules.
/// It is available without the `cuda` feature, like [`super::DeviceView`].
pub trait DeviceElement: sealed::Sealed + Copy + Send + Sync + 'static {}

impl DeviceElement for u8 {}
impl DeviceElement for u32 {}
impl DeviceElement for u64 {}
impl DeviceElement for i32 {}
impl DeviceElement for f32 {}
