use std::ffi::{c_char, c_void};

#[repr(C)]
pub struct TrtSessionHandle {
    _private: [u8; 0],
}
#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TensorInfo {
    pub io_mode: i32,
    pub data_type: i32,
    pub location: i32,
    pub rank: i32,
}

unsafe extern "C" {
    pub fn trt_shim_create(
        path: *const c_char,
        device: i32,
        error: *mut c_char,
        cap: usize,
    ) -> *mut TrtSessionHandle;
    pub fn trt_shim_destroy(session: *mut TrtSessionHandle);
    pub fn trt_shim_tensor_count(session: *const TrtSessionHandle) -> i32;
    pub fn trt_shim_tensor_name(
        session: *const TrtSessionHandle,
        index: i32,
        name: *mut c_char,
        name_cap: usize,
        required: *mut usize,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
    pub fn trt_shim_tensor_info_at(
        session: *const TrtSessionHandle,
        index: i32,
        info: *mut TensorInfo,
        dims: *mut i64,
        dims_cap: i32,
        rank: *mut i32,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
    pub fn trt_shim_tensor_profile_dims(
        session: *const TrtSessionHandle,
        index: i32,
        profile: i32,
        selector: i32,
        dims: *mut i64,
        dims_cap: i32,
        rank: *mut i32,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
    pub fn trt_shim_profile_count(session: *const TrtSessionHandle) -> i32;
    pub fn trt_shim_set_profile(
        session: *mut TrtSessionHandle,
        profile: i32,
        stream: *mut c_void,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
    pub fn trt_shim_set_input_shape(
        session: *mut TrtSessionHandle,
        name: *const c_char,
        dims: *const i64,
        rank: i32,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
    pub fn trt_shim_infer_shapes(
        session: *mut TrtSessionHandle,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
    pub fn trt_shim_context_tensor_dims(
        session: *const TrtSessionHandle,
        index: i32,
        dims: *mut i64,
        dims_cap: i32,
        rank: *mut i32,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
    pub fn trt_shim_set_tensor_address(
        session: *mut TrtSessionHandle,
        name: *const c_char,
        address: *mut c_void,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
    pub fn trt_shim_enqueue(
        session: *mut TrtSessionHandle,
        stream: *mut c_void,
        error: *mut c_char,
        error_cap: usize,
    ) -> i32;
}
