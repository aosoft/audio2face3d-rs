#![cfg(feature = "animation")]
#[test]
fn new_api_is_a_compile_pass_contract() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/new_api_compile_pass.rs");
}

#[test]
fn device_element_is_a_sealed_portable_contract() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/device_element_unsupported.rs");
    tests.compile_fail("tests/ui/device_element_is_sealed.rs");
    tests.pass("tests/ui/device_elements_compile_pass.rs");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lifetimes_and_auto_traits_are_compile_contracts() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/device_view_outlives_buffer.rs");
    tests.compile_fail("tests/ui/device_view_mut_is_exclusive.rs");
    tests.compile_fail("tests/ui/device_view_mut_is_not_clone.rs");
    tests.pass("tests/ui/cuda_auto_traits_are_thread_safe.rs");
}

#[cfg(all(feature = "animation", feature = "cuda"))]
#[test]
fn animation_cuda_lifetimes_are_compile_contracts() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/interactive_gpu_view_escapes_callback.rs");
    tests.compile_fail("tests/ui/teeth_fence_outlives_stream.rs");
}

#[cfg(feature = "tensorrt")]
#[test]
fn tensorrt_outputs_and_submissions_are_exclusive() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/tensorrt_output_requires_mutable_view.rs");
    tests.compile_fail("tests/ui/tensorrt_output_is_not_clone.rs");
    tests.compile_fail("tests/ui/tensorrt_bindings_are_exclusive.rs");
}
