#![cfg(feature = "animation")]
#[test]
fn new_api_is_a_compile_pass_contract() {
    let tests = trybuild::TestCases::new();
    tests.pass("tests/ui/new_api_compile_pass.rs");
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_lifetimes_and_auto_traits_are_compile_contracts() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/device_view_outlives_buffer.rs");
    tests.pass("tests/ui/cuda_auto_traits_are_thread_safe.rs");
}

#[cfg(all(feature = "animation", feature = "cuda"))]
#[test]
fn animation_cuda_lifetimes_are_compile_contracts() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/interactive_gpu_view_escapes_callback.rs");
    tests.compile_fail("tests/ui/teeth_fence_outlives_stream.rs");
}
