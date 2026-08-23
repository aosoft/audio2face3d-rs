#[cfg(feature = "cuda")]
#[test]
fn device_view_cannot_outlive_allocation() {
    let tests = trybuild::TestCases::new();
    tests.compile_fail("tests/ui/device_view_outlives_buffer.rs");
}
