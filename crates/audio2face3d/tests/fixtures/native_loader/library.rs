#[link(name = "a2f_fixture_dependency")]
unsafe extern "C" { fn fixture_dependency_version() -> u32; }
#[unsafe(no_mangle)]
pub extern "C" fn fixture_version() -> u32 {
    // SAFETY: the fixture dependency exports this exact ABI.
    unsafe { fixture_dependency_version() }
}
