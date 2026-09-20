// Controlled versions returned by a real loaded fixture, not installed NVIDIA libraries.
#[unsafe(no_mangle)]
pub extern "C" fn fixture_version(case: u32) -> u32 {
    match case { 0 => 101601, 1 => 111601, 2 => 91601, 3 => 101501, 4 => 101701, 5 => 101699, _ => 0 }
}
