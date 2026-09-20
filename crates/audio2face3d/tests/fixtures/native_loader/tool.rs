fn main() {
    let variable=if cfg!(windows) { "PATH" } else { "LD_LIBRARY_PATH" };
    println!("{}",std::env::var_os(variable).unwrap_or_default().to_string_lossy());
}
