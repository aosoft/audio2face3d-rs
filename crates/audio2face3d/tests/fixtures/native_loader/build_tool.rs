//! A tool process that records its own SDK environment without loading an SDK.
fn main() {
    let variable = if cfg!(windows) { "PATH" } else { "LD_LIBRARY_PATH" };
    let environment = std::env::var_os(variable).unwrap_or_default();
    let output = format!("[TensorRT vfixture]\n{}", environment.to_string_lossy());
    for argument in std::env::args().skip(1) {
        if let Some(path) = argument.strip_prefix("--saveEngine=") {
            std::fs::write(path, output.as_bytes()).unwrap();
        }
    }
    println!("{output}");
}
