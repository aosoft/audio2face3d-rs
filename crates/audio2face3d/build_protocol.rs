use std::{env, error::Error, path::PathBuf};

pub fn build() -> Result<(), Box<dyn Error>> {
    let protos = [
        "proto/nvidia_ace.services.a2f_controller.v1.proto",
        "proto/nvidia_ace.emotion_aggregate.v1.proto",
    ]
    .map(PathBuf::from);
    println!("cargo:rerun-if-changed=proto");
    let mut config = tonic_prost_build::Config::new();
    config.protoc_executable(protoc_bin_vendored::protoc_bin_path()?);
    config.include_file("ace.rs");
    tonic_prost_build::configure()
        .build_client(cfg!(feature = "client-grpc"))
        .build_server(cfg!(feature = "grpc-server"))
        .file_descriptor_set_path(PathBuf::from(env::var("OUT_DIR")?).join("ace_descriptor.bin"))
        .compile_with_config(
            config,
            &protos,
            &[PathBuf::from("proto"), protoc_bin_vendored::include_path()?],
        )?;
    Ok(())
}
