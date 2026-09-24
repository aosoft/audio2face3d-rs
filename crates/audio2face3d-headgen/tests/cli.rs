mod common;
use std::process::Command;
#[test]
fn cli_round_trip_and_output_protection() {
    let (dir, c) = common::fixture();
    let config = dir.path().join("設定 file.toml");
    std::fs::write(&config, toml::to_string(&c).unwrap()).unwrap();
    let output = dir.path().join("head.glb");
    let run = |command: &str, extra: &[&std::ffi::OsStr]| {
        Command::new(env!("CARGO_BIN_EXE_audio2face3d-headgen"))
            .arg(command)
            .arg("--config")
            .arg(&config)
            .arg("--input-root")
            .arg(dir.path())
            .args(extra)
            .output()
            .unwrap()
    };
    let inspected = run("inspect", &[]);
    assert!(inspected.status.success(), "{:?}", inspected);
    assert!(!output.exists());
    let converted = run("convert", &["--output".as_ref(), output.as_os_str()]);
    assert!(converted.status.success(), "{:?}", converted);
    let bytes = std::fs::read(&output).unwrap();
    let model = audio2face3d_gui_core::gltf::from_glb(&bytes).unwrap();
    assert_eq!(model.channel_names().len(), 2);
    assert_eq!(
        run("convert", &["--output".as_ref(), output.as_os_str()])
            .status
            .code(),
        Some(4)
    );
    let bad_report = dir.path().join("missing/report.json");
    assert_eq!(
        run(
            "convert",
            &[
                "--output".as_ref(),
                output.as_os_str(),
                "--force".as_ref(),
                "--report".as_ref(),
                bad_report.as_os_str()
            ]
        )
        .status
        .code(),
        Some(4)
    );
    assert_eq!(std::fs::read(&output).unwrap(), bytes);
    assert_eq!(
        run(
            "convert",
            &["--output".as_ref(), config.as_os_str(), "--force".as_ref()]
        )
        .status
        .code(),
        Some(4)
    );
    std::fs::write(dir.path().join("jaw.obj"), "broken").unwrap();
    assert_eq!(
        run(
            "convert",
            &["--output".as_ref(), output.as_os_str(), "--force".as_ref()]
        )
        .status
        .code(),
        Some(3)
    );
    assert_eq!(std::fs::read(&output).unwrap(), bytes);
    let bad = std::fs::read_to_string(&config)
        .unwrap()
        .replace("audio2face_rs_tester_v1", "unknown");
    std::fs::write(&config, bad).unwrap();
    assert_eq!(run("inspect", &[]).status.code(), Some(2));
}
