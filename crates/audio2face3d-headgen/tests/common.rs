use audio2face3d_headgen::config::Config;
pub fn fixture() -> (tempfile::TempDir, Config) {
    let dir = tempfile::tempdir().unwrap();
    let mut config = Config::parse(include_str!("../presets/ict-facekit.toml")).unwrap();
    config.expected = None;
    config.neutral = "基準.obj".into();
    config.targets = std::collections::BTreeMap::from([
        (
            "BrowInnerUp".into(),
            vec!["left.obj".into(), "right.obj".into()],
        ),
        ("JawOpen".into(), vec!["jaw.obj".into()]),
    ]);
    config.unsupported_channels = audio2face3d_gui::rig::CHANNELS
        .iter()
        .filter(|s| !config.targets.contains_key(**s))
        .map(|s| s.to_string())
        .collect();
    config.transform.fit_height_m = 1.;
    config.transform.center_m = [0.5, 0.5, 0.];
    config.geometry.exclude_materials.clear();
    for (name, z) in [
        ("基準.obj", [0., 0., 0.]),
        ("left.obj", [0.25, 0., 0.]),
        ("right.obj", [0., 0.5, 0.]),
        ("jaw.obj", [0., 0., 0.75]),
    ] {
        std::fs::write(
            dir.path().join(name),
            format!("v 0 0 {}\nv 1 0 {}\nv 0 1 {}\nf 1 2 3\n", z[0], z[1], z[2]),
        )
        .unwrap();
    }
    config.validate().unwrap();
    (dir, config)
}
