use audio2face3d_headgen::{Config, generate};
#[test]
fn deterministic_valid_head_has_52_real_deformations() {
    let model = generate(&Config::default()).unwrap();
    let bytes = audio2face3d_gui_core::gltf::to_glb(&model).unwrap();
    assert_eq!(
        bytes,
        audio2face3d_gui_core::gltf::to_glb(&generate(&Config::default()).unwrap()).unwrap()
    );
    assert_eq!(model.channel_names().len(), 52);
    assert!((500..=1000).contains(&model.meshes[0].positions.len()));
    for name in model.channel_names() {
        assert!(
            model
                .meshes
                .iter()
                .flat_map(|m| &m.targets)
                .filter(|t| t.name == name)
                .any(|t| t.positions.iter().flatten().any(|x| x.abs() > 0.0001)),
            "{name}"
        );
    }
    for mesh in &model.meshes {
        let evaluated =
            audio2face3d_gui_core::model::evaluate(mesh, &vec![0.; mesh.targets.len()]).unwrap();
        assert_eq!(evaluated.positions, mesh.positions);
    }
}
#[test]
fn invalid_configuration_is_rejected_before_allocation() {
    let config = Config {
        segments: u32::MAX,
        ..Config::default()
    };
    assert!(generate(&config).is_err());
    let config = Config {
        expression_scale: f32::NAN,
        ..Config::default()
    };
    assert!(generate(&config).is_err());
}

#[test]
fn left_right_and_mouth_close_follow_the_rig_contract() {
    let model = generate(&Config::default()).unwrap();
    let left = model.meshes.iter().find(|m| m.name == "lidLeft").unwrap();
    assert!(left.targets.iter().any(|t| t.name == "EyeBlinkLeft"));
    assert!(!left.targets.iter().any(|t| t.name == "EyeBlinkRight"));
    let lips = model.meshes.iter().find(|m| m.name == "lips").unwrap();
    let close = lips
        .targets
        .iter()
        .find(|t| t.name == "MouthClose")
        .unwrap();
    assert!(close.positions.iter().all(|p| p[1] < 0.));
    for mesh in &model.meshes {
        let posed =
            audio2face3d_gui_core::model::evaluate(mesh, &vec![1.; mesh.targets.len()]).unwrap();
        assert!(
            posed
                .positions
                .iter()
                .chain(&posed.normals)
                .flatten()
                .all(|v| v.is_finite())
        );
    }
}
