mod common;
use audio2face3d_gui_core::{gltf::to_glb, model::evaluate};
use audio2face3d_headgen::convert;
#[test]
fn additive_shapes_evaluate_and_round_trip_deterministically() {
    let (dir, c) = common::fixture();
    let result = convert(&c, dir.path()).unwrap();
    let mesh = &result.model.meshes[0];
    assert_eq!(mesh.targets[0].name, "BrowInnerUp");
    for w in [0., 0.5, 1.] {
        let p = evaluate(mesh, &[w, 0.]).unwrap();
        assert_eq!(p.positions[0], [0., 0., 0.25 * w]);
        assert_eq!(p.positions[1], [1., 0., 0.5 * w]);
    }
    let both = evaluate(mesh, &[1., 1.]).unwrap();
    assert_eq!(both.positions[2], [0., 1., 0.75]);
    let full = evaluate(mesh, &[1., 0.]).unwrap();
    let expected = [-0.25f32, 0.25, 1.];
    let length = expected.iter().map(|x| x * x).sum::<f32>().sqrt();
    for (actual, expected) in full.normals[0].iter().zip(expected) {
        assert!((actual - expected / length).abs() < 1e-6);
    }
    let bytes = to_glb(&result.model).unwrap();
    assert_eq!(
        bytes,
        to_glb(&convert(&c, dir.path()).unwrap().model).unwrap()
    );
    assert_eq!(result.report.channels.len(), 2);
    assert!(result.report.output_glb_sha256.is_none());
}
#[test]
fn rejects_zero_shapes_and_excluded_topology_mismatch() {
    let (dir, mut c) = common::fixture();
    std::fs::copy(dir.path().join(&c.neutral), dir.path().join("jaw.obj")).unwrap();
    assert!(
        convert(&c, dir.path())
            .err()
            .unwrap()
            .to_string()
            .contains("all retained")
    );
    std::fs::write(
        dir.path().join("jaw.obj"),
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 3 2",
    )
    .unwrap();
    c.geometry.exclude_materials = vec![String::new()];
    assert!(
        convert(&c, dir.path())
            .err()
            .unwrap()
            .to_string()
            .contains("topology mismatch")
    );
}
#[test]
fn rejects_degenerate_composite_and_transform_overflow() {
    let (dir, mut c) = common::fixture();
    std::fs::write(
        dir.path().join("jaw.obj"),
        "v 0 0 0\nv 0 0 0\nv 0 1 0\nf 1 2 3",
    )
    .unwrap();
    assert!(convert(&c, dir.path()).is_err());
    c.transform.fit_height_m = f32::MAX;
    assert!(convert(&c, dir.path()).is_err());
}
