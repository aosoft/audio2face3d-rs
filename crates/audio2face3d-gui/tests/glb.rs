#![cfg(all(feature = "gltf-read", feature = "gltf-write"))]

use audio2face3d_gui::{
    gltf::{from_glb, to_glb},
    model::evaluate,
};
use serde_json::{Value, json};

// An independent tiny asset: interleaved position/normal, u16 indices and two
// distinct targets. Does not call the production writer to build the fixture.
fn fixture() -> (Value, Vec<u8>) {
    let mut bin = Vec::new();
    for v in [
        [0f32, 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 1.],
    ] {
        for x in v {
            bin.extend(x.to_le_bytes());
        }
    }
    for i in [0u16, 1, 2, 0] {
        bin.extend(i.to_le_bytes());
    }
    for delta in [[0f32, 0., 1.], [0., 1., 0.], [0., 0., 0.]] {
        for _ in 0..3 {
            for x in delta {
                bin.extend(x.to_le_bytes());
            }
        }
    }
    let doc = json!({"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],
        "buffers":[{"byteLength":188}],"bufferViews":[
            {"buffer":0,"byteOffset":0,"byteLength":72,"byteStride":24,"target":34962},
            {"buffer":0,"byteOffset":72,"byteLength":6,"target":34963},
            {"buffer":0,"byteOffset":80,"byteLength":108,"target":34962}],
        "accessors":[
            {"bufferView":0,"byteOffset":0,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,0],"max":[1,1,0]},
            {"bufferView":0,"byteOffset":12,"componentType":5126,"count":3,"type":"VEC3"},
            {"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"},
            {"bufferView":2,"componentType":5126,"count":3,"type":"VEC3","min":[0,0,1],"max":[0,0,1]},
            {"bufferView":2,"byteOffset":36,"componentType":5126,"count":3,"type":"VEC3","min":[0,1,0],"max":[0,1,0]},
            {"bufferView":2,"byteOffset":72,"componentType":5126,"count":3,"type":"VEC3"}],
        "materials":[{"pbrMetallicRoughness":{"baseColorFactor":[0.7,0.5,0.4,1.],"metallicFactor":0.,"roughnessFactor":1.},"doubleSided":true}],
        "meshes":[{"name":"triangle","weights":[0,0],"extras":{"targetNames":["JawForward","JawOpen"]},"primitives":[{
            "attributes":{"POSITION":0,"NORMAL":1},"indices":2,"material":0,"targets":[{"POSITION":3,"NORMAL":5},{"POSITION":4,"NORMAL":5}]}]}],
        "extras":{"audio2face3d_preview":{"schema_version":1,"rig_profile":"audio2face_rs_tester_v1","generator_version":"fixture"}}});
    (doc, bin)
}

fn pack(doc: &Value, bin: &[u8]) -> Vec<u8> {
    let mut json = serde_json::to_vec(doc).unwrap();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let mut data = Vec::new();
    for v in [
        0x46546c67u32,
        2,
        (28 + json.len() + bin.len()) as u32,
        json.len() as u32,
        0x4e4f534a,
    ] {
        data.extend(v.to_le_bytes());
    }
    data.extend(json);
    data.extend((bin.len() as u32).to_le_bytes());
    data.extend(0x004e4942u32.to_le_bytes());
    data.extend(bin);
    data
}

#[test]
fn reads_independent_interleaved_fixture_and_round_trips() {
    let (doc, bin) = fixture();
    let model = from_glb(&pack(&doc, &bin)).unwrap();
    assert_eq!(model.meshes[0].positions[1], [1., 0., 0.]);
    assert_eq!(model.meshes[0].normals, vec![[0., 0., 1.]; 3]);
    let evaluated = evaluate(&model.meshes[0], &[0.5, 0.25]).unwrap();
    assert_eq!(evaluated.positions[0], [0., 0.25, 0.5]);
    let bytes = to_glb(&model).unwrap();
    assert_eq!(from_glb(&bytes).unwrap(), model);
    assert_eq!(to_glb(&model).unwrap(), bytes);
    if let Ok(path) = std::env::var("A2F_TEST_GLB_OUTPUT") {
        std::fs::write(path, bytes).unwrap();
    }
}

#[test]
fn rejects_corrupt_or_unsupported_inputs_without_panics() {
    let (doc, bin) = fixture();
    let cases = [
        ("/accessors/0/count", json!(999999999)),
        ("/accessors/3/count", json!(4)),
        ("/accessors/0/byteOffset", json!(9999)),
        ("/buffers/0/byteLength", json!(9999)),
        ("/bufferViews/2/byteLength", json!(10)),
        (
            "/meshes/0/extras/targetNames",
            json!(["duplicate", "duplicate"]),
        ),
        ("/meshes/0/extras/targetNames", json!(["one"])),
        ("/extras/audio2face3d_preview/schema_version", json!(2)),
        ("/extras/audio2face3d_preview/rig_profile", json!("unknown")),
        ("/meshes/0/weights", json!([1, 0])),
        ("/meshes/0/primitives/0/mode", json!(1)),
    ];
    for (pointer, value) in cases {
        let mut bad = doc.clone();
        if pointer.ends_with("/mode") {
            bad["meshes"][0]["primitives"][0]["mode"] = value;
        } else {
            *bad.pointer_mut(pointer).unwrap() = value;
        }
        assert!(from_glb(&pack(&bad, &bin)).is_err(), "{pointer}");
    }
    let mut bad = bin.clone();
    bad[72..74].copy_from_slice(&999u16.to_le_bytes());
    assert!(from_glb(&pack(&doc, &bad)).is_err());
    let mut bad = bin.clone();
    bad[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(from_glb(&pack(&doc, &bad)).is_err());
    let complete = pack(&doc, &bin);
    for n in 0..complete.len() {
        assert!(from_glb(&complete[..n]).is_err());
    }
}

#[test]
fn shared_names_across_meshes_are_valid_but_false_full_profile_is_not() {
    let (doc, bin) = fixture();
    let mut model = from_glb(&pack(&doc, &bin)).unwrap();
    model.meshes.push(model.meshes[0].clone());
    assert!(from_glb(&to_glb(&model).unwrap()).is_ok());
    model.metadata.rig_profile = "debug_face_52_v1".into();
    assert!(model.validate().is_err());
}

#[test]
fn tester_profile_rejects_legacy_unknown_and_zero_targets() {
    let (doc, bin) = fixture();
    for profile in ["debug_face_prototype_v1", "debug_face_52_v1"] {
        let mut bad = doc.clone();
        bad["extras"]["audio2face3d_preview"]["rig_profile"] = json!(profile);
        assert!(from_glb(&pack(&bad, &bin)).is_err());
    }
    let mut model = from_glb(&pack(&doc, &bin)).unwrap();
    assert_eq!(model.unsupported_channels().len(), 50);
    model.meshes[0].targets[0].positions.fill([0.; 3]);
    assert!(model.validate().is_err());
    model.meshes[0].targets[0].name = "Unknown".into();
    assert!(model.validate().is_err());
}

#[test]
fn original_full_asset_uses_the_same_tester_contract() {
    let model = from_glb(include_bytes!(
        "../../audio2face3d-gui/assets/default-head.glb"
    ))
    .unwrap();
    assert_eq!(model.channel_names().len(), 52);
    assert!(model.unsupported_channels().is_empty());
    let bytes = to_glb(&model).unwrap();
    assert_eq!(from_glb(&bytes).unwrap(), model);
    assert_eq!(
        audio2face3d_gui::gltf::encoded_size(&model).unwrap(),
        bytes.len()
    );
}
