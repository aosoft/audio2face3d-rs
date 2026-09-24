use crate::{
    HeadModel, ModelError,
    model::Result,
    validation::{MAX_GLB_BYTES, require},
};
use serde_json::{Value, json};

#[derive(Default)]
struct Buffer {
    bytes: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Buffer {
    fn accessor(
        &mut self,
        bytes: &[u8],
        count: usize,
        component: u32,
        kind: &str,
        bounds: Option<([f32; 3], [f32; 3])>,
        target: u32,
    ) -> usize {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
        let view = self.views.len();
        self.views.push(json!({"buffer":0,"byteOffset":self.bytes.len(),"byteLength":bytes.len(),"target":target}));
        self.bytes.extend(bytes);
        let mut accessor =
            json!({"bufferView":view,"componentType":component,"count":count,"type":kind});
        if let Some((min, max)) = bounds {
            accessor["min"] = json!(min);
            accessor["max"] = json!(max);
        }
        let index = self.accessors.len();
        self.accessors.push(accessor);
        index
    }
    fn vectors(&mut self, values: &[[f32; 3]], position: bool) -> usize {
        let bytes: Vec<_> = values
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for v in values {
            for k in 0..3 {
                min[k] = min[k].min(v[k]);
                max[k] = max[k].max(v[k]);
            }
        }
        self.accessor(
            &bytes,
            values.len(),
            5126,
            "VEC3",
            position.then_some((min, max)),
            34962,
        )
    }
}

/// Encode the restricted, self-contained GLB profile deterministically.
pub fn to_glb(model: &HeadModel) -> Result<Vec<u8>> {
    model.validate()?;
    let estimated: usize = model
        .meshes
        .iter()
        .map(|m| m.positions.len() * 24 * (1 + m.targets.len()) + m.indices.len() * 4)
        .sum();
    require(estimated <= MAX_GLB_BYTES, "asset exceeds GLB size limit")?;
    let mut buffer = Buffer::default();
    let mut meshes = Vec::new();
    let mut materials = Vec::new();
    let mut nodes = Vec::new();
    for mesh in &model.meshes {
        let position = buffer.vectors(&mesh.positions, true);
        let normal = buffer.vectors(&mesh.normals, false);
        let small = mesh.indices.iter().all(|&i| i <= u16::MAX as u32);
        let bytes: Vec<_> = if small {
            mesh.indices
                .iter()
                .flat_map(|&i| (i as u16).to_le_bytes())
                .collect()
        } else {
            mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect()
        };
        let indices = buffer.accessor(
            &bytes,
            mesh.indices.len(),
            if small { 5123 } else { 5125 },
            "SCALAR",
            None,
            34963,
        );
        let targets: Vec<_> = mesh.targets.iter().map(|t| json!({"POSITION":buffer.vectors(&t.positions, true),"NORMAL":buffer.vectors(&t.normals, false)})).collect();
        let names: Vec<_> = mesh.targets.iter().map(|t| &t.name).collect();
        materials.push(json!({"pbrMetallicRoughness":{"baseColorFactor":mesh.material.color,"metallicFactor":0.,"roughnessFactor":1.},"doubleSided":true}));
        let mut primitive = json!({"attributes":{"POSITION":position,"NORMAL":normal},"indices":indices,"mode":4,"material":materials.len()-1});
        let mut value = json!({"name":mesh.name,"extras":{"targetNames":names}});
        if !targets.is_empty() {
            primitive["targets"] = json!(targets);
            value["weights"] = json!(vec![0.; mesh.targets.len()]);
        }
        value["primitives"] = json!([primitive]);
        nodes.push(json!({"name":mesh.name,"mesh":meshes.len()}));
        meshes.push(value);
    }
    let document = json!({"asset":{"version":"2.0","generator":"audio2face3d-headgen"},"scene":0,
        "scenes":[{"nodes":(0..nodes.len()).collect::<Vec<_>>()}],"nodes":nodes,"meshes":meshes,"materials":materials,
        "buffers":[{"byteLength":buffer.bytes.len()}],"bufferViews":buffer.views,"accessors":buffer.accessors,
        "extras":{"audio2face3d_preview":model.metadata}});
    let root: gltf_json::Root =
        serde_json::from_value(document).map_err(|e| ModelError(e.to_string()))?;
    use gltf_json::validation::Validate;
    let mut errors = Vec::new();
    root.validate(&root, gltf_json::Path::new, &mut |path, error| {
        errors.push(format!("{}: {error:?}", path()))
    });
    require(errors.is_empty(), &errors.join(", "))?;
    let mut json = serde_json::to_vec(&root).map_err(|e| ModelError(e.to_string()))?;
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    while !buffer.bytes.len().is_multiple_of(4) {
        buffer.bytes.push(0);
    }
    let size = 12 + 8 + json.len() + 8 + buffer.bytes.len();
    require(size <= MAX_GLB_BYTES, "asset exceeds GLB size limit")?;
    let mut bytes = Vec::with_capacity(size);
    for v in [
        0x46546c67_u32,
        2,
        size as u32,
        json.len() as u32,
        0x4e4f534a,
    ] {
        bytes.extend(v.to_le_bytes());
    }
    bytes.extend(json);
    bytes.extend((buffer.bytes.len() as u32).to_le_bytes());
    bytes.extend(0x004e4942_u32.to_le_bytes());
    bytes.extend(buffer.bytes);
    Ok(bytes)
}
