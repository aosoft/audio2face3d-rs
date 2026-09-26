use crate::{
    HeadModel, Material, Mesh, Metadata, ModelError, MorphTarget,
    model::Result,
    validation::{MAX_GLB_BYTES, MAX_INDICES, MAX_VERTICES, require},
};
use gltf::{
    accessor::{DataType, Dimensions},
    mesh::Semantic,
};
use serde_json::Value;

fn error(e: impl std::fmt::Display) -> ModelError {
    ModelError(e.to_string())
}

/// Decode GLB bytes only. File loading and path policy belong to the host.
pub fn from_glb(bytes: &[u8]) -> Result<HeadModel> {
    require(
        bytes.len() <= MAX_GLB_BYTES && bytes.len() >= 20,
        "invalid GLB size",
    )?;
    require(bytes.starts_with(b"glTF"), "GLB required")?;
    require(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize == bytes.len(),
        "GLB length mismatch",
    )?;
    let glb = gltf::binary::Glb::from_slice(bytes).map_err(error)?;
    let raw: Value = serde_json::from_slice(&glb.json).map_err(error)?;
    reject_unsupported(&raw)?;
    let metadata: Metadata =
        serde_json::from_value(raw["extras"]["audio2face3d_preview"].clone()).map_err(error)?;
    let parsed = gltf::Gltf::from_slice(bytes).map_err(error)?;
    let document = &parsed.document;
    let blob = parsed
        .blob
        .as_deref()
        .ok_or_else(|| error("missing BIN buffer"))?;
    require(
        document.buffers().len() == 1 && document.meshes().len() <= 64,
        "invalid buffer or mesh count",
    )?;
    let buffer = document.buffers().next().unwrap();
    require(
        matches!(buffer.source(), gltf::buffer::Source::Bin) && buffer.length() <= blob.len(),
        "external or truncated buffer",
    )?;
    for view in document.views() {
        require(
            view.offset()
                .checked_add(view.length())
                .is_some_and(|end| end <= buffer.length()),
            "buffer view out of bounds",
        )?;
    }
    for accessor in document.accessors() {
        require(
            accessor.sparse().is_none() && !accessor.normalized(),
            "sparse/normalized accessor unsupported",
        )?;
        let view = accessor
            .view()
            .ok_or_else(|| error("accessor requires buffer view"))?;
        let stride = view.stride().unwrap_or(accessor.size());
        require(
            accessor.count() > 0 && accessor.count() <= MAX_INDICES && stride >= accessor.size(),
            "invalid accessor size/stride",
        )?;
        let end = accessor
            .count()
            .checked_sub(1)
            .and_then(|n| n.checked_mul(stride))
            .and_then(|n| n.checked_add(accessor.offset()))
            .and_then(|n| n.checked_add(accessor.size()));
        require(
            end.is_some_and(|end| end <= view.length()),
            "accessor out of bounds",
        )?;
    }
    let scene = document
        .default_scene()
        .ok_or_else(|| error("default scene required"))?;
    require(
        document.scenes().len() == 1
            && scene.nodes().count() == document.nodes().len()
            && document.nodes().len() == document.meshes().len(),
        "independent scene roots required",
    )?;
    let mut used = std::collections::BTreeSet::new();
    for node in scene.nodes() {
        let mesh = node
            .mesh()
            .ok_or_else(|| error("node must reference mesh"))?;
        require(used.insert(mesh.index()), "instanced mesh unsupported")?;
    }
    let mut meshes = Vec::new();
    let mut total_vertices = 0;
    let mut total_indices = 0;
    let mut decoded_bytes = 0usize;
    for mesh in document.meshes() {
        require(
            mesh.primitives().len() == 1,
            "one primitive per mesh required",
        )?;
        let primitive = mesh.primitives().next().unwrap();
        require(
            primitive.mode() == gltf::mesh::Mode::Triangles,
            "triangle mesh required",
        )?;
        require(
            primitive.attributes().count() == 2,
            "only POSITION and NORMAL supported",
        )?;
        let position = primitive
            .get(&Semantic::Positions)
            .ok_or_else(|| error("missing positions"))?;
        let normal = primitive
            .get(&Semantic::Normals)
            .ok_or_else(|| error("missing normals"))?;
        vector(&position)?;
        vector(&normal)?;
        require(
            normal.count() == position.count(),
            "base normal count mismatch",
        )?;
        total_vertices += position.count();
        require(total_vertices <= MAX_VERTICES, "vertex limit exceeded")?;
        let indices = primitive
            .indices()
            .ok_or_else(|| error("indices required"))?;
        require(
            indices.dimensions() == Dimensions::Scalar
                && matches!(indices.data_type(), DataType::U16 | DataType::U32),
            "indices must be u16/u32",
        )?;
        total_indices += indices.count();
        require(total_indices <= MAX_INDICES, "index limit exceeded")?;
        let extras: Value = serde_json::from_str(
            mesh.extras()
                .as_ref()
                .ok_or_else(|| error("targetNames required"))?
                .get(),
        )
        .map_err(error)?;
        let names: Vec<String> =
            serde_json::from_value(extras["targetNames"].clone()).map_err(error)?;
        require(
            names.len() == primitive.morph_targets().len() && names.len() <= 52,
            "targetNames count mismatch",
        )?;
        if let Some(weights) = mesh.weights() {
            require(
                weights.len() == names.len() && weights.iter().all(|&x| x == 0.),
                "initial weights must be zero",
            )?;
        }
        decoded_bytes += position.count() * 24 * (1 + names.len()) + indices.count() * 4;
        require(
            decoded_bytes <= MAX_GLB_BYTES,
            "decoded model exceeds memory limit",
        )?;
        for target in primitive.morph_targets() {
            let p = target
                .positions()
                .ok_or_else(|| error("target position deltas required"))?;
            let n = target
                .normals()
                .ok_or_else(|| error("target normal deltas required"))?;
            vector(&p)?;
            vector(&n)?;
            require(
                p.count() == position.count() && n.count() == position.count(),
                "target count mismatch",
            )?;
            require(target.tangents().is_none(), "tangent deltas unsupported")?;
        }
        let reader = primitive.reader(|_| Some(blob));
        let positions = reader
            .read_positions()
            .ok_or_else(|| error("unreadable positions"))?
            .collect();
        let normals = reader
            .read_normals()
            .ok_or_else(|| error("unreadable normals"))?
            .collect();
        let indices = reader
            .read_indices()
            .ok_or_else(|| error("unreadable indices"))?
            .into_u32()
            .collect();
        let targets = reader
            .read_morph_targets()
            .zip(names)
            .map(|((p, n, _), name)| {
                Ok(MorphTarget {
                    name,
                    positions: p.ok_or_else(|| error("unreadable target"))?.collect(),
                    normals: n
                        .ok_or_else(|| error("unreadable target normals"))?
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        meshes.push(Mesh {
            name: mesh.name().unwrap_or("mesh").into(),
            positions,
            normals,
            indices,
            targets,
            material: Material {
                color: primitive
                    .material()
                    .pbr_metallic_roughness()
                    .base_color_factor(),
            },
        });
    }
    let model = HeadModel { metadata, meshes };
    model.validate()?;
    Ok(model)
}

fn vector(accessor: &gltf::Accessor<'_>) -> Result<()> {
    require(
        accessor.dimensions() == Dimensions::Vec3 && accessor.data_type() == DataType::F32,
        "FLOAT VEC3 required",
    )
}

fn reject_unsupported(raw: &Value) -> Result<()> {
    fn extensions(value: &Value) -> bool {
        match value {
            Value::Object(o) => {
                o.contains_key("extensions")
                    || o.iter().any(|(k, v)| k != "extras" && extensions(v))
            }
            Value::Array(a) => a.iter().any(extensions),
            _ => false,
        }
    }
    require(!extensions(raw), "extensions unsupported")?;
    for key in [
        "skins",
        "animations",
        "cameras",
        "images",
        "textures",
        "extensionsRequired",
        "extensionsUsed",
    ] {
        require(
            raw.get(key)
                .is_none_or(|v| v.as_array().is_some_and(|a| a.is_empty())),
            &format!("unsupported {key}"),
        )?;
    }
    for node in raw["nodes"]
        .as_array()
        .ok_or_else(|| error("nodes required"))?
    {
        for key in [
            "children",
            "matrix",
            "translation",
            "rotation",
            "scale",
            "skin",
            "camera",
            "weights",
            "extensions",
        ] {
            require(node.get(key).is_none(), &format!("unsupported node {key}"))?;
        }
    }
    if let Some(materials) = raw["materials"].as_array() {
        for material in materials {
            require(
                material["alphaMode"].as_str().unwrap_or("OPAQUE") == "OPAQUE",
                "opaque materials only",
            )?;
            for key in [
                "normalTexture",
                "occlusionTexture",
                "emissiveTexture",
                "extensions",
            ] {
                require(material.get(key).is_none(), "unsupported material property")?;
            }
            let pbr = &material["pbrMetallicRoughness"];
            require(
                pbr["metallicFactor"].as_f64() == Some(0.)
                    && pbr["roughnessFactor"].as_f64().unwrap_or(1.) == 1.,
                "plain nonmetallic material required",
            )?;
            require(
                pbr.get("baseColorTexture").is_none()
                    && pbr.get("metallicRoughnessTexture").is_none(),
                "textures unsupported",
            )?;
            require(
                material["emissiveFactor"]
                    .as_array()
                    .is_none_or(|a| a.iter().all(|v| v.as_f64() == Some(0.))),
                "emission unsupported",
            )?;
        }
    }
    Ok(())
}
