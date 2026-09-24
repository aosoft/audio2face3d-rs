//! Shared, neutral-space topology and vertex remapping.
use crate::{
    config::{Config, SplitBy},
    error::{Error, Result},
    obj::Obj,
    topology,
    transform::Transform,
};
use audio2face3d_gui_core::{Material, Mesh};
use std::collections::{BTreeMap, BTreeSet};
pub struct Part {
    pub mesh: Mesh,
    pub original_vertices: Vec<u32>,
}
pub struct Geometry {
    pub parts: Vec<Part>,
    pub transform: Transform,
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub face_numbers: Vec<usize>,
    pub warnings: Vec<String>,
    pub excluded: BTreeMap<String, usize>,
}
pub fn geometry(config: &Config, neutral: &Obj) -> Result<Geometry> {
    let (triangles, mut warnings) = topology::triangulate(neutral)?;
    let mut groups = BTreeMap::<(String, String, String), Vec<u32>>::new();
    let mut indices = Vec::new();
    let mut face_numbers = Vec::new();
    let mut excluded = BTreeMap::new();
    let mut materials = BTreeSet::new();
    for (face_id, (face, tri)) in neutral.faces.iter().zip(triangles).enumerate() {
        materials.insert(face.material.clone());
        if config.geometry.exclude_materials.contains(&face.material) {
            *excluded.entry(face.material.clone()).or_insert(0) += 1;
            continue;
        }
        let key = match config.geometry.split_by {
            SplitBy::Material => (String::new(), String::new(), face.material.clone()),
            SplitBy::ObjectGroupMaterial => (
                face.object.clone(),
                face.group.clone(),
                face.material.clone(),
            ),
        };
        face_numbers.extend(std::iter::repeat_n(face_id + 1, tri.len() / 3));
        indices.extend_from_slice(&tri);
        groups.entry(key).or_default().extend(tri);
    }
    if groups.is_empty() || groups.len() > 64 {
        return Err(Error::Output("retained mesh count must be 1..64".into()));
    }
    for name in config
        .geometry
        .exclude_materials
        .iter()
        .chain(config.materials.colors.keys())
    {
        if !materials.contains(name) {
            warnings.push(format!("configured material not found: {name}"));
        }
    }
    for name in &materials {
        if !config.materials.colors.contains_key(name) && !excluded.contains_key(name) {
            warnings.push(format!("default color used for material: {name}"));
        }
    }
    let transform = Transform::new(
        &config.transform,
        indices.iter().map(|&i| neutral.positions[i as usize]),
    )?;
    let positions = neutral
        .positions
        .iter()
        .map(|&p| transform.position(p))
        .collect::<Vec<_>>();
    if positions.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::Input("nonfinite transformed neutral".into()));
    }
    let normals = crate::normals::checked(&positions, &indices, "neutral")?;
    let mut parts = Vec::new();
    let mut vertices = 0usize;
    for ((object, group, material), indices) in groups {
        let original_vertices = indices
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        vertices = vertices
            .checked_add(original_vertices.len())
            .ok_or_else(|| Error::Output("vertex count overflow".into()))?;
        if vertices > 100_000 {
            return Err(Error::Output("split vertex limit exceeded".into()));
        }
        let remap = original_vertices
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i as u32))
            .collect::<BTreeMap<_, _>>();
        let mesh = Mesh {
            name: format!("{object}/{group}/{material}"),
            positions: original_vertices
                .iter()
                .map(|&i| positions[i as usize])
                .collect(),
            normals: original_vertices
                .iter()
                .map(|&i| normals[i as usize])
                .collect(),
            indices: indices.iter().map(|i| remap[i]).collect(),
            material: Material {
                color: config
                    .materials
                    .colors
                    .get(&material)
                    .copied()
                    .unwrap_or(config.materials.default_color),
            },
            targets: vec![],
        };
        parts.push(Part {
            mesh,
            original_vertices,
        });
    }
    Ok(Geometry {
        parts,
        transform,
        positions,
        indices,
        face_numbers,
        warnings,
        excluded,
    })
}

pub struct Conversion {
    pub model: audio2face3d_gui_core::HeadModel,
    pub report: crate::report::Report,
}
/// Complete all validation before returning a model. No files are written here.
pub fn convert(config: &Config, input_root: &std::path::Path) -> Result<Conversion> {
    use crate::{
        obj,
        report::{self, Channel, Input, Report},
        transform::{cross, dot, sub},
    };
    use audio2face3d_gui_core::{HeadModel, Metadata, MorphTarget};
    let resolved = obj::resolve_inputs(config, input_root)?;
    let neutral = obj::read(&resolved[0].1)?;
    let mut inputs = vec![Input::new(config.neutral.clone(), &neutral)];
    if let Some(e) = &config.expected
        && (neutral.positions.len() != e.source_vertices || neutral.faces.len() != e.source_faces)
    {
        return Err(Error::Input(format!(
            "neutral counts: got {} vertices / {} faces, expected {} / {}",
            neutral.positions.len(),
            neutral.faces.len(),
            e.source_vertices,
            e.source_faces
        )));
    }
    let paths = resolved
        .iter()
        .map(|(name, path)| (name.as_str(), path))
        .collect::<BTreeMap<_, _>>();
    let mut total = neutral.bytes;
    // Validate the entire original topology, including excluded faces, first.
    for (name, path) in resolved.iter().skip(1) {
        let expression = obj::read(path)?;
        topology::validate(&neutral, &expression, name)?;
        total = total
            .checked_add(expression.bytes)
            .ok_or_else(|| Error::Output("input size overflow".into()))?;
        if total > obj::MAX_TOTAL_BYTES {
            return Err(Error::Output("total input bytes exceeded".into()));
        }
        inputs.push(Input::new(name.clone(), &expression));
    }
    let hashes = inputs
        .iter()
        .map(|i| (i.path.as_str(), i.sha256.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut g = geometry(config, &neutral)?;
    let neutral_normals = crate::normals::checked(&g.positions, &g.indices, "neutral")?;
    let retained = g.indices.iter().copied().collect::<BTreeSet<_>>();
    let mut channels = BTreeMap::new();
    let mut decoded_bytes = g
        .parts
        .iter()
        .map(|p| p.mesh.positions.len() * 24 + p.mesh.indices.len() * 4)
        .sum::<usize>();
    for (name, sources) in &config.targets {
        let mut delta = vec![[0.; 3]; neutral.positions.len()];
        for source in sources {
            let expression = obj::read(paths[source.as_str()])?;
            if expression.sha256 != hashes[source.as_str()] {
                return Err(Error::Input(format!(
                    "{source}: input changed during conversion"
                )));
            }
            for (i, p) in expression.positions.iter().enumerate() {
                let d = g.transform.delta(sub(*p, neutral.positions[i]));
                for c in 0..3 {
                    delta[i][c] += d[c];
                }
            }
        }
        let posed = g
            .positions
            .iter()
            .zip(&delta)
            .map(|(p, d)| std::array::from_fn(|i| p[i] + d[i]))
            .collect::<Vec<[f32; 3]>>();
        if posed.iter().flatten().any(|x| !x.is_finite()) {
            return Err(Error::Input(format!("{name}: position overflow")));
        }
        let normal_result = crate::normals::posed(
            &posed,
            &g.indices,
            name,
            &g.face_numbers,
            config.geometry.degenerate_pose_triangles,
        )?;
        let posed_normals = normal_result.normals;
        let skipped_normal_triangles = normal_result.skipped_triangles;
        let skipped_normal_source_faces = skipped_normal_triangles
            .iter()
            .map(|i| g.face_numbers[i - 1])
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        if !skipped_normal_triangles.is_empty() {
            g.warnings.push(format!("{name}: skipped {} zero-area triangle normal contributions at source faces {:?}; geometry unchanged", skipped_normal_triangles.len(), skipped_normal_source_faces));
        }
        let mut reversed = Vec::new();
        for (i, t) in g.indices.chunks_exact(3).enumerate() {
            let normal = |p: &[[f32; 3]]| {
                cross(
                    sub(p[t[1] as usize], p[t[0] as usize]),
                    sub(p[t[2] as usize], p[t[0] as usize]),
                )
            };
            if dot(normal(&g.positions), normal(&posed)) < 0. {
                reversed.push(g.face_numbers[i]);
            }
        }
        reversed.sort_unstable();
        reversed.dedup();
        if !reversed.is_empty() {
            g.warnings.push(format!("{name}: {} source faces have normal reversal candidates; inspect large rotations visually",reversed.len()));
        }
        let mut max = 0f64;
        let mut sum = 0f64;
        let mut nonzero = 0usize;
        for &i in &retained {
            let d = delta[i as usize];
            let squared = d.iter().map(|&x| (x as f64).powi(2)).sum::<f64>();
            max = max.max(squared.sqrt());
            sum += squared;
            if d.iter().any(|x| x.abs() > 1e-8) {
                nonzero += 1;
            }
        }
        if nonzero == 0 {
            return Err(Error::Input(format!(
                "{name}: all retained position deltas are zero"
            )));
        }
        for part in &mut g.parts {
            if !part
                .original_vertices
                .iter()
                .any(|&i| delta[i as usize].iter().any(|x| x.abs() > 1e-8))
            {
                continue;
            }
            decoded_bytes = decoded_bytes
                .checked_add(part.original_vertices.len() * 24)
                .ok_or_else(|| Error::Output("decoded size overflow".into()))?;
            if decoded_bytes > audio2face3d_gui_core::validation::MAX_GLB_BYTES {
                return Err(Error::Output("decoded morph data exceeds 64 MiB".into()));
            }
            part.mesh.targets.push(MorphTarget {
                name: name.clone(),
                positions: part
                    .original_vertices
                    .iter()
                    .map(|&i| delta[i as usize])
                    .collect(),
                normals: part
                    .original_vertices
                    .iter()
                    .map(|&i| sub(posed_normals[i as usize], neutral_normals[i as usize]))
                    .collect(),
            });
        }
        channels.insert(
            name.clone(),
            Channel {
                skipped_normal_triangles,
                skipped_normal_source_faces,
                sources: sources.clone(),
                max_displacement: max,
                rms_displacement: (sum / retained.len() as f64).sqrt(),
                nonzero_vertices: nonzero,
                reversed_triangle_candidates: reversed,
            },
        );
    }
    let model = HeadModel {
        metadata: Metadata {
            schema_version: 1,
            rig_profile: config.output_profile.clone(),
            generator_version: crate::GENERATOR_VERSION.into(),
        },
        meshes: g.parts.into_iter().map(|p| p.mesh).collect(),
    };
    model.validate().map_err(|e| Error::Input(e.to_string()))?;
    let output_vertices = model.meshes.iter().map(|m| m.positions.len()).sum();
    let output_triangles = model.meshes.iter().map(|m| m.indices.len() / 3).sum();
    let output_morph_targets = model.meshes.iter().map(|m| m.targets.len()).sum::<usize>();
    // vec4 positions/normals; base, morph storage, evaluated vertices, and indices.
    let gpu_geometry_bytes = model
        .meshes
        .iter()
        .map(|m| m.positions.len() * 32 * (2 + m.targets.len()) + m.indices.len() * 4)
        .sum();
    let largest_gpu_storage_buffer_bytes = model
        .meshes
        .iter()
        .map(|m| m.positions.len() * 32 * m.targets.len().max(1))
        .max()
        .unwrap_or(0);
    let glb_upper_bound_bytes = audio2face3d_gui_core::gltf::encoded_size(&model)
        .map_err(|e| Error::Output(e.to_string()))?;
    let effective = serde_json::to_vec(config).map_err(|e| Error::Config(e.to_string()))?;
    let report = Report {
        schema_version: 1,
        generator_version: crate::GENERATOR_VERSION.into(),
        config_sha256: config.source_sha256.clone(),
        effective_config_sha256: report::hash(&effective),
        inputs,
        transform: g.transform,
        excluded_material_faces: g.excluded,
        output_vertices,
        output_triangles,
        output_meshes: model.meshes.len(),
        output_morph_targets,
        channels,
        unsupported_channels: model.unsupported_channels(),
        decoded_bytes,
        gpu_geometry_bytes,
        largest_gpu_storage_buffer_bytes,
        glb_upper_bound_bytes,
        output_glb_sha256: None,
        output_glb_bytes: None,
        warnings: g.warnings,
        reference: config.reference.clone(),
    };
    Ok(Conversion { model, report })
}
pub fn inspect(config: &Config, input_root: &std::path::Path) -> Result<crate::report::Report> {
    Ok(convert(config, input_root)?.report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn material_split_keeps_shared_normals_and_exclusion_is_explicit() {
        let obj = crate::obj::parse(
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nusemtl a\nf 1 2 3\nusemtl b\nf 1 4 2\n"
                .as_slice(),
            "fixture",
        )
        .unwrap();
        let mut c = Config::parse(include_str!("../presets/ict-facekit.toml")).unwrap();
        let g = geometry(&c, &obj).unwrap();
        assert_eq!(g.parts.len(), 2);
        assert_eq!(g.parts[0].mesh.normals[0], g.parts[1].mesh.normals[0]);
        assert_eq!(
            g.parts
                .iter()
                .map(|p| p.mesh.positions.len())
                .sum::<usize>(),
            6
        );
        c.geometry.exclude_materials = vec!["b".into()];
        let g = geometry(&c, &obj).unwrap();
        assert_eq!(g.parts.len(), 1);
        assert_eq!(g.excluded["b"], 1);
        c.geometry.exclude_materials.push("a".into());
        assert!(geometry(&c, &obj).is_err());
    }
}
