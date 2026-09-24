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
