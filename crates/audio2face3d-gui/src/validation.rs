use crate::{HeadModel, Mesh, ModelError, model::Result, rig};
use std::collections::BTreeSet;

pub const MAX_GLB_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_VERTICES: usize = 100_000;
pub const MAX_INDICES: usize = 600_000;

pub(crate) fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(ModelError(message.into()))
    }
}

pub fn validate(model: &HeadModel) -> Result<()> {
    require(
        model.metadata.schema_version == crate::SCHEMA_VERSION,
        "unsupported schema version",
    )?;
    require(
        model.metadata.rig_profile == rig::TESTER,
        "unsupported rig profile",
    )?;
    require(
        !model.metadata.generator_version.trim().is_empty(),
        "missing generator version",
    )?;
    require(
        !model.meshes.is_empty() && model.meshes.len() <= 64,
        "mesh count exceeds profile",
    )?;
    require(
        model
            .meshes
            .iter()
            .map(|m| m.positions.len())
            .sum::<usize>()
            <= MAX_VERTICES,
        "vertex limit exceeded",
    )?;
    require(
        model.meshes.iter().map(|m| m.indices.len()).sum::<usize>() <= MAX_INDICES,
        "index limit exceeded",
    )?;
    let mut names = BTreeSet::new();
    let mut active = BTreeSet::new();
    for mesh in &model.meshes {
        validate_mesh(mesh)?;
        for target in &mesh.targets {
            names.insert(target.name.as_str());
            if target.positions.iter().flatten().any(|x| x.abs() > 1e-8) {
                active.insert(target.name.as_str());
            }
        }
    }
    let expected = rig::CHANNELS.into_iter().collect::<BTreeSet<_>>();
    require(
        !names.is_empty() && names.is_subset(&expected) && active == names,
        "tester rig requires 1..52 canonical, nonzero targets",
    )?;
    Ok(())
}

pub fn validate_mesh(mesh: &Mesh) -> Result<()> {
    let n = mesh.positions.len();
    require(n > 0 && n <= MAX_VERTICES, "invalid vertex count")?;
    require(mesh.normals.len() == n, "normal count mismatch")?;
    require(
        !mesh.indices.is_empty()
            && mesh.indices.len().is_multiple_of(3)
            && mesh.indices.len() <= MAX_INDICES,
        "invalid triangle index count",
    )?;
    require(
        mesh.indices.iter().all(|&i| (i as usize) < n),
        "index outside vertex array",
    )?;
    require(
        mesh.positions
            .iter()
            .chain(&mesh.normals)
            .flatten()
            .all(|x| x.is_finite()),
        "nonfinite vertex",
    )?;
    require(
        mesh.normals
            .iter()
            .all(|v| v.iter().map(|x| x * x).sum::<f32>() > 1e-12),
        "zero base normal",
    )?;
    require(
        mesh.material
            .color
            .iter()
            .all(|x| x.is_finite() && (0.0..=1.0).contains(x))
            && mesh.material.color[3] == 1.,
        "invalid opaque material",
    )?;
    require(mesh.targets.len() <= 52, "too many morph targets")?;
    let mut names = BTreeSet::new();
    for target in &mesh.targets {
        require(
            !target.name.trim().is_empty() && names.insert(&target.name),
            "empty or duplicate target name",
        )?;
        require(
            target.positions.len() == n && target.normals.len() == n,
            "target vertex count mismatch",
        )?;
        require(
            target
                .positions
                .iter()
                .chain(&target.normals)
                .flatten()
                .all(|x| x.is_finite()),
            "nonfinite target delta",
        )?;
    }
    Ok(())
}
