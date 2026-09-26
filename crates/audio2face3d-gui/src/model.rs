use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
#[error("invalid head model: {0}")]
pub struct ModelError(pub String);

pub type Result<T> = std::result::Result<T, ModelError>;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    pub schema_version: u32,
    pub rig_profile: String,
    pub generator_version: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Material {
    pub color: [f32; 4],
}

#[derive(Clone, Debug, PartialEq)]
pub struct MorphTarget {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub name: String,
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
    pub material: Material,
    pub targets: Vec<MorphTarget>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HeadModel {
    pub metadata: Metadata,
    pub meshes: Vec<Mesh>,
}

impl HeadModel {
    pub fn validate(&self) -> Result<()> {
        crate::validation::validate(self)
    }

    pub fn unsupported_channels(&self) -> Vec<String> {
        let supported = self.channel_names();
        crate::rig::CHANNELS
            .iter()
            .filter(|name| !supported.iter().any(|s| s == **name))
            .map(|s| (*s).to_owned())
            .collect()
    }

    pub fn channel_names(&self) -> Vec<String> {
        let mut names = std::collections::BTreeSet::new();
        for mesh in &self.meshes {
            names.extend(mesh.targets.iter().map(|target| target.name.clone()));
        }
        names.into_iter().collect()
    }
}

pub struct EvaluatedMesh {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
}

/// CPU reference morph evaluation. Weights correspond to this mesh's target names.
pub fn evaluate(mesh: &Mesh, weights: &[f32]) -> Result<EvaluatedMesh> {
    if weights.len() != mesh.targets.len() || weights.iter().any(|w| !w.is_finite()) {
        return Err(ModelError("invalid morph weights".into()));
    }
    crate::validation::validate_mesh(mesh)?;
    let mut positions = mesh.positions.clone();
    let mut normals = mesh.normals.clone();
    for (target, weight) in mesh.targets.iter().zip(weights) {
        for i in 0..positions.len() {
            for c in 0..3 {
                positions[i][c] += target.positions[i][c] * weight;
                normals[i][c] += target.normals[i][c] * weight;
            }
        }
    }
    for normal in &mut normals {
        let length = normal.iter().map(|x| x * x).sum::<f32>().sqrt();
        *normal = if length > 1e-8 {
            normal.map(|x| x / length)
        } else {
            [0., 0., 1.]
        };
    }
    Ok(EvaluatedMesh { positions, normals })
}
