use serde::Serialize;
use std::collections::BTreeMap;
#[derive(Debug, Serialize)]
pub struct Input {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub vertices: usize,
    pub faces: usize,
    pub texture_coordinates: usize,
    pub supplied_normals: usize,
    pub material_libraries: Vec<String>,
}
impl Input {
    pub(crate) fn new(path: String, obj: &crate::obj::Obj) -> Self {
        Self {
            path,
            sha256: obj.sha256.clone(),
            bytes: obj.bytes,
            vertices: obj.positions.len(),
            faces: obj.faces.len(),
            texture_coordinates: obj.texture_coordinates,
            supplied_normals: obj.supplied_normals,
            material_libraries: obj.material_libraries.clone(),
        }
    }
}
#[derive(Debug, Serialize)]
pub struct Channel {
    pub skipped_normal_triangles: Vec<usize>,
    pub skipped_normal_source_faces: Vec<usize>,
    pub sources: Vec<String>,
    pub max_displacement: f64,
    pub rms_displacement: f64,
    pub nonzero_vertices: usize,
    pub reversed_triangle_candidates: Vec<usize>,
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub generator_version: String,
    pub config_sha256: String,
    pub effective_config_sha256: String,
    pub inputs: Vec<Input>,
    pub transform: crate::transform::Transform,
    pub excluded_material_faces: BTreeMap<String, usize>,
    pub output_vertices: usize,
    pub output_triangles: usize,
    pub output_meshes: usize,
    pub output_morph_targets: usize,
    pub channels: BTreeMap<String, Channel>,
    pub unsupported_channels: Vec<String>,
    pub decoded_bytes: usize,
    pub gpu_geometry_bytes: usize,
    pub largest_gpu_storage_buffer_bytes: usize,
    pub glb_upper_bound_bytes: usize,
    pub output_glb_sha256: Option<String>,
    pub output_glb_bytes: Option<usize>,
    pub warnings: Vec<String>,
    pub reference: Option<crate::config::Reference>,
}
pub fn hash(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
