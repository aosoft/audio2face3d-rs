//! Deterministic, CPU-only development-time debug head generation.

/// Generator version recorded in generated assets.
pub const GENERATOR_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod config;
pub mod error;
mod legacy_config;
mod mesh;
mod morph;
mod normals;
use audio2face3d_gui_core::{HeadModel, Metadata, model::Result, rig};
pub use legacy_config::Config;

/// Deterministic standard debug head with all 52 server channels.
pub fn generate(config: &Config) -> Result<HeadModel> {
    config.validate()?;
    let mut meshes = mesh::parts(config);
    for mesh in &mut meshes {
        mesh.normals = normals::calculate(&mesh.positions, &mesh.indices);
        morph::add_targets(mesh, config);
    }
    let model = HeadModel {
        metadata: Metadata {
            schema_version: 1,
            rig_profile: rig::TESTER.into(),
            generator_version: GENERATOR_VERSION.into(),
        },
        meshes,
    };
    model.validate()?;
    Ok(model)
}
pub mod convert;
pub mod obj;
pub mod report;
pub mod topology;
pub mod transform;
pub use convert::{Conversion, convert, inspect};
