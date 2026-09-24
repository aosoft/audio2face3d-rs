use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
#[command(version, about = "Generate an original MIT-licensed debug head GLB")]
struct Arguments {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Arguments::parse();
    let config = if let Some(path) = args.config {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        audio2face3d_headgen::Config::default()
    };
    let model = audio2face3d_headgen::generate(&config)?;
    let bytes = audio2face3d_gui_core::gltf::to_glb(&model)?;
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&args.output, &bytes)?;
    println!(
        "{}: {} vertices (head {}), {} triangles, {} channels, {} bytes, {}",
        args.output.display(),
        model
            .meshes
            .iter()
            .map(|m| m.positions.len())
            .sum::<usize>(),
        model.meshes[0].positions.len(),
        model
            .meshes
            .iter()
            .map(|m| m.indices.len() / 3)
            .sum::<usize>(),
        model.channel_names().len(),
        bytes.len(),
        model.metadata.rig_profile
    );
    Ok(())
}
