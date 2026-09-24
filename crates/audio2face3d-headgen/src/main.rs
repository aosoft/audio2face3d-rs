use audio2face3d_headgen::{
    config::Config,
    error::{Error, Result},
    report::{self, Report},
};
use clap::{Args, Parser, Subcommand};
use std::{
    io::Write,
    path::{Path, PathBuf},
};
#[derive(Parser)]
#[command(
    version,
    about = "Convert matching neutral and expression OBJ meshes into a preview GLB"
)]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}
#[derive(Args)]
struct Inputs {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    input_root: PathBuf,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long)]
    force: bool,
}
#[derive(Subcommand)]
enum Command {
    /// Validate all inputs, composed shapes and exact output size, without writing a GLB.
    Inspect {
        #[command(flatten)]
        input: Inputs,
    },
    /// Write a self-contained GLB and its conversion report.
    Convert {
        #[command(flatten)]
        input: Inputs,
        #[arg(long)]
        output: PathBuf,
    },
}
fn io_error(path: &Path, e: impl std::fmt::Display) -> Error {
    Error::Output(format!("{}: {e}", path.display()))
}
fn destination(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        return path.canonicalize().map_err(|e| io_error(path, e));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(parent
        .canonicalize()
        .map_err(|e| io_error(parent, e))?
        .join(
            path.file_name()
                .ok_or_else(|| io_error(path, "missing file name"))?,
        ))
}
fn validate_outputs(
    outputs: &[&Path],
    inputs: &[(String, PathBuf)],
    config: &Path,
    force: bool,
) -> Result<()> {
    let mut destinations = std::collections::BTreeSet::new();
    let config = config.canonicalize().map_err(|e| io_error(config, e))?;
    for path in outputs {
        let resolved = destination(path)?;
        if !destinations.insert(resolved.clone())
            || resolved == config
            || inputs.iter().any(|(_, p)| *p == resolved)
        {
            return Err(io_error(
                path,
                "output collides with an input or another output",
            ));
        }
        if path.exists() && (!force || !path.is_file()) {
            return Err(io_error(
                path,
                "output exists (use --force for a regular file)",
            ));
        }
        if std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink()) {
            return Err(io_error(path, "refusing a symlink output"));
        }
    }
    Ok(())
}
fn stage(path: &Path, bytes: &[u8]) -> Result<tempfile::NamedTempFile> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(|e| io_error(path, e))?;
    file.write_all(bytes).map_err(|e| io_error(path, e))?;
    file.as_file().sync_all().map_err(|e| io_error(path, e))?;
    Ok(file)
}
fn commit(file: tempfile::NamedTempFile, path: &Path, force: bool) -> Result<()> {
    let result = if force {
        file.persist(path)
    } else {
        file.persist_noclobber(path)
    };
    result.map(|_| ()).map_err(|e| io_error(path, e.error))
}
fn print_report(report: &Report) {
    println!(
        "{} vertices, {} triangles, {} meshes; {} mapped channels; unsupported: {}",
        report.output_vertices,
        report.output_triangles,
        report.output_meshes,
        report.channels.len(),
        report.unsupported_channels.join(", ")
    );
    println!(
        "decoded: {} bytes; GPU geometry: {} bytes; largest storage buffer: {} bytes; GLB: {} bytes",
        report.decoded_bytes,
        report.gpu_geometry_bytes,
        report.largest_gpu_storage_buffer_bytes,
        report.glb_upper_bound_bytes
    );
    println!(
        "{} warnings (details in JSON report)",
        report.warnings.len()
    );
    for warning in report
        .warnings
        .iter()
        .filter(|w| !w.starts_with("nonplanar neutral face"))
        .take(12)
    {
        println!("warning: {warning}");
    }
}
fn run(command: Command) -> Result<()> {
    let (input, output) = match command {
        Command::Inspect { input } => (input, None),
        Command::Convert { input, output } => (input, Some(output)),
    };
    let config_text =
        std::fs::read_to_string(&input.config).map_err(|e| io_error(&input.config, e))?;
    let config = Config::parse(&config_text)?;
    let paths = audio2face3d_headgen::obj::resolve_inputs(&config, &input.input_root)?;
    let report_path = input
        .report
        .or_else(|| output.as_ref().map(|p| p.with_extension("report.json")));
    let outputs = output
        .iter()
        .chain(report_path.iter())
        .map(PathBuf::as_path)
        .collect::<Vec<_>>();
    validate_outputs(&outputs, &paths, &input.config, input.force)?;
    let mut converted = audio2face3d_headgen::convert(&config, &input.input_root)?;
    let glb = if output.is_some() {
        let bytes = audio2face3d_gui_core::gltf::to_glb(&converted.model)
            .map_err(|e| Error::Output(e.to_string()))?;
        converted.report.output_glb_sha256 = Some(report::hash(&bytes));
        converted.report.output_glb_bytes = Some(bytes.len());
        Some(bytes)
    } else {
        None
    };
    let json =
        serde_json::to_vec_pretty(&converted.report).map_err(|e| Error::Output(e.to_string()))?;
    // Stage both files before replacing either destination.
    let staged_glb = output
        .as_ref()
        .zip(glb.as_ref())
        .map(|(p, b)| stage(p, b))
        .transpose()?;
    let staged_report = report_path.as_ref().map(|p| stage(p, &json)).transpose()?;
    if let (Some(file), Some(path)) = (staged_glb, output.as_ref()) {
        commit(file, path, input.force)?;
    }
    if let (Some(file), Some(path)) = (staged_report, report_path.as_ref()) {
        commit(file, path, input.force).map_err(|e| {
            Error::Output(format!(
                "{e}; GLB committed: {}; SHA-256: {}",
                output.is_some(),
                converted
                    .report
                    .output_glb_sha256
                    .as_deref()
                    .unwrap_or("not written")
            ))
        })?;
    }
    print_report(&converted.report);
    if let Some(path) = output {
        println!("GLB: {}", path.display());
    }
    if let Some(path) = report_path {
        println!("Report: {}", path.display());
    }
    Ok(())
}
fn main() {
    if let Err(error) = run(Arguments::parse().command) {
        eprintln!("{error}");
        std::process::exit(error.exit_code());
    }
}
