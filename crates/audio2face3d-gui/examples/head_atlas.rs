//! Contact sheets: rows follow the canonical channel order, columns are
//! front 0, front 0.5, front 1, three-quarter 1. Outputs are diagnostic artifacts.
use audio2face3d_gui::render::{Camera, HeadRenderer, RenderTarget};
use std::{collections::BTreeMap, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let model =
        audio2face3d_gui::gltf::from_glb(&std::fs::read(args.get(1).ok_or("GLB path required")?)?)?;
    let directory = PathBuf::from(args.get(2).ok_or("output directory required")?);
    std::fs::create_dir_all(&directory)?;
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).ok_or("GPU required")?;
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None))?;
    let renderer = HeadRenderer::new(&device, &model)?;
    let target = RenderTarget::new(&device, [192, 192]);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 192 * 192 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let supported = model.channel_names();
    let names = audio2face3d_gui::rig::CHANNELS
        .into_iter()
        .filter(|n| supported.iter().any(|s| s == n))
        .collect::<Vec<_>>();
    println!(
        "Unsupported channels: {}",
        model.unsupported_channels().join(", ")
    );
    for (page, names) in names.chunks(9).enumerate() {
        let mut sheet = image::RgbaImage::new(192 * 4, 192 * names.len() as u32);
        for (row, name) in names.iter().enumerate() {
            for (column, (value, yaw)) in [(0., 0.), (0.5, 0.), (1., 0.), (1., 0.6)]
                .into_iter()
                .enumerate()
            {
                let weights = BTreeMap::from([(name.to_string(), value)]);
                let mut encoder = device.create_command_encoder(&Default::default());
                renderer.render(
                    &queue,
                    &mut encoder,
                    &target,
                    Camera {
                        yaw,
                        ..Default::default()
                    },
                    &weights,
                )?;
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &target.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &staging,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(768),
                            rows_per_image: Some(192),
                        },
                    },
                    wgpu::Extent3d {
                        width: 192,
                        height: 192,
                        depth_or_array_layers: 1,
                    },
                );
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                staging
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::Maintain::Wait);
                rx.recv()?.map_err(|e| e.to_string())?;
                let pixels = staging.slice(..).get_mapped_range().to_vec();
                staging.unmap();
                let frame = image::RgbaImage::from_raw(192, 192, pixels).ok_or("invalid pixels")?;
                image::imageops::overlay(&mut sheet, &frame, column as i64 * 192, row as i64 * 192);
            }
        }
        let path = directory.join(format!("page-{}.png", page + 1));
        sheet.save(&path)?;
        std::fs::write(
            directory.join(format!("page-{}.txt", page + 1)),
            names.join("\n"),
        )?;
        println!("{}: {}", path.display(), names.join(", "));
    }
    Ok(())
}
