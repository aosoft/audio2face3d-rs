//! Offscreen renderer/CPU parity diagnostic, also usable without the desktop host.
use audio2face3d_gui::render::{Camera, HeadRenderer, RenderTarget, Vertex};
use std::collections::BTreeMap;

fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let data = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    data
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let model =
        audio2face3d_gui::gltf::from_glb(&std::fs::read(args.get(1).ok_or("GLB path required")?)?)?;
    let output = args.get(2).ok_or("PNG path required")?;
    let mut weights = BTreeMap::new();
    let mut camera = Camera::default();
    for arg in &args[3..] {
        let (name, value) = arg
            .split_once('=')
            .ok_or("expected Channel=value or yaw=value")?;
        if name == "yaw" {
            camera.yaw = value.parse()?;
        } else {
            weights.insert(name.into(), value.parse()?);
        }
    }
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&Default::default()))
        .ok_or("no GPU adapter")?;
    println!("Adapter: {:?}", adapter.get_info());
    let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None))?;
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let renderer = HeadRenderer::new(&device, &model)?;
    let target = RenderTarget::new(&device, [512, 512]);
    let mut encoder = device.create_command_encoder(&Default::default());
    renderer.render(&queue, &mut encoder, &target, camera, &weights)?;
    let image_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 512 * 512 * 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &image_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(2048),
                rows_per_image: Some(512),
            },
        },
        wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
    );
    let vertex_buffers: Vec<_> = model
        .meshes
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: m.positions.len() as u64 * 32,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            renderer.copy_vertices(&mut encoder, i, &buffer);
            buffer
        })
        .collect();
    queue.submit([encoder.finish()]);
    let pixels = read(&device, &image_buffer);
    image::save_buffer(output, &pixels, 512, 512, image::ColorType::Rgba8)?;
    let mut max_error = 0f32;
    for (mesh, buffer) in model.meshes.iter().zip(vertex_buffers) {
        let gpu = read(&device, &buffer);
        let cpu = audio2face3d_gui::model::evaluate(
            mesh,
            &mesh
                .targets
                .iter()
                .map(|t| *weights.get(&t.name).unwrap_or(&0.))
                .collect::<Vec<_>>(),
        )?;
        for (i, bytes) in gpu.chunks_exact(32).enumerate() {
            let v: Vertex = bytemuck::pod_read_unaligned(bytes);
            for c in 0..3 {
                max_error = max_error
                    .max((v.position[c] - cpu.positions[i][c]).abs())
                    .max((v.normal[c] - cpu.normals[i][c]).abs());
            }
        }
    }
    if let Some(error) = pollster::block_on(device.pop_error_scope()) {
        return Err(error.into());
    }
    if max_error > 0.00002 {
        return Err(format!("CPU/GPU error {max_error}").into());
    }
    println!("CPU/GPU maximum error {max_error}; captured {output}");
    Ok(())
}
