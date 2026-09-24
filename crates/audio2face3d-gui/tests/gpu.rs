#![cfg(feature = "render-wgpu")]
use audio2face3d_gui::render::{Camera, HeadRenderer, RenderTarget, Vertex};
use audio2face3d_gui_core::{HeadModel, Material, Mesh, Metadata, MorphTarget};

#[test]
#[ignore = "requires a GPU adapter; run explicitly in the GPU validation tier"]
fn all_52_targets_match_cpu_after_name_reordering_and_resize() {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&Default::default())).expect("GPU adapter");
    eprintln!("{:?}", adapter.get_info());
    let (device, queue) =
        pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut model = HeadModel {
        metadata: Metadata {
            schema_version: 1,
            rig_profile: "audio2face_rs_tester_v1".into(),
            generator_version: "test".into(),
        },
        meshes: vec![Mesh {
            name: "triangle".into(),
            positions: vec![[0., 0., 0.], [0.05, 0., 0.], [0., 0.05, 0.]],
            normals: vec![[0., 0., 1.]; 3],
            indices: vec![0, 1, 2],
            material: Material {
                color: [0.5, 0.5, 0.5, 1.],
            },
            targets: audio2face3d_gui_core::rig::CHANNELS
                .iter()
                .enumerate()
                .map(|(i, &name)| MorphTarget {
                    name: name.into(),
                    positions: vec![[0.0001 * i as f32, 0.0001, 0.]; 3],
                    normals: vec![[0.001, 0., 0.]; 3],
                })
                .collect(),
        }],
    };
    let weights = audio2face3d_gui_core::rig::CHANNELS
        .iter()
        .enumerate()
        .map(|(i, &n)| (n.to_string(), i as f32 / 51.))
        .collect();
    for size in [[128, 96], [64, 128]] {
        model.meshes[0].targets.reverse();
        let renderer = HeadRenderer::new(&device, &model).unwrap();
        let target = RenderTarget::new(&device, size);
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 96,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        renderer
            .render(&queue, &mut encoder, &target, Camera::default(), &weights)
            .unwrap();
        renderer.copy_vertices(&mut encoder, 0, &output);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        output
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let bytes = output.slice(..).get_mapped_range();
        let mesh = &model.meshes[0];
        let cpu = audio2face3d_gui_core::model::evaluate(
            mesh,
            &mesh
                .targets
                .iter()
                .map(|t| weights[&t.name])
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for (i, bytes) in bytes.chunks_exact(32).enumerate() {
            let v: Vertex = bytemuck::pod_read_unaligned(bytes);
            for c in 0..3 {
                assert!((v.position[c] - cpu.positions[i][c]).abs() < 1e-5);
                assert!((v.normal[c] - cpu.normals[i][c]).abs() < 1e-5);
            }
        }
    }
    assert!(pollster::block_on(device.pop_error_scope()).is_none());
}

