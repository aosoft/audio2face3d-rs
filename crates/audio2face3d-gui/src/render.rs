//! Host-owned wgpu device/queue, reusable offscreen head rendering.
use audio2face3d_gui_core::{HeadModel, ModelError};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use std::collections::BTreeMap;
use wgpu::util::DeviceExt;

pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    matrix: [[f32; 4]; 4],
    color: [f32; 4],
    counts: [u32; 4],
}

#[derive(Clone, Copy)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.,
            pitch: 0.,
            distance: 0.48,
        }
    }
}
impl Camera {
    fn matrix(self, aspect: f32) -> [[f32; 4]; 4] {
        let center = Vec3::new(0., -0.015, 0.);
        let direction = Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        );
        (Mat4::perspective_rh(40f32.to_radians(), aspect, 0.01, 10.)
            * Mat4::look_at_rh(center + direction * self.distance, center, Vec3::Y))
        .to_cols_array_2d()
    }
}

pub struct RenderTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    pub size: [u32; 2],
}
impl RenderTarget {
    pub fn new(device: &wgpu::Device, size: [u32; 2]) -> Self {
        let size = size.map(|x| x.clamp(1, device.limits().max_texture_dimension_2d));
        let descriptor = wgpu::TextureDescriptor {
            label: Some("head target"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = device.create_texture(&descriptor);
        let view = texture.create_view(&Default::default());
        let depth = device
            .create_texture(&wgpu::TextureDescriptor {
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                ..descriptor
            })
            .create_view(&Default::default());
        Self {
            texture,
            view,
            depth,
            size,
        }
    }
}

struct GpuMesh {
    result: wgpu::Buffer,
    indices: wgpu::Buffer,
    weights: wgpu::Buffer,
    params: wgpu::Buffer,
    compute: wgpu::BindGroup,
    draw: wgpu::BindGroup,
    vertices: u32,
    index_count: u32,
    names: Vec<String>,
    color: [f32; 4],
}
pub struct HeadRenderer {
    meshes: Vec<GpuMesh>,
    compute: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
}

fn buffer(
    device: &wgpu::Device,
    label: &str,
    data: &[u8],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: data,
        usage,
    })
}
fn vertices(p: &[[f32; 3]], n: &[[f32; 3]], w: f32) -> Vec<Vertex> {
    p.iter()
        .zip(n)
        .map(|(p, n)| Vertex {
            position: [p[0], p[1], p[2], w],
            normal: [n[0], n[1], n[2], 0.],
        })
        .collect()
}
impl HeadRenderer {
    pub fn new(device: &wgpu::Device, model: &HeadModel) -> Result<Self, ModelError> {
        model.validate()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("morph"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render/morph.wgsl").into()),
        });
        let compute = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("morph"),
            layout: None,
            module: &shader,
            entry_point: Some("morph"),
            compilation_options: Default::default(),
            cache: None,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("head"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render/head.wgsl").into()),
        });
        let draw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("head"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0=>Float32x4,1=>Float32x4],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: COLOR_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let mut meshes = Vec::new();
        for mesh in &model.meshes {
            let base = vertices(&mesh.positions, &mesh.normals, 1.);
            let mut deltas: Vec<_> = mesh
                .targets
                .iter()
                .flat_map(|t| vertices(&t.positions, &t.normals, 0.))
                .collect();
            if deltas.is_empty() {
                deltas.push(Vertex::zeroed());
            }
            if std::mem::size_of_val(deltas.as_slice()) as u64
                > device.limits().max_storage_buffer_binding_size as u64
            {
                return Err(ModelError("morph data exceeds GPU storage limit".into()));
            }
            let base = buffer(
                device,
                "base",
                bytemuck::cast_slice(&base),
                wgpu::BufferUsages::STORAGE,
            );
            let deltas = buffer(
                device,
                "deltas",
                bytemuck::cast_slice(&deltas),
                wgpu::BufferUsages::STORAGE,
            );
            let result = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("posed vertices"),
                size: mesh.positions.len() as u64 * 32,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::VERTEX
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let weights = buffer(
                device,
                "weights",
                bytemuck::cast_slice(&[0f32; 52]),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            );
            let params = buffer(
                device,
                "params",
                bytemuck::bytes_of(&Params::zeroed()),
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );
            let compute_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("morph"),
                layout: &compute.get_bind_group_layout(0),
                entries: &[&base, &deltas, &weights, &result, &params]
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            });
            let draw_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("head"),
                layout: &draw.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                }],
            });
            meshes.push(GpuMesh {
                result,
                indices: buffer(
                    device,
                    "indices",
                    bytemuck::cast_slice(&mesh.indices),
                    wgpu::BufferUsages::INDEX,
                ),
                weights,
                params,
                compute: compute_group,
                draw: draw_group,
                vertices: mesh.positions.len() as u32,
                index_count: mesh.indices.len() as u32,
                names: mesh.targets.iter().map(|t| t.name.clone()).collect(),
                color: mesh.material.color,
            });
        }
        Ok(Self {
            meshes,
            compute,
            draw,
        })
    }

    /// Encode into a host-owned target. Submit before reusing mutable weights/uniforms.
    pub fn render(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &RenderTarget,
        camera: Camera,
        weights: &BTreeMap<String, f32>,
    ) -> Result<(), ModelError> {
        if weights.values().any(|v| !v.is_finite())
            || !camera.yaw.is_finite()
            || !camera.pitch.is_finite()
            || !camera.distance.is_finite()
            || camera.distance <= 0.
        {
            return Err(ModelError("nonfinite render state".into()));
        }
        let matrix = camera.matrix(target.size[0] as f32 / target.size[1] as f32);
        for mesh in &self.meshes {
            let mut values = [0.; 52];
            for (i, name) in mesh.names.iter().enumerate() {
                values[i] = *weights.get(name).unwrap_or(&0.);
            }
            queue.write_buffer(&mesh.weights, 0, bytemuck::cast_slice(&values));
            queue.write_buffer(
                &mesh.params,
                0,
                bytemuck::bytes_of(&Params {
                    matrix,
                    color: mesh.color,
                    counts: [mesh.vertices, mesh.names.len() as u32, 0, 0],
                }),
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("morph"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute);
            for mesh in &self.meshes {
                pass.set_bind_group(0, &mesh.compute, &[]);
                pass.dispatch_workgroups(mesh.vertices.div_ceil(64), 1, 1);
            }
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("head"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.018,
                        g: 0.024,
                        b: 0.035,
                        a: 1.,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &target.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.draw);
        for mesh in &self.meshes {
            pass.set_bind_group(0, &mesh.draw, &[]);
            pass.set_vertex_buffer(0, mesh.result.slice(..));
            pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.index_count, 0, 0..1);
        }
        Ok(())
    }
    /// GPU diagnostics: copy evaluated vertex data without exposing the model loader.
    pub fn copy_vertices(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        mesh: usize,
        destination: &wgpu::Buffer,
    ) {
        let mesh = &self.meshes[mesh];
        encoder.copy_buffer_to_buffer(&mesh.result, 0, destination, 0, mesh.vertices as u64 * 32);
    }
}
