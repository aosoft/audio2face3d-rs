use crate::Config;
use audio2face3d_gui_core::{Material, Mesh};
use std::f32::consts::{PI, TAU};
fn empty(name: &str, color: [f32; 4]) -> Mesh {
    Mesh {
        name: name.into(),
        positions: vec![],
        normals: vec![],
        indices: vec![],
        material: Material { color },
        targets: vec![],
    }
}
fn ellipsoid(
    name: &str,
    center: [f32; 3],
    radius: [f32; 3],
    segments: u32,
    rings: u32,
    color: [f32; 4],
) -> Mesh {
    let mut mesh = empty(name, color);
    mesh.positions
        .push([center[0], center[1] + radius[1], center[2]]);
    for row in 1..rings {
        let phi = PI * row as f32 / rings as f32;
        for column in 0..segments {
            let theta = TAU * column as f32 / segments as f32;
            mesh.positions.push([
                center[0] + radius[0] * phi.sin() * theta.cos(),
                center[1] + radius[1] * phi.cos(),
                center[2] + radius[2] * phi.sin() * theta.sin(),
            ]);
        }
    }
    let bottom = mesh.positions.len() as u32;
    mesh.positions
        .push([center[0], center[1] - radius[1], center[2]]);
    for i in 0..segments {
        let next = (i + 1) % segments;
        mesh.indices.extend([0, 1 + next, 1 + i]);
        for row in 0..rings - 2 {
            let a = 1 + row * segments + i;
            let b = 1 + row * segments + next;
            mesh.indices
                .extend([a, b, a + segments, b, b + segments, a + segments]);
        }
        let base = 1 + (rings - 2) * segments;
        mesh.indices.extend([bottom, base + i, base + next]);
    }
    mesh
}
fn ring(name: &str, center: [f32; 3], outer: [f32; 2], inner: [f32; 2], color: [f32; 4]) -> Mesh {
    let mut mesh = empty(name, color);
    for i in 0..32 {
        let a = TAU * i as f32 / 32.;
        for (radius, z) in [(outer, -0.003), (inner, 0.)] {
            mesh.positions.push([
                center[0] + radius[0] * a.cos(),
                center[1] + radius[1] * a.sin(),
                center[2] + z,
            ]);
        }
        let j = (i + 1) % 32;
        mesh.indices
            .extend([2 * i, 2 * j, 2 * i + 1, 2 * j, 2 * j + 1, 2 * i + 1]);
    }
    mesh
}
pub fn parts(config: &Config) -> Vec<Mesh> {
    let skin = [0.58, 0.69, 0.73, 1.];
    let mut parts = vec![
        ellipsoid(
            "head",
            [0., 0., 0.],
            [0.08, 0.12, 0.07],
            config.segments,
            config.rings,
            skin,
        ),
        ellipsoid(
            "neck",
            [0., -0.122, -0.016],
            [0.034, 0.035, 0.034],
            16,
            6,
            skin,
        ),
        ellipsoid("nose", [0., 0., 0.07], [0.012, 0.021, 0.016], 16, 8, skin),
        ellipsoid(
            "mouth",
            [0., -0.038, 0.073],
            [0.033, 0.005, 0.001],
            24,
            6,
            [0.055, 0.018, 0.025, 1.],
        ),
        ring(
            "lips",
            [0., -0.038, 0.077],
            [0.040, 0.010],
            [0.030, 0.003],
            [0.52, 0.29, 0.32, 1.],
        ),
        ellipsoid(
            "upper_teeth",
            [0., -0.039, 0.0745],
            [0.022, 0.002, 0.0006],
            16,
            4,
            [0.85, 0.86, 0.79, 1.],
        ),
        ellipsoid(
            "lower_teeth",
            [0., -0.041, 0.0745],
            [0.020, 0.0015, 0.0006],
            16,
            4,
            [0.85, 0.86, 0.79, 1.],
        ),
        ellipsoid(
            "tongue",
            [0., -0.042, 0.0735],
            [0.012, 0.0015, 0.001],
            16,
            4,
            [0.75, 0.23, 0.29, 1.],
        ),
    ];
    for (side, sign) in [("Left", 1.), ("Right", -1.)] {
        parts.push(ellipsoid(
            &format!("eye{side}"),
            [sign * 0.030, 0.027, 0.065],
            [0.019, 0.010, 0.013],
            20,
            10,
            [0.92, 0.94, 0.91, 1.],
        ));
        parts.push(ellipsoid(
            &format!("iris{side}"),
            [sign * 0.030, 0.027, 0.078],
            [0.006, 0.006, 0.0015],
            16,
            8,
            [0.035, 0.13, 0.17, 1.],
        ));
        parts.push(ring(
            &format!("lid{side}"),
            [sign * 0.030, 0.027, 0.081],
            [0.027, 0.018],
            [0.017, 0.008],
            skin,
        ));
        parts.push(ellipsoid(
            &format!("brow{side}"),
            [sign * 0.032, 0.052, 0.068],
            [0.021, 0.003, 0.002],
            16,
            4,
            [0.13, 0.20, 0.23, 1.],
        ));
        parts.push(ellipsoid(
            &format!("ear{side}"),
            [sign * 0.078, 0.0, -0.006],
            [0.012, 0.026, 0.014],
            12,
            8,
            skin,
        ));
    }
    for part in &mut parts {
        for position in &mut part.positions {
            *position = config.scale(*position);
        }
    }
    parts
}
