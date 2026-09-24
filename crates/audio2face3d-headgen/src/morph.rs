use crate::{Config, normals};
use audio2face3d_gui_core::{Mesh, MorphTarget};
pub fn add_targets(mesh: &mut Mesh, config: &Config) {
    for name in audio2face3d_gui_core::rig::CHANNELS {
        let positions: Vec<_> = mesh
            .positions
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let d = displacement(&mesh.name, i, config.unscale(p), name);
                config.scale(d.map(|x| x * config.expression_scale))
            })
            .collect();
        if !positions.iter().flatten().any(|x| x.abs() > 1e-8) {
            continue;
        }
        let posed: Vec<_> = mesh
            .positions
            .iter()
            .zip(&positions)
            .map(|(p, d)| std::array::from_fn(|i| p[i] + d[i]))
            .collect();
        let normals = normals::calculate(&posed, &mesh.indices)
            .iter()
            .zip(&mesh.normals)
            .map(|(p, n)| std::array::from_fn(|i| p[i] - n[i]))
            .collect();
        mesh.targets.push(MorphTarget {
            name: name.into(),
            positions,
            normals,
        });
    }
}
fn displacement(part: &str, index: usize, p: [f32; 3], name: &str) -> [f32; 3] {
    let [x, y, z] = p;
    if name == "JawOpen" {
        let amount = match part {
            "head" => ((-y - 0.015) / 0.085).clamp(0., 1.) * (z / 0.03).clamp(0., 1.),
            "lips" | "mouth" => ((-y - 0.035) / 0.006).clamp(0., 1.),
            "lower_teeth" | "tongue" => 1.,
            _ => 0.,
        };
        return [0., -0.028 * amount, -0.004 * amount];
    }
    if name.starts_with("EyeBlink")
        && part
            == format!(
                "lid{}",
                if name.ends_with("Left") {
                    "Left"
                } else {
                    "Right"
                }
            )
    {
        let factor = if index % 2 == 1 { 1. } else { 0.15 };
        return [0., (0.027 - y) * factor, 0.];
    }
    if name.starts_with("MouthSmile") && matches!(part, "lips" | "mouth" | "head") {
        let side = if name.ends_with("Left") { 1. } else { -1. };
        let area = (-(y + 0.038).powi(2) / 0.0004).exp()
            * (x * side / 0.035).clamp(0., 1.)
            * (z / 0.04).clamp(0., 1.);
        return [0.006 * side * area, 0.012 * area, 0.];
    }
    let side = if name.ends_with("Left") {
        1.
    } else if name.ends_with("Right") {
        -1.
    } else {
        0.
    };
    let matches_side =
        (side > 0. && part.ends_with("Left")) || (side < 0. && part.ends_with("Right"));
    if name.starts_with("Eye") && matches_side {
        if part.starts_with("iris") {
            if name.contains("LookDown") {
                return [0., -0.006, 0.];
            }
            if name.contains("LookUp") {
                return [0., 0.006, 0.];
            }
            if name.contains("LookIn") {
                return [-side * 0.006, 0., 0.];
            }
            if name.contains("LookOut") {
                return [side * 0.006, 0., 0.];
            }
        }
        if part.starts_with("lid") {
            let local = y - 0.027;
            let factor = if index % 2 == 1 { 1. } else { 0.2 };
            if name.contains("Squint") {
                return [
                    0.,
                    if local < 0. {
                        -local * 0.75 * factor
                    } else {
                        -local * 0.15 * factor
                    },
                    0.,
                ];
            }
            if name.contains("Wide") {
                return [0., local * 0.55 * factor, 0.];
            }
        }
    }
    let jaw = match part {
        "head" => ((-y - 0.015) / 0.085).clamp(0., 1.) * (z / 0.03).clamp(0., 1.),
        "lips" | "mouth" | "lower_teeth" | "tongue" => 1.,
        _ => 0.,
    };
    match name {
        "JawForward" => return [0., 0., 0.012 * jaw],
        "JawLeft" => return [0.012 * jaw, 0., 0.],
        "JawRight" => return [-0.012 * jaw, 0., 0.],
        "MouthClose" => {
            if matches!(part, "lips" | "mouth") {
                return [0., -0.018 - (y + 0.038) * 0.7, 0.];
            }
            if matches!(part, "upper_teeth" | "lower_teeth" | "tongue") {
                return [0., -0.018 - (y + 0.038) * 0.7, -0.002];
            }
            return [0., -0.028 * jaw, -0.002 * jaw];
        }
        "TongueOut" if part == "tongue" => return [0., -0.003, 0.025],
        _ => {}
    }
    if name.starts_with("Brow") {
        let inner = 1. - (x.abs() / 0.065).clamp(0., 1.);
        let outer = ((x.abs() - 0.01) / 0.04).clamp(0., 1.);
        let amount = if part.starts_with("brow") {
            1.
        } else if part == "head" {
            gaussian(x.abs() - 0.03, 0.03) * gaussian(y - 0.05, 0.018) * (z / 0.04).clamp(0., 1.)
        } else {
            0.
        };
        if name == "BrowInnerUp" {
            return [0., 0.017 * inner * amount, 0.];
        }
        if x * side > 0. {
            if name.starts_with("BrowDown") {
                return [
                    0.,
                    -0.013 * amount,
                    if part.starts_with("brow") {
                        0.010 * amount
                    } else {
                        0.
                    },
                ];
            }
            if name.starts_with("BrowOuterUp") {
                return [0., 0.016 * outer * amount, 0.];
            }
        }
    }
    if part == "head" {
        let cheek = gaussian(x.abs() - 0.047, 0.023)
            * gaussian(y + 0.008, 0.025)
            * (z / 0.04).clamp(0., 1.);
        if name == "CheekPuff" {
            return [0.008 * x.signum() * cheek, 0., 0.010 * cheek];
        }
        if name.starts_with("CheekSquint") && x * side > 0. {
            return [0., 0.012 * cheek, 0.003 * cheek];
        }
    }
    if name.starts_with("NoseSneer") && matches!(part, "nose" | "head") && x * side > 0. {
        let amount = gaussian(x - side * 0.009, 0.017)
            * gaussian(y + 0.004, 0.016)
            * (z / 0.04).clamp(0., 1.);
        return [0., 0.012 * amount, 0.004 * amount];
    }
    if name.starts_with("Mouth") && matches!(part, "lips" | "mouth" | "head") {
        let local = y + 0.038;
        let area = if part == "head" {
            gaussian(local, 0.026) * gaussian(x, 0.050) * (z / 0.04).clamp(0., 1.)
        } else {
            1.
        };
        let half = if side == 0. {
            1.
        } else {
            (x * side / 0.015).clamp(0., 1.)
        };
        let corner = (x.abs() / 0.035).clamp(0., 1.);
        let upper = (local / 0.006).clamp(0., 1.);
        let lower = (-local / 0.006).clamp(0., 1.);
        let d = match name {
            "MouthFunnel" => [-x * 0.35, local * 1.4, 0.012],
            "MouthPucker" => [-x * 0.65, -local * 0.45, 0.018],
            "MouthLeft" => [0.012, 0., 0.],
            "MouthRight" => [-0.012, 0., 0.],
            "MouthFrownLeft" | "MouthFrownRight" => [0., -0.012 * corner, 0.],
            "MouthDimpleLeft" | "MouthDimpleRight" => [side * 0.004 * corner, 0., -0.006 * corner],
            "MouthStretchLeft" | "MouthStretchRight" => [side * 0.012 * corner, 0., 0.],
            "MouthRollLower" => [0., 0.003 * lower, -0.006 * lower],
            "MouthRollUpper" => [0., -0.003 * upper, -0.006 * upper],
            "MouthShrugLower" => [0., 0.007 * lower, 0.002 * lower],
            "MouthShrugUpper" => [0., 0.007 * upper, 0.002 * upper],
            "MouthPressLeft" | "MouthPressRight" => [0., -local * 0.8, 0.002],
            "MouthLowerDownLeft" | "MouthLowerDownRight" => [0., -0.012 * lower, 0.],
            "MouthUpperUpLeft" | "MouthUpperUpRight" => [0., 0.012 * upper, 0.],
            _ => [0.; 3],
        };
        return d.map(|v| v * area * half);
    }
    [0.; 3]
}

fn gaussian(value: f32, width: f32) -> f32 {
    (-(value / width).powi(2)).exp()
}
