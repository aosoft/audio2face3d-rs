pub fn calculate(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.; 3]; positions.len()];
    for triangle in indices.chunks_exact(3) {
        let [a, b, c] = [
            positions[triangle[0] as usize],
            positions[triangle[1] as usize],
            positions[triangle[2] as usize],
        ];
        let u = std::array::from_fn::<_, 3, _>(|i| b[i] - a[i]);
        let v = std::array::from_fn::<_, 3, _>(|i| c[i] - a[i]);
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        for &index in triangle {
            for (i, value) in n.iter().enumerate() {
                normals[index as usize][i] += value;
            }
        }
    }
    for normal in &mut normals {
        let length = normal.iter().map(|x| x * x).sum::<f32>().sqrt();
        *normal = if length > 1e-12 {
            normal.map(|x| x / length)
        } else {
            [0., 0., 1.]
        };
    }
    normals
}

/// Area weighted normals using shared original indices, before material splitting.
pub fn checked(
    positions: &[[f32; 3]],
    indices: &[u32],
    label: &str,
) -> crate::error::Result<Vec<[f32; 3]>> {
    use crate::transform::{cross, dot, sub};
    let mut normals = vec![[0.; 3]; positions.len()];
    let mut used = vec![false; positions.len()];
    for (i, t) in indices.chunks_exact(3).enumerate() {
        let n = cross(
            sub(positions[t[1] as usize], positions[t[0] as usize]),
            sub(positions[t[2] as usize], positions[t[0] as usize]),
        );
        if !dot(n, n).is_finite() || dot(n, n) <= 1e-30 {
            return Err(crate::error::Error::Input(format!(
                "{label}: degenerate triangle {}",
                i + 1
            )));
        }
        for &v in t {
            used[v as usize] = true;
            for (c, nc) in n.iter().enumerate() {
                normals[v as usize][c] += nc;
            }
        }
    }
    for (i, n) in normals.iter_mut().enumerate() {
        if !used[i] {
            continue;
        }
        let length = dot(*n, *n).sqrt();
        if !length.is_finite() || length <= 1e-20 {
            return Err(crate::error::Error::Input(format!(
                "{label}: undefined normal at vertex {}",
                i + 1
            )));
        }
        *n = n.map(|x| x / length);
    }
    Ok(normals)
}
