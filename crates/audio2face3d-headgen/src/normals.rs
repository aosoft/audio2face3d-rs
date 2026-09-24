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
