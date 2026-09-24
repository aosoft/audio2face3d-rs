use crate::{
    config::DegeneratePoseTriangles,
    error::{Error, Result},
    transform::{cross, dot, sub},
};
pub struct Normals {
    pub normals: Vec<[f32; 3]>,
    pub skipped_triangles: Vec<usize>,
}
/// Area weighted normals using shared original indices, before material splitting.
pub fn checked(positions: &[[f32; 3]], indices: &[u32], label: &str) -> Result<Vec<[f32; 3]>> {
    Ok(posed(
        positions,
        indices,
        label,
        &[],
        DegeneratePoseTriangles::Error,
    )?
    .normals)
}
pub fn posed(
    positions: &[[f32; 3]],
    indices: &[u32],
    label: &str,
    face_numbers: &[usize],
    policy: DegeneratePoseTriangles,
) -> Result<Normals> {
    let mut normals = vec![[0.; 3]; positions.len()];
    let mut used = vec![false; positions.len()];
    let mut skipped_triangles = Vec::new();
    for (i, t) in indices.chunks_exact(3).enumerate() {
        for &v in t {
            used[v as usize] = true;
        }
        let n = cross(
            sub(positions[t[1] as usize], positions[t[0] as usize]),
            sub(positions[t[2] as usize], positions[t[0] as usize]),
        );
        let area_squared = dot(n, n);
        if !area_squared.is_finite() {
            return Err(Error::Input(format!(
                "{label}: triangle {} normal overflow",
                i + 1
            )));
        }
        if area_squared <= 1e-30 {
            if policy == DegeneratePoseTriangles::SkipNormalContribution {
                skipped_triangles.push(i + 1);
                continue;
            }
            return Err(Error::Input(format!(
                "{label}: degenerate triangle {} (source face {})",
                i + 1,
                face_numbers
                    .get(i)
                    .map_or_else(|| "unknown".into(), usize::to_string)
            )));
        }
        for &v in t {
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
            return Err(Error::Input(format!(
                "{label}: undefined normal at vertex {} (no usable surrounding surface)",
                i + 1
            )));
        }
        *n = n.map(|x| x / length);
    }
    Ok(Normals {
        normals,
        skipped_triangles,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tolerant_pose_uses_neighboring_faces_but_never_invents_normals() {
        let p = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]];
        let indices = [0, 1, 2, 0, 2, 3, 0, 1, 3];
        assert!(checked(&p, &indices, "neutral").is_err());
        let n = posed(
            &p,
            &indices,
            "pose",
            &[1, 1, 2],
            DegeneratePoseTriangles::SkipNormalContribution,
        )
        .unwrap();
        assert_eq!(n.skipped_triangles, [2]);
        assert_eq!(n.normals, vec![[0., 0., 1.]; 4]);
        assert!(
            posed(
                &p,
                &indices[..6],
                "pose",
                &[1, 1],
                DegeneratePoseTriangles::SkipNormalContribution
            )
            .is_err()
        );
    }
}
