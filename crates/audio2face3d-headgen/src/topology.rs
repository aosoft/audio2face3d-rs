use crate::{
    error::{Error, Result},
    obj::Obj,
};
/// Compare before exclusion, triangulation, or vertex splitting.
pub fn validate(neutral: &Obj, expression: &Obj, label: &str) -> Result<()> {
    if neutral.positions.len() != expression.positions.len()
        || neutral.faces.len() != expression.faces.len()
    {
        return Err(Error::Input(format!(
            "{label}: vertex/face count mismatch with neutral"
        )));
    }
    for (i, (a, b)) in neutral.faces.iter().zip(&expression.faces).enumerate() {
        if a.vertices != b.vertices {
            return Err(Error::Input(format!(
                "{label}:{}: topology mismatch at face {}",
                b.line,
                i + 1
            )));
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_counts_are_not_sufficient() {
        let v = "v 0 0 0\nv 1 0 0\nv 0 1 0\n";
        let a = crate::obj::parse(format!("{v}f 1 2 3").as_bytes(), "a").unwrap();
        let b = crate::obj::parse(format!("{v}f 1 3 2").as_bytes(), "b").unwrap();
        assert!(
            validate(&a, &b, "b")
                .unwrap_err()
                .to_string()
                .contains("face 1")
        );
        let c = crate::obj::parse(format!("{v}vt 0\nf 1/1 2/1 3/1").as_bytes(), "c").unwrap();
        validate(&a, &c, "c").unwrap();
    }
}

/// Select a neutral-space diagonal once. The same indices apply to every pose.
pub fn triangulate(obj: &Obj) -> Result<(Vec<Vec<u32>>, Vec<String>)> {
    use crate::transform::{cross, dot, sub};
    let mut result = Vec::new();
    let mut warnings = Vec::new();
    for (face_index, face) in obj.faces.iter().enumerate() {
        let v = &face.vertices;
        let p = v
            .iter()
            .map(|&i| obj.positions[i as usize])
            .collect::<Vec<_>>();
        let fail = || {
            Error::Input(format!(
                "neutral:{}: degenerate or self-intersecting face {}",
                face.line,
                face_index + 1
            ))
        };
        let n = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        if v.len() == 3 {
            if !dot(n, n).is_finite() || dot(n, n) <= 1e-30 {
                return Err(fail());
            }
            result.push(v.clone());
            continue;
        }
        let n2 = cross(sub(p[2], p[0]), sub(p[3], p[0]));
        let normal = std::array::from_fn(|i| n[i] + n2[i]);
        let length = dot(normal, normal).sqrt();
        if !length.is_finite() || length <= 1e-15 {
            return Err(fail());
        }
        let normal = normal.map(|x| x / length);
        // Project orthogonally onto the polygon's own plane. Dropping the
        // dominant coordinate can fold a valid nonplanar thin quad in projection.
        let axis = (0..3)
            .min_by(|&a, &b| normal[a].abs().total_cmp(&normal[b].abs()))
            .unwrap();
        let mut reference = [0.; 3];
        reference[axis] = 1.;
        let tangent = cross(reference, normal);
        let tangent = tangent.map(|x| x / dot(tangent, tangent).sqrt());
        let bitangent = cross(normal, tangent);
        let project = |point: [f32; 3]| -> [f64; 2] {
            let relative = sub(point, p[0]);
            [
                dot(relative, tangent) as f64,
                dot(relative, bitangent) as f64,
            ]
        };
        let q = p.iter().copied().map(project).collect::<Vec<_>>();
        let area = |a: usize, b: usize, c: usize| {
            (q[b][0] - q[a][0]) * (q[c][1] - q[a][1]) - (q[b][1] - q[a][1]) * (q[c][0] - q[a][0])
        };
        let intersects =
            |a, b, c, d| area(a, b, c) * area(a, b, d) <= 0. && area(c, d, a) * area(c, d, b) <= 0.;
        if intersects(0, 1, 2, 3) || intersects(1, 2, 3, 0) {
            return Err(fail());
        }
        let valid = |a, b, c| area(a, b, c) > 1e-20;
        let ac = valid(0, 1, 2) && valid(0, 2, 3);
        let bd = valid(0, 1, 3) && valid(1, 2, 3);
        if !ac && !bd {
            return Err(fail());
        }
        let dist = |a, b| dot(sub(p[a], p[b]), sub(p[a], p[b]));
        let tri = if ac && (!bd || dist(0, 2) <= dist(1, 3)) {
            vec![v[0], v[1], v[2], v[0], v[2], v[3]]
        } else {
            vec![v[0], v[1], v[3], v[1], v[2], v[3]]
        };
        let deviation = dot(sub(p[3], p[0]), n).abs() / dot(n, n).sqrt().max(1e-20);
        if deviation > dist(0, 2).sqrt() * 1e-4 {
            warnings.push(format!(
                "nonplanar neutral face {} (line {})",
                face_index + 1,
                face.line
            ));
        }
        result.push(tri);
    }
    Ok((result, warnings))
}
#[cfg(test)]
mod triangulation_tests {
    use super::*;
    fn obj(v: &str) -> Obj {
        crate::obj::parse(format!("{v}\nf 1 2 3 4").as_bytes(), "quad").unwrap()
    }
    #[test]
    fn convex_concave_and_crossed_quads() {
        let convex = obj("v 0 0 0\nv 2 0 0\nv 2 1 0\nv 0 1 0");
        assert_eq!(triangulate(&convex).unwrap().0[0], [0, 1, 2, 0, 2, 3]);
        let concave = obj("v 0 0 0\nv 2 0 0\nv 0.5 0.5 0\nv 0 2 0");
        assert_eq!(triangulate(&concave).unwrap().0[0], [0, 1, 2, 0, 2, 3]);
        let crossed = obj("v 0 0 0\nv 1 1 0\nv 0 1 0\nv 1 0 0");
        assert!(triangulate(&crossed).is_err());
    }
}

#[cfg(test)]
mod nonplanar_projection_tests {
    #[test]
    fn face_plane_projection_does_not_fold_a_nonplanar_quad() {
        // Independent integer fixture; no source dataset geometry is embedded.
        let obj = crate::obj::parse(
            b"v 0 0 0\nv -3 -3 -1\nv -2 1 -1\nv -3 -3 1\nf 1 2 3 4".as_slice(),
            "synthetic",
        )
        .unwrap();
        let (triangles, warnings) = super::triangulate(&obj).unwrap();
        assert_eq!(triangles[0].len(), 6);
        assert!(!warnings.is_empty());
    }
}
