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
