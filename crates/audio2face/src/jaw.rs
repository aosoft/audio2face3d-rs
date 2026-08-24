use audio2x_core::{Audio2xError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JawParameters {
    pub strength: f32,
    pub height_offset: f32,
    pub depth_offset: f32,
}

impl Default for JawParameters {
    fn default() -> Self {
        Self {
            strength: 1.0,
            height_offset: 0.0,
            depth_offset: 0.0,
        }
    }
}

/// CPU parity oracle for lower-teeth rigid-transform reconstruction.
#[derive(Debug, Clone, PartialEq)]
pub struct JawTransform {
    neutral_pose: Vec<f32>,
}

impl JawTransform {
    pub fn new(neutral_pose: Vec<f32>) -> Result<Self> {
        validate_pose("neutral jaw", &neutral_pose)?;
        Ok(Self { neutral_pose })
    }

    pub fn neutral_pose(&self) -> &[f32] {
        &self.neutral_pose
    }

    /// Produces the column-major 4x4 transform used by the SDK result contract.
    pub fn compute(&self, deltas: &[f32], parameters: JawParameters) -> Result<[f32; 16]> {
        if deltas.len() != self.neutral_pose.len() {
            return Err(Audio2xError::InvalidSchema(format!(
                "jaw deltas have {} elements, expected {}",
                deltas.len(),
                self.neutral_pose.len()
            )));
        }
        if [
            parameters.strength,
            parameters.height_offset,
            parameters.depth_offset,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(Audio2xError::InvalidSchema(
                "jaw parameters must be finite".into(),
            ));
        }
        let mut target = Vec::with_capacity(deltas.len());
        for (neutral, delta) in self
            .neutral_pose
            .chunks_exact(3)
            .zip(deltas.chunks_exact(3))
        {
            target.extend([
                neutral[0] + delta[0] * parameters.strength,
                neutral[1] + delta[1] * parameters.strength + parameters.height_offset,
                neutral[2] + delta[2] * parameters.strength + parameters.depth_offset,
            ]);
        }
        rigid_transform(&target, &self.neutral_pose)
    }
}

/// Finds the proper rigid transform mapping `from_pose` onto `to_pose`.
pub fn rigid_transform(to_pose: &[f32], from_pose: &[f32]) -> Result<[f32; 16]> {
    validate_pose("target pose", to_pose)?;
    if from_pose.len() != to_pose.len() {
        return Err(Audio2xError::InvalidSchema(format!(
            "source pose has {} elements, expected {}",
            from_pose.len(),
            to_pose.len()
        )));
    }
    validate_pose("source pose", from_pose)?;
    let count = to_pose.len() / 3;
    let mean = |pose: &[f32]| {
        let mut result = [0.0_f64; 3];
        for point in pose.chunks_exact(3) {
            for component in 0..3 {
                result[component] += f64::from(point[component]);
            }
        }
        for value in &mut result {
            *value /= count as f64;
        }
        result
    };
    let target_mean = mean(to_pose);
    let source_mean = mean(from_pose);
    let mut s = [[0.0_f64; 3]; 3];
    for (source, target) in from_pose.chunks_exact(3).zip(to_pose.chunks_exact(3)) {
        for row in 0..3 {
            for column in 0..3 {
                s[row][column] += (f64::from(source[row]) - source_mean[row])
                    * (f64::from(target[column]) - target_mean[column]);
            }
        }
    }
    let trace = s[0][0] + s[1][1] + s[2][2];
    let n = [
        [
            trace,
            s[1][2] - s[2][1],
            s[2][0] - s[0][2],
            s[0][1] - s[1][0],
        ],
        [
            s[1][2] - s[2][1],
            s[0][0] - s[1][1] - s[2][2],
            s[0][1] + s[1][0],
            s[0][2] + s[2][0],
        ],
        [
            s[2][0] - s[0][2],
            s[0][1] + s[1][0],
            -s[0][0] + s[1][1] - s[2][2],
            s[1][2] + s[2][1],
        ],
        [
            s[0][1] - s[1][0],
            s[0][2] + s[2][0],
            s[1][2] + s[2][1],
            -s[0][0] - s[1][1] + s[2][2],
        ],
    ];
    let [w, x, y, z] = largest_eigenvector(n);
    let rotation = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut translation = target_mean;
    for row in 0..3 {
        translation[row] -= rotation[row][0] * source_mean[0]
            + rotation[row][1] * source_mean[1]
            + rotation[row][2] * source_mean[2];
    }
    let mut output = [0.0_f32; 16];
    output[15] = 1.0;
    for row in 0..3 {
        for column in 0..3 {
            output[column * 4 + row] = rotation[row][column] as f32;
        }
        output[12 + row] = translation[row] as f32;
    }
    Ok(output)
}

fn validate_pose(name: &str, pose: &[f32]) -> Result<()> {
    if pose.is_empty() || !pose.len().is_multiple_of(3) {
        return Err(Audio2xError::InvalidSchema(format!(
            "{name} size must be a non-zero multiple of 3"
        )));
    }
    if pose.iter().any(|value| !value.is_finite()) {
        return Err(Audio2xError::InvalidSchema(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

fn largest_eigenvector(mut matrix: [[f64; 4]; 4]) -> [f64; 4] {
    let mut vectors = [[0.0_f64; 4]; 4];
    for (index, row) in vectors.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    for _ in 0..32 {
        for p in 0..3 {
            for q in (p + 1)..4 {
                if matrix[p][q].abs() <= f64::EPSILON {
                    continue;
                }
                let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
                let (sine, cosine) = angle.sin_cos();
                let mut row = 0;
                while row < 4 {
                    let (a, b) = (matrix[row][p], matrix[row][q]);
                    matrix[row][p] = cosine * a - sine * b;
                    matrix[row][q] = sine * a + cosine * b;
                    row += 1;
                }
                let mut column = 0;
                while column < 4 {
                    let (a, b) = (matrix[p][column], matrix[q][column]);
                    matrix[p][column] = cosine * a - sine * b;
                    matrix[q][column] = sine * a + cosine * b;
                    column += 1;
                }
                for row in &mut vectors {
                    let (a, b) = (row[p], row[q]);
                    row[p] = cosine * a - sine * b;
                    row[q] = sine * a + cosine * b;
                }
            }
        }
    }
    let largest = (1..4).fold(0, |best, index| {
        if matrix[index][index] > matrix[best][best] {
            index
        } else {
            best
        }
    });
    [
        vectors[0][largest],
        vectors[1][largest],
        vectors[2][largest],
        vectors[3][largest],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    fn close(actual: &[f32], expected: &[f32]) {
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < 1.0e-5, "element {index}: {a} != {e}");
        }
    }
    #[test]
    fn applies_strength_and_offsets_before_fitting() {
        let jaw = JawTransform::new(vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
        let result = jaw
            .compute(
                &[2., 0., 0., 2., 0., 0., 2., 0., 0.],
                JawParameters {
                    strength: 0.5,
                    height_offset: 2.,
                    depth_offset: -3.,
                },
            )
            .unwrap();
        close(
            &result,
            &[
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 2., -3., 1.,
            ],
        );
    }
    #[test]
    fn recovers_rotation_and_translation_in_column_major_order() {
        let source = [0., 0., 0., 2., 0., 0., 0., 1., 0., 0., 0., 3.];
        let target: Vec<f32> = source
            .chunks_exact(3)
            .flat_map(|p| [-p[1] + 4., p[0] - 2., p[2] + 1.])
            .collect();
        let result = rigid_transform(&target, &source).unwrap();
        close(
            &result,
            &[
                0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 4., -2., 1., 1.,
            ],
        );
    }
    #[test]
    fn rejects_invalid_inputs() {
        assert!(JawTransform::new(vec![]).is_err());
        let jaw = JawTransform::new(vec![0., 0., 0.]).unwrap();
        assert!(jaw.compute(&[], JawParameters::default()).is_err());
    }
}
