use crate::common::{Error, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct PcaReconstruction {
    shape_size: usize,
    shape_count: usize,
    /// Column-major `[shape_size, shape_count]` matrix.
    shapes: Vec<f32>,
}

impl PcaReconstruction {
    pub fn new(shape_size: usize, shape_count: usize, shapes: Vec<f32>) -> Result<Self> {
        let expected = shape_size
            .checked_mul(shape_count)
            .ok_or(Error::IntegerOverflow {
                field: "pca_matrix_size",
                value: shape_count,
                target: "usize",
            })?;
        if shape_size == 0 || shape_count == 0 || shapes.len() != expected {
            return Err(Error::InvalidSchema(format!(
                "PCA matrix has {} elements, expected {expected}",
                shapes.len()
            )));
        }
        Ok(Self {
            shape_size,
            shape_count,
            shapes,
        })
    }

    pub const fn shape_size(&self) -> usize {
        self.shape_size
    }
    pub const fn shape_count(&self) -> usize {
        self.shape_count
    }
    /// CPU parity oracle for the cuBLAS column-major SGEMV/SGEMM path.
    pub fn reconstruct(&self, coefficients: &[f32], batch_size: usize) -> Result<Vec<f32>> {
        let coefficient_count =
            self.shape_count
                .checked_mul(batch_size)
                .ok_or(Error::IntegerOverflow {
                    field: "pca_coefficient_count",
                    value: batch_size,
                    target: "usize",
                })?;
        if batch_size == 0 || coefficients.len() != coefficient_count {
            return Err(Error::InvalidSchema(format!(
                "PCA coefficients have {} elements, expected {coefficient_count}",
                coefficients.len()
            )));
        }
        let mut output = vec![0.0; self.shape_size * batch_size];
        for batch in 0..batch_size {
            for shape in 0..self.shape_count {
                let coefficient = coefficients[batch * self.shape_count + shape];
                let matrix = &self.shapes[shape * self.shape_size..(shape + 1) * self.shape_size];
                let result = &mut output[batch * self.shape_size..(batch + 1) * self.shape_size];
                for (value, basis) in result.iter_mut().zip(matrix) {
                    *value += basis * coefficient;
                }
            }
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstructs_single_and_batched_coefficients() {
        let pca = PcaReconstruction::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(pca.reconstruct(&[1.0, 2.0], 1).unwrap(), [9.0, 12.0, 15.0]);
        assert_eq!(
            pca.reconstruct(&[1.0, 2.0, 3.0, 4.0], 2).unwrap(),
            [9.0, 12.0, 15.0, 19.0, 26.0, 33.0]
        );
    }
}
