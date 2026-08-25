//! Audio-specific view over the pooled float accumulator.

use crate::common::{FloatAccumulator, Result};

/// Thread-safe indexed audio stream with input-strength scaling.
#[derive(Debug)]
pub struct AudioAccumulator {
    inner: FloatAccumulator,
}

impl AudioAccumulator {
    pub fn new(tensor_size: usize, tensor_count: usize) -> Result<Self> {
        Ok(Self {
            inner: FloatAccumulator::new(tensor_size, tensor_count)?,
        })
    }

    pub fn accumulate(&self, samples: &[f32]) -> Result<()> {
        self.inner.accumulate(samples)
    }
    pub fn close(&self) -> Result<()> {
        self.inner.close()
    }
    pub fn drop_samples_before(&self, start: usize) -> Result<()> {
        self.inner.drop_samples_before(start)
    }

    pub fn read(&self, start: i64, length: usize, input_strength: f32) -> Result<Vec<f32>> {
        let mut output = self.inner.read(start, length)?;
        output
            .iter_mut()
            .for_each(|sample| *sample *= input_strength);
        Ok(output)
    }

    pub fn nb_accumulated_samples(&self) -> usize {
        self.inner.nb_accumulated_samples()
    }
    pub fn nb_dropped_samples(&self) -> usize {
        self.inner.nb_dropped_samples()
    }
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }
    pub fn reset(&self) -> Result<()> {
        self.inner.reset()
    }
}

#[cfg(test)]
mod tests {
    use super::AudioAccumulator;

    #[test]
    fn strength_padding_and_pooled_drop() {
        let accumulator = AudioAccumulator::new(2, 1).unwrap();
        accumulator.accumulate(&[1.0, -2.0, 3.0]).unwrap();
        assert!(accumulator.read(2, 2, 1.0).is_err());
        accumulator.drop_samples_before(1).unwrap();
        assert_eq!(accumulator.nb_dropped_samples(), 0);
        accumulator.drop_samples_before(2).unwrap();
        assert!(accumulator.read(0, 1, 1.0).is_err());
        accumulator.close().unwrap();
        assert_eq!(accumulator.read(2, 3, 2.5).unwrap(), [7.5, 0.0, 0.0]);
        accumulator.reset().unwrap();
        assert_eq!(accumulator.nb_accumulated_samples(), 0);
        assert!(!accumulator.is_closed());
    }
}
