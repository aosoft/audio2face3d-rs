//! Thread-safe streaming accumulator for single-channel `f32` samples.
//!
//! The CUDA implementation in the original SDK stores the same logical stream
//! in pooled device tensors. This core implementation keeps the ownership and
//! timestamp contract independent of CUDA; GPU frontends can use the same
//! state machine while replacing the storage/copy operations.

use crate::{Audio2xError, Result};
use std::collections::VecDeque;
use std::sync::Mutex;

#[derive(Debug)]
struct State {
    samples: VecDeque<f32>,
    next: usize,
    dropped: usize,
    closed: bool,
}

/// A thread-safe append-only sample stream with random reads and prefix drops.
#[derive(Debug)]
pub struct FloatAccumulator {
    tensor_size: usize,
    tensor_count: usize,
    state: Mutex<State>,
}

impl FloatAccumulator {
    /// Creates an accumulator. The sizing arguments mirror the SDK's pooled
    /// tensor configuration and are validated even though core storage grows
    /// as needed.
    pub fn new(tensor_size: usize, tensor_count: usize) -> Result<Self> {
        if tensor_size == 0 {
            return Err(Audio2xError::InvalidSchema(
                "accumulator tensor size must be non-zero".into(),
            ));
        }
        Ok(Self {
            tensor_size,
            tensor_count,
            state: Mutex::new(State {
                samples: VecDeque::with_capacity(tensor_size.saturating_mul(tensor_count)),
                next: 0,
                dropped: 0,
                closed: false,
            }),
        })
    }

    pub const fn tensor_size(&self) -> usize {
        self.tensor_size
    }
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    pub fn accumulate(&self, samples: &[f32]) -> Result<()> {
        if samples.is_empty() {
            return Err(Audio2xError::InvalidSchema("empty samples".into()));
        }
        let mut state = self.lock()?;
        if state.closed {
            return Err(Audio2xError::InvalidSchema("accumulator is closed".into()));
        }
        state.samples.extend(samples.iter().copied());
        state.next =
            state
                .next
                .checked_add(samples.len())
                .ok_or(Audio2xError::IntegerOverflow {
                    field: "accumulated_samples",
                    value: samples.len(),
                    target: "usize",
                })?;
        Ok(())
    }

    pub fn close(&self) -> Result<()> {
        let mut state = self.lock()?;
        if state.closed {
            return Err(Audio2xError::InvalidSchema(
                "accumulator is already closed".into(),
            ));
        }
        state.closed = true;
        Ok(())
    }

    /// Reads `length` samples at an absolute sample index, zero-padding only
    /// before the stream or after it once the accumulator is closed.
    pub fn read(&self, start: i64, length: usize) -> Result<Vec<f32>> {
        if length == 0 {
            return Err(Audio2xError::InvalidSchema("empty destination".into()));
        }
        let state = self.lock()?;
        let end = start
            .checked_add(
                i64::try_from(length).map_err(|_| Audio2xError::IntegerOverflow {
                    field: "read_length",
                    value: length,
                    target: "i64",
                })?,
            )
            .ok_or(Audio2xError::InvalidSchema("read range overflow".into()))?;
        let first_non_padding = start.max(0);
        if end > 0 && first_non_padding < i64::try_from(state.dropped).unwrap_or(i64::MAX) {
            return Err(Audio2xError::InvalidSchema(
                "trying to read discarded samples".into(),
            ));
        }
        if !state.closed && end > i64::try_from(state.next).unwrap_or(i64::MAX) {
            return Err(Audio2xError::InvalidSchema(
                "read exceeds accumulated samples".into(),
            ));
        }
        let mut output = vec![0.0; length];
        for (offset, value) in output.iter_mut().enumerate() {
            let absolute = start + i64::try_from(offset).unwrap_or(i64::MAX);
            if absolute < i64::try_from(state.dropped).unwrap_or(i64::MAX)
                || absolute < 0
                || absolute >= i64::try_from(state.next).unwrap_or(i64::MAX)
            {
                continue;
            }
            let index = usize::try_from(absolute).unwrap() - state.dropped;
            if let Some(sample) = state.samples.get(index) {
                *value = *sample;
            }
        }
        Ok(output)
    }

    pub fn drop_samples_before(&self, start: usize) -> Result<()> {
        let mut state = self.lock()?;
        if start > state.next {
            return Err(Audio2xError::InvalidSchema(
                "cannot drop future samples".into(),
            ));
        }
        let requested = start.saturating_sub(state.dropped);
        let count = (requested / self.tensor_size)
            .saturating_mul(self.tensor_size)
            .min(state.samples.len());
        state.samples.drain(..count);
        state.dropped = state.dropped.saturating_add(count);
        Ok(())
    }

    pub fn nb_accumulated_samples(&self) -> usize {
        self.lock().map(|s| s.next).unwrap_or(0)
    }
    pub fn nb_dropped_samples(&self) -> usize {
        self.lock().map(|s| s.dropped).unwrap_or(0)
    }
    pub fn is_closed(&self) -> bool {
        self.lock().map(|s| s.closed).unwrap_or(false)
    }

    pub fn reset(&self) -> Result<()> {
        let mut state = self.lock()?;
        state.samples.clear();
        state.next = 0;
        state.dropped = 0;
        state.closed = false;
        Ok(())
    }

    fn lock(&self) -> Result<std::sync::MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| Audio2xError::InvalidSchema("accumulator mutex poisoned".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn accumulates_reads_with_zero_padding_and_drop() {
        let acc = FloatAccumulator::new(4, 2).unwrap();
        acc.accumulate(&[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert_eq!(acc.read(-2, 4).unwrap(), vec![0.0, 0.0, 1.0, 2.0]);
        acc.drop_samples_before(4).unwrap();
        assert!(acc.read(0, 1).is_err());
        assert_eq!(acc.read(4, 1).unwrap(), vec![5.0]);
        assert_eq!(acc.nb_accumulated_samples(), 5);
        assert_eq!(acc.nb_dropped_samples(), 4);
    }

    #[test]
    fn drop_keeps_partial_tensor_and_zero_preallocation_is_valid() {
        let acc = FloatAccumulator::new(4, 0).unwrap();
        acc.accumulate(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        acc.drop_samples_before(3).unwrap();
        assert_eq!(acc.nb_dropped_samples(), 0);
        assert_eq!(acc.read(0, 4).unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn close_reset_and_thread_safety_contract() {
        let acc = FloatAccumulator::new(2, 1).unwrap();
        acc.accumulate(&[7.0]).unwrap();
        acc.close().unwrap();
        assert!(acc.accumulate(&[8.0]).is_err());
        assert_eq!(acc.read(1, 2).unwrap(), vec![0.0, 0.0]);
        assert!(acc.close().is_err());
        acc.reset().unwrap();
        assert!(!acc.is_closed());
        assert_eq!(acc.nb_accumulated_samples(), 0);
    }

    #[test]
    fn concurrent_accumulate_preserves_every_sample() {
        let accumulator = Arc::new(FloatAccumulator::new(8, 2).unwrap());
        let threads: Vec<_> = (0..4)
            .map(|value| {
                let accumulator = Arc::clone(&accumulator);
                std::thread::spawn(move || accumulator.accumulate(&[value as f32; 8]).unwrap())
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(accumulator.nb_accumulated_samples(), 32);
        let samples = accumulator.read(0, 32).unwrap();
        for value in 0..4 {
            assert_eq!(
                samples
                    .iter()
                    .filter(|sample| **sample == value as f32)
                    .count(),
                8
            );
        }
    }
}
