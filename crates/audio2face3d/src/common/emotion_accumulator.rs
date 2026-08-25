use std::collections::VecDeque;
use std::sync::Mutex;
use thiserror::Error;

pub type Timestamp = i64;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EmotionAccumulatorError {
    #[error("emotion size and emotions per buffer must be non-zero")]
    InvalidConfiguration,
    #[error("emotion accumulator is empty")]
    Empty,
    #[error("emotion accumulator is closed")]
    Closed,
    #[error("timestamp {actual} must be greater than {previous}")]
    TimestampOrder {
        previous: Timestamp,
        actual: Timestamp,
    },
    #[error("emotion has {actual} values, expected {expected}")]
    SizeMismatch { expected: usize, actual: usize },
    #[error("timestamp {requested} is outside available range")]
    OutOfBounds { requested: Timestamp },
    #[error("emotion count overflow")]
    IntegerOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmotionAccumulatorState {
    pub emotion_size: usize,
    pub key_count: usize,
    pub last_accumulated_timestamp: Timestamp,
    pub last_dropped_timestamp: Timestamp,
    pub dropped_emotions: usize,
    pub closed: bool,
}

#[derive(Debug)]
struct State {
    timestamps: VecDeque<Timestamp>,
    values: VecDeque<Vec<f32>>,
    last_accumulated: Timestamp,
    last_dropped: Timestamp,
    dropped: usize,
    closed: bool,
}

/// Thread-safe timestamped emotion storage with chunked pool-drop semantics.
#[derive(Debug)]
pub struct EmotionAccumulator {
    emotion_size: usize,
    emotions_per_buffer: usize,
    state: Mutex<State>,
}

impl EmotionAccumulator {
    pub fn new(
        emotion_size: usize,
        emotions_per_buffer: usize,
    ) -> Result<Self, EmotionAccumulatorError> {
        if emotion_size == 0 || emotions_per_buffer == 0 {
            return Err(EmotionAccumulatorError::InvalidConfiguration);
        }
        Ok(Self {
            emotion_size,
            emotions_per_buffer,
            state: Mutex::new(State::new()),
        })
    }

    pub fn accumulate(
        &self,
        timestamp: Timestamp,
        emotion: &[f32],
    ) -> Result<(), EmotionAccumulatorError> {
        if emotion.len() != self.emotion_size {
            return Err(EmotionAccumulatorError::SizeMismatch {
                expected: self.emotion_size,
                actual: emotion.len(),
            });
        }
        let mut state = self
            .state
            .lock()
            .expect("emotion accumulator mutex poisoned");
        if state.closed {
            return Err(EmotionAccumulatorError::Closed);
        }
        if timestamp <= state.last_accumulated {
            return Err(EmotionAccumulatorError::TimestampOrder {
                previous: state.last_accumulated,
                actual: timestamp,
            });
        }
        state.timestamps.push_back(timestamp);
        state.values.push_back(emotion.to_vec());
        state.last_accumulated = timestamp;
        Ok(())
    }

    pub fn read(&self, timestamp: Timestamp) -> Result<Vec<f32>, EmotionAccumulatorError> {
        let state = self
            .state
            .lock()
            .expect("emotion accumulator mutex poisoned");
        if state.timestamps.is_empty() {
            return Err(EmotionAccumulatorError::Empty);
        }
        if timestamp < state.last_dropped || (!state.closed && timestamp > state.last_accumulated) {
            return Err(EmotionAccumulatorError::OutOfBounds {
                requested: timestamp,
            });
        }
        let after = state
            .timestamps
            .partition_point(|candidate| *candidate < timestamp);
        if after == 0 {
            return Ok(state.values[0].clone());
        }
        if after == state.timestamps.len() {
            return Ok(state.values[state.values.len() - 1].clone());
        }
        if state.timestamps[after] == timestamp {
            return Ok(state.values[after].clone());
        }
        let before = after - 1;
        let factor = (timestamp - state.timestamps[before]) as f32
            / (state.timestamps[after] - state.timestamps[before]) as f32;
        Ok(state.values[before]
            .iter()
            .zip(&state.values[after])
            .map(|(before, after)| before + (after - before) * factor)
            .collect())
    }

    pub fn drop_before(&self, timestamp: Timestamp) -> Result<(), EmotionAccumulatorError> {
        let mut state = self
            .state
            .lock()
            .expect("emotion accumulator mutex poisoned");
        if !state.closed && timestamp > state.last_accumulated {
            return Err(EmotionAccumulatorError::OutOfBounds {
                requested: timestamp,
            });
        }
        let mut count: usize = 0;
        loop {
            let next = count
                .checked_add(self.emotions_per_buffer)
                .ok_or(EmotionAccumulatorError::IntegerOverflow)?;
            if next >= state.timestamps.len() || state.timestamps[next] > timestamp {
                break;
            }
            count = next;
        }
        state.timestamps.drain(..count);
        state.values.drain(..count);
        state.last_dropped = timestamp;
        state.dropped = state
            .dropped
            .checked_add(count)
            .ok_or(EmotionAccumulatorError::IntegerOverflow)?;
        Ok(())
    }

    pub fn close(&self) -> Result<(), EmotionAccumulatorError> {
        let mut state = self
            .state
            .lock()
            .expect("emotion accumulator mutex poisoned");
        if state.closed {
            return Err(EmotionAccumulatorError::Closed);
        }
        if state.timestamps.is_empty() {
            return Err(EmotionAccumulatorError::Empty);
        }
        state.closed = true;
        Ok(())
    }
    pub fn reset(&self) {
        *self
            .state
            .lock()
            .expect("emotion accumulator mutex poisoned") = State::new();
    }
    pub fn state(&self) -> EmotionAccumulatorState {
        let state = self
            .state
            .lock()
            .expect("emotion accumulator mutex poisoned");
        EmotionAccumulatorState {
            emotion_size: self.emotion_size,
            key_count: state.timestamps.len(),
            last_accumulated_timestamp: state.last_accumulated,
            last_dropped_timestamp: state.last_dropped,
            dropped_emotions: state.dropped,
            closed: state.closed,
        }
    }
}

impl State {
    fn new() -> Self {
        Self {
            timestamps: VecDeque::new(),
            values: VecDeque::new(),
            last_accumulated: Timestamp::MIN,
            last_dropped: Timestamp::MIN,
            dropped: 0,
            closed: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn interpolates_clamps_drops_and_resets() {
        let a = EmotionAccumulator::new(2, 2).unwrap();
        assert!(a.close().is_err());
        a.accumulate(1, &[1.0, 3.0]).unwrap();
        a.accumulate(4, &[4.0, 6.0]).unwrap();
        a.accumulate(5, &[5.0, 7.0]).unwrap();
        a.accumulate(7, &[7.0, 9.0]).unwrap();
        a.accumulate(9, &[9.0, 11.0]).unwrap();
        assert_eq!(a.read(3).unwrap(), vec![3.0, 5.0]);
        assert!(a.read(10).is_err());
        a.drop_before(5).unwrap();
        assert_eq!(a.state().dropped_emotions, 2);
        assert!(a.read(4).is_err());
        a.close().unwrap();
        assert_eq!(a.read(100).unwrap(), vec![9.0, 11.0]);
        a.reset();
        assert_eq!(a.state().key_count, 0);
    }

    #[test]
    fn rejects_order_and_supports_concurrent_access() {
        let a = Arc::new(EmotionAccumulator::new(1, 4).unwrap());
        let producer = Arc::clone(&a);
        std::thread::spawn(move || {
            for t in 0..100 {
                producer.accumulate(t, &[t as f32]).unwrap();
            }
            producer.close().unwrap();
        })
        .join()
        .unwrap();
        assert!(matches!(
            a.accumulate(99, &[0.0]),
            Err(EmotionAccumulatorError::Closed)
        ));
        for t in 0..110 {
            assert_eq!(a.read(t).unwrap()[0], t.min(99) as f32);
        }
    }
}
