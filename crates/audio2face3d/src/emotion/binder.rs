use crate::common::{EmotionAccumulator, Error, Result};
use crate::emotion::EmotionCallbackMetadata;

pub struct EmotionBinder<'a> {
    accumulators: Vec<&'a EmotionAccumulator>,
    emotion_length: usize,
}

impl<'a> EmotionBinder<'a> {
    pub fn new(accumulators: Vec<&'a EmotionAccumulator>, emotion_length: usize) -> Result<Self> {
        if accumulators.is_empty()
            || emotion_length == 0
            || accumulators
                .iter()
                .any(|accumulator| accumulator.state().emotion_size != emotion_length)
        {
            return Err(Error::InvalidSchema(
                "emotion binder accumulator dimensions do not match".into(),
            ));
        }
        Ok(Self {
            accumulators,
            emotion_length,
        })
    }

    pub fn accumulate(&self, metadata: EmotionCallbackMetadata, emotions: &[f32]) -> Result<bool> {
        if emotions.len() != self.emotion_length {
            return Err(Error::InvalidSchema(
                "emotion binder callback dimensions do not match".into(),
            ));
        }
        self.accumulators
            .get(metadata.track)
            .ok_or_else(|| Error::InvalidSchema("emotion binder track is out of range".into()))?
            .accumulate(metadata.timestamp, emotions)
            .map_err(|error| {
                Error::InvalidSchema(format!("emotion binder accumulate failed: {error}"))
            })?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binds_callback_results_by_track_and_timestamp() {
        let first = EmotionAccumulator::new(2, 2).unwrap();
        let second = EmotionAccumulator::new(2, 2).unwrap();
        let binder = EmotionBinder::new(vec![&first, &second], 2).unwrap();
        binder
            .accumulate(
                EmotionCallbackMetadata {
                    track: 1,
                    frame: 0,
                    timestamp: 7,
                    next_timestamp: 8,
                },
                &[0.25, 0.75],
            )
            .unwrap();
        assert_eq!(second.read(7).unwrap(), [0.25, 0.75]);
        assert_eq!(first.state().key_count, 0);
    }

    #[test]
    fn propagates_accumulator_order_and_backpressure_errors() {
        let output = EmotionAccumulator::new(2, 1).unwrap();
        let binder = EmotionBinder::new(vec![&output], 2).unwrap();
        let metadata = |timestamp| EmotionCallbackMetadata {
            track: 0,
            frame: timestamp as usize,
            timestamp,
            next_timestamp: timestamp + 1,
        };
        assert!(binder.accumulate(metadata(1), &[0.0, 1.0]).unwrap());
        assert!(binder.accumulate(metadata(1), &[1.0, 0.0]).is_err());
        output.close().unwrap();
        assert!(binder.accumulate(metadata(2), &[1.0, 0.0]).is_err());
    }
}
