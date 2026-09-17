use crate::types::{Error, MediaTime, Result};
use std::collections::BTreeMap;

/// Sparse values retain the distinction between missing names and explicit zero.
pub type EmotionValues = BTreeMap<String, f32>;

#[derive(Clone, Debug, PartialEq)]
pub struct EmotionKeyframe {
    time: MediaTime,
    values: EmotionValues,
}
impl EmotionKeyframe {
    /// Output traces require finite values; input imposes the stricter 0..=1 range.
    pub fn new(time: MediaTime, values: EmotionValues) -> Result<Self> {
        if values
            .iter()
            .any(|(name, value)| name.is_empty() || !value.is_finite())
        {
            return Err(Error::invalid(
                "emotion names/values must be non-empty/finite",
            ));
        }
        Ok(Self { time, values })
    }
    pub fn time(&self) -> MediaTime {
        self.time
    }
    pub fn values(&self) -> &EmotionValues {
        &self.values
    }
    pub fn validate_input(&self) -> Result<()> {
        if self
            .values
            .values()
            .any(|value| !(0.0..=1.0).contains(value))
        {
            return Err(Error::invalid("input emotion must be in 0..=1"));
        }
        Ok(())
    }
    pub fn into_parts(self) -> (MediaTime, EmotionValues) {
        (self.time, self.values)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmotionTrace {
    pub input: Vec<EmotionKeyframe>,
    pub mixed: Vec<EmotionKeyframe>,
    pub smoothed: Vec<EmotionKeyframe>,
}
