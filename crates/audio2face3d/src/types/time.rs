use crate::types::{Error, Result};

/// Exact position in sample frames (one sample per channel), not bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SamplePosition(pub u64);
impl SamplePosition {
    pub fn checked_add(self, frames: u64) -> Result<Self> {
        self.0
            .checked_add(frames)
            .map(Self)
            .ok_or_else(|| Error::invalid("sample position overflow"))
    }
}

/// Non-negative time relative to the start of one utterance, in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MediaTime(u64);
impl MediaTime {
    pub const ZERO: Self = Self(0);
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
    pub fn as_seconds(self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }
    /// Rounds to nearest nanosecond; rejects non-finite or out-of-range values.
    pub fn from_seconds(seconds: f64) -> Result<Self> {
        let nanos = (seconds * 1_000_000_000.0).round();
        if !seconds.is_finite() || seconds < 0.0 || nanos >= u64::MAX as f64 {
            return Err(Error::invalid(
                "time must be finite, non-negative and fit nanoseconds",
            ));
        }
        Ok(Self(nanos as u64))
    }
    pub fn from_samples(position: SamplePosition, sample_rate: u32) -> Result<Self> {
        if sample_rate == 0 {
            return Err(Error::invalid("sample rate must be positive"));
        }
        let rate = u128::from(sample_rate);
        let nanos = (u128::from(position.0) * 1_000_000_000 + rate / 2) / rate;
        u64::try_from(nanos)
            .map(Self)
            .map_err(|_| Error::invalid("sample time overflow"))
    }
    /// Rounds only this absolute position, never an accumulated duration.
    pub fn nearest_sample(self, sample_rate: u32) -> Result<SamplePosition> {
        if sample_rate == 0 {
            return Err(Error::invalid("sample rate must be positive"));
        }
        let samples = (u128::from(self.0) * u128::from(sample_rate) + 500_000_000) / 1_000_000_000;
        u64::try_from(samples)
            .map(SamplePosition)
            .map_err(|_| Error::invalid("sample position overflow"))
    }
}
