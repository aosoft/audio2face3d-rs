use crate::{Audio2xError, Result, Timestamp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowProgressParameters {
    pub window_size: usize,
    pub start_offset: Timestamp,
    pub target_offset: Timestamp,
    pub stride_numerator: usize,
    pub stride_denominator: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleWindow {
    pub start: Timestamp,
    pub target: Timestamp,
    pub end: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowProgress {
    parameters: WindowProgressParameters,
    read_window_count: usize,
}

impl WindowProgress {
    pub fn new(mut parameters: WindowProgressParameters) -> Result<Self> {
        if parameters.window_size == 0
            || parameters.stride_numerator == 0
            || parameters.stride_denominator == 0
        {
            return Err(Audio2xError::InvalidSchema(
                "window and stride dimensions must be non-zero".into(),
            ));
        }
        let divisor = gcd(parameters.stride_numerator, parameters.stride_denominator);
        parameters.stride_numerator /= divisor;
        parameters.stride_denominator /= divisor;
        Ok(Self {
            parameters,
            read_window_count: 0,
        })
    }

    pub const fn parameters(&self) -> WindowProgressParameters {
        self.parameters
    }
    pub const fn read_window_count(&self) -> usize {
        self.read_window_count
    }
    pub fn reset(&mut self, count: usize) {
        self.read_window_count = count;
    }
    pub fn advance(&mut self, count: usize) -> Result<()> {
        self.read_window_count =
            self.read_window_count
                .checked_add(count)
                .ok_or(Audio2xError::IntegerOverflow {
                    field: "read_window_count",
                    value: count,
                    target: "usize",
                })?;
        Ok(())
    }

    pub fn current_window(&self, offset: usize) -> Result<SampleWindow> {
        let index =
            self.read_window_count
                .checked_add(offset)
                .ok_or(Audio2xError::IntegerOverflow {
                    field: "window_index",
                    value: offset,
                    target: "usize",
                })?;
        self.window(index)
    }

    pub fn window(&self, index: usize) -> Result<SampleWindow> {
        let scaled = index.checked_mul(self.parameters.stride_numerator).ok_or(
            Audio2xError::IntegerOverflow {
                field: "window_stride",
                value: index,
                target: "usize",
            },
        )? / self.parameters.stride_denominator;
        let start = self
            .parameters
            .start_offset
            .checked_add(
                i64::try_from(scaled).map_err(|_| Audio2xError::IntegerOverflow {
                    field: "window_start",
                    value: scaled,
                    target: "i64",
                })?,
            )
            .ok_or_else(|| Audio2xError::InvalidSchema("window start overflow".into()))?;
        let target = start
            .checked_add(self.parameters.target_offset)
            .ok_or_else(|| Audio2xError::InvalidSchema("window target overflow".into()))?;
        let end = start
            .checked_add(i64::try_from(self.parameters.window_size).map_err(|_| {
                Audio2xError::IntegerOverflow {
                    field: "window_size",
                    value: self.parameters.window_size,
                    target: "i64",
                }
            })?)
            .ok_or_else(|| Audio2xError::InvalidSchema("window end overflow".into()))?;
        Ok(SampleWindow { start, target, end })
    }

    pub fn available_windows(&self, end_timestamp: Timestamp, closed: bool) -> Result<usize> {
        let additional = if closed {
            self.parameters.target_offset
        } else {
            i64::try_from(self.parameters.window_size).map_err(|_| {
                Audio2xError::IntegerOverflow {
                    field: "window_size",
                    value: self.parameters.window_size,
                    target: "i64",
                }
            })?
        };
        let limit = if closed { 0 } else { -1 };
        let offset = self
            .parameters
            .start_offset
            .checked_add(additional)
            .ok_or_else(|| Audio2xError::InvalidSchema("window offset overflow".into()))?;
        let denominator = i64::try_from(self.parameters.stride_denominator).map_err(|_| {
            Audio2xError::IntegerOverflow {
                field: "stride_denominator",
                value: self.parameters.stride_denominator,
                target: "i64",
            }
        })?;
        let stride = i64::try_from(self.parameters.stride_numerator).map_err(|_| {
            Audio2xError::IntegerOverflow {
                field: "stride_numerator",
                value: self.parameters.stride_numerator,
                target: "i64",
            }
        })?;
        let distance = end_timestamp
            .checked_mul(denominator)
            .and_then(|size| {
                offset
                    .checked_mul(denominator)
                    .and_then(|scaled| size.checked_sub(scaled))
            })
            .ok_or_else(|| Audio2xError::InvalidSchema("window availability overflow".into()))?;
        let mut count = div_ceil_signed(distance, stride);
        let position = offset
            .checked_add(
                count
                    .checked_mul(stride)
                    .ok_or_else(|| Audio2xError::InvalidSchema("window count overflow".into()))?
                    / denominator,
            )
            .and_then(|value| value.checked_add(limit))
            .ok_or_else(|| Audio2xError::InvalidSchema("window limit overflow".into()))?;
        if position >= end_timestamp {
            count -= 1;
        }
        Ok(count
            .saturating_sub(i64::try_from(self.read_window_count).unwrap_or(i64::MAX))
            .saturating_add(1) as usize)
    }
}

const fn gcd(mut left: usize, mut right: usize) -> usize {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}
fn div_ceil_signed(value: i64, divisor: i64) -> i64 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder > 0 {
        quotient + 1
    } else {
        quotient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_window_uses_rational_stride() {
        let mut progress = WindowProgress::new(WindowProgressParameters {
            window_size: 8320,
            start_offset: -4160,
            target_offset: 4160,
            stride_numerator: 16000,
            stride_denominator: 30,
        })
        .unwrap();
        assert_eq!(
            progress.window(0).unwrap(),
            SampleWindow {
                start: -4160,
                target: 0,
                end: 4160
            }
        );
        assert_eq!(progress.window(1).unwrap().target, 533);
        assert_eq!(progress.window(2).unwrap().target, 1066);
        assert_eq!(progress.window(3).unwrap().target, 1600);
        assert_eq!(progress.available_windows(16000, true).unwrap(), 30);
        progress.advance(10).unwrap();
        assert_eq!(progress.available_windows(16000, true).unwrap(), 20);
    }
}
