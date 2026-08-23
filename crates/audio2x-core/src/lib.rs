//! Shared errors, checked conversions, and tensor metadata.
//!
//! Library crates emit structured [`tracing`] events but never install a
//! subscriber. Applications own subscriber selection and filtering. Logs may
//! include operation names, device ordinals, tensor names, dtypes, and shapes;
//! device pointers, model contents, audio samples, and credentials are never
//! logged.

mod error;
mod tensor;

pub use error::{Audio2xError, Result, checked_i32, checked_u32};
pub use tensor::{Binding, BindingSchema, Dimension, ElementType, IoMode, Shape};
