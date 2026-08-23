//! Shared errors, checked conversions, and tensor metadata.
//!
//! Library crates emit structured [`tracing`] events but never install a
//! subscriber. Applications own subscriber selection and filtering. Logs may
//! include operation names, device ordinals, tensor names, dtypes, and shapes;
//! device pointers, model contents, audio samples, and credentials are never
//! logged.

mod audio_accumulator;
mod emotion_accumulator;
mod error;
mod float_accumulator;
mod model;
mod tensor;

pub use audio_accumulator::AudioAccumulator;
pub use emotion_accumulator::{
    EmotionAccumulator, EmotionAccumulatorError, EmotionAccumulatorState, Timestamp,
};
pub use error::{Audio2xError, Result, checked_i32, checked_u32};
pub use float_accumulator::FloatAccumulator;
pub use model::{
    ConfigDocument, DiffusionAudioParameters, DiffusionParameters, EmotionAudioParameters,
    EmotionConfigRoot, EmotionNetwork, EmotionPostProcessingConfig, GeometryAudioParameters,
    GeometryConfig, GeometryConfigRoot, GeometryNetwork, GeometryParameters, ModelDataPaths,
    ModelDocument, MultiModelDocument, NetworkDocument, NetworkId, RegressionAudioParameters,
    RegressionParameters, SingleModelDocument, load_config, load_model, load_network, parse_config,
    parse_model, parse_network,
};
pub use tensor::{Binding, BindingSchema, Dimension, ElementType, IoMode, Shape};
