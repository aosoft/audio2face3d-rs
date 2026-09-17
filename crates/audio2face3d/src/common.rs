//! Shared errors, checked conversions, and tensor metadata.
//!
//! Explicit Audio2Face3DContext injection routes structured [`tracing`] events
//! to the application Logger. Legacy constructors preserve the calling subscriber;
//! the library never installs a global subscriber. Logs may
//! include operation names, device ordinals, tensor names, dtypes, and shapes;
//! device pointers, model contents, audio samples, and credentials are never
//! logged.

mod audio_accumulator;
mod blendshape;
mod emotion_accumulator;
mod error;
mod float_accumulator;
mod model;
#[cfg(feature = "animation")]
mod npz;
mod tensor;
mod window_progress;

pub use audio_accumulator::AudioAccumulator;
pub use blendshape::{
    BlendshapeConfig, BlendshapeConfigRoot, load_blendshape_config, parse_blendshape_config,
};
pub use emotion_accumulator::{
    EmotionAccumulator, EmotionAccumulatorError, EmotionAccumulatorState, Timestamp,
};
pub use error::{Error, Result, checked_i32, checked_u32};
pub use float_accumulator::FloatAccumulator;
pub use model::{
    ConfigDocument, DiffusionAudioParameters, DiffusionParameters, EmotionAudioParameters,
    EmotionConfigRoot, EmotionNetwork, EmotionPostProcessingConfig, GeometryAudioParameters,
    GeometryConfig, GeometryConfigRoot, GeometryNetwork, GeometryParameters, ModelDataPaths,
    ModelDocument, MultiModelDocument, NetworkDocument, NetworkId, RegressionAudioParameters,
    RegressionParameters, SingleModelDocument, load_config, load_model, load_network, parse_config,
    parse_model, parse_network,
};
#[cfg(feature = "animation")]
pub(crate) use npz::NpzArchive;
pub use tensor::{Binding, BindingSchema, Dimension, ElementType, IoMode, Shape};
pub use window_progress::{SampleWindow, WindowProgress, WindowProgressParameters};
