//! Audio2Face pipeline components.

mod pca;
mod regression;

pub use pca::PcaReconstruction;

pub use regression::{
    RegressionContract, RegressionFrameInput, RegressionResultLayout, RegressionResultSlices,
};

pub const PIPELINE_NAME: &str = "audio2face";
