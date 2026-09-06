//! Closed, non-generic owning bundles for completed geometry facades.
//!
//! The public surface exposes only completed model-specific executors and
//! never a backend or post-processor type.

use std::sync::Arc;

use crate::Result;
use crate::audio2face::diffusion::{
    DiffusionGeometryExecutor, DiffusionGeometryExecutorCreationParameters,
    DiffusionGeometryExecutorFactory,
};
use crate::audio2face::diffusion::{
    DiffusionGeometryInteractiveExecutor, DiffusionGeometryInteractiveExecutorCreationParameters,
    DiffusionGeometryInteractiveExecutorFactory,
};
use crate::audio2face::regression::{
    RegressionGeometryExecutor, RegressionGeometryExecutorCreationParameters,
    RegressionGeometryExecutorFactory, RegressionGeometryInteractiveExecutor,
    RegressionGeometryInteractiveExecutorCreationParameters,
    RegressionGeometryInteractiveExecutorFactory,
};
use crate::audio2face::{
    DeviceBlendshapeSolveExecutor, DeviceBlendshapeSolveExecutorCreationParameters,
    HostBlendshapeSolveExecutor, HostBlendshapeSolveExecutorCreationParameters,
};
use crate::audio2x::{AudioAccumulator, EmotionAccumulator, ExecutorFuture};
use crate::cuda::CudaStream;

/// Completed geometry executor borrowed from an owning bundle.
pub enum GeometryExecutorRef<'a> {
    Regression(&'a RegressionGeometryExecutor),
    Diffusion(&'a DiffusionGeometryExecutor),
}

/// Mutably borrowed completed geometry executor.
pub enum GeometryExecutorMut<'a> {
    Regression(&'a mut RegressionGeometryExecutor),
    Diffusion(&'a mut DiffusionGeometryExecutor),
}

/// Model-specific parameters accepted by [`GeometryExecutorBundleFactory::load`].
pub enum GeometryExecutorBundleCreationParameters {
    Regression(RegressionGeometryExecutorCreationParameters),
    Diffusion(DiffusionGeometryExecutorCreationParameters),
}

/// A non-generic owning geometry bundle.
///
/// The enum is intentionally closed. It keeps the model-specific executor
/// concrete while preventing backend implementation types from appearing in
/// the public creation or accessor API.
pub enum GeometryExecutorBundle {
    Regression(RegressionGeometryExecutor),
    Diffusion(DiffusionGeometryExecutor),
}

pub enum InteractiveGeometryExecutorRef<'a> {
    Regression(&'a RegressionGeometryInteractiveExecutor),
    Diffusion(&'a DiffusionGeometryInteractiveExecutor),
}

pub enum InteractiveGeometryExecutorMut<'a> {
    Regression(&'a mut RegressionGeometryInteractiveExecutor),
    Diffusion(&'a mut DiffusionGeometryInteractiveExecutor),
}

pub enum InteractiveGeometryBundleCreationParameters {
    Regression(RegressionGeometryInteractiveExecutorCreationParameters),
    Diffusion(DiffusionGeometryInteractiveExecutorCreationParameters),
}

/// Closed, non-generic interactive geometry bundle backed by completed
/// model-specific facades.
pub enum InteractiveGeometryExecutorBundle {
    // Both completed interactive executors exceed one KiB. Indirection keeps
    // runtime model selection from inheriting either executor's stack size.
    Regression(Box<RegressionGeometryInteractiveExecutor>),
    Diffusion(Box<DiffusionGeometryInteractiveExecutor>),
}

impl InteractiveGeometryExecutorBundle {
    pub fn executor(&self) -> InteractiveGeometryExecutorRef<'_> {
        match self {
            Self::Regression(executor) => InteractiveGeometryExecutorRef::Regression(executor),
            Self::Diffusion(executor) => InteractiveGeometryExecutorRef::Diffusion(executor),
        }
    }

    pub fn executor_mut(&mut self) -> InteractiveGeometryExecutorMut<'_> {
        match self {
            Self::Regression(executor) => InteractiveGeometryExecutorMut::Regression(executor),
            Self::Diffusion(executor) => InteractiveGeometryExecutorMut::Diffusion(executor),
        }
    }

    pub fn cuda_stream(&self) -> &CudaStream {
        match self {
            Self::Regression(executor) => executor.cuda_stream(),
            Self::Diffusion(executor) => executor.cuda_stream(),
        }
    }

    pub fn audio_accumulator(&self) -> &Arc<AudioAccumulator> {
        match self {
            Self::Regression(executor) => executor.audio_accumulator(),
            Self::Diffusion(executor) => executor.audio_accumulator(),
        }
    }

    pub fn emotion_accumulator(&self) -> &Arc<EmotionAccumulator> {
        match self {
            Self::Regression(executor) => executor.emotion_accumulator(),
            Self::Diffusion(executor) => executor.emotion_accumulator(),
        }
    }
}

pub struct InteractiveGeometryExecutorBundleFactory;

impl InteractiveGeometryExecutorBundleFactory {
    pub fn load(
        parameters: InteractiveGeometryBundleCreationParameters,
    ) -> ExecutorFuture<'static, InteractiveGeometryExecutorBundle> {
        Box::pin(async move {
            match parameters {
                InteractiveGeometryBundleCreationParameters::Regression(parameters) => {
                    RegressionGeometryInteractiveExecutorFactory::load(parameters)
                        .await
                        .map(Box::new)
                        .map(InteractiveGeometryExecutorBundle::Regression)
                }
                InteractiveGeometryBundleCreationParameters::Diffusion(parameters) => {
                    DiffusionGeometryInteractiveExecutorFactory::load(parameters)
                        .await
                        .map(Box::new)
                        .map(InteractiveGeometryExecutorBundle::Diffusion)
                }
            }
        })
    }
}

/// Completed BlendShape executor borrowed from an owning bundle.
pub enum BlendshapeExecutorRef<'a> {
    Host(&'a HostBlendshapeSolveExecutor),
    Device(&'a DeviceBlendshapeSolveExecutor),
}

/// Mutably borrowed completed BlendShape executor.
pub enum BlendshapeExecutorMut<'a> {
    Host(&'a mut HostBlendshapeSolveExecutor),
    Device(&'a mut DeviceBlendshapeSolveExecutor),
}

/// A closed, non-generic owning BlendShape bundle.
pub enum BlendshapeExecutorBundle {
    Host(HostBlendshapeSolveExecutor),
    Device(DeviceBlendshapeSolveExecutor),
}

impl GeometryExecutorBundle {
    /// Borrows the completed executor without exposing its backend.
    pub fn executor(&self) -> GeometryExecutorRef<'_> {
        match self {
            Self::Regression(executor) => GeometryExecutorRef::Regression(executor),
            Self::Diffusion(executor) => GeometryExecutorRef::Diffusion(executor),
        }
    }

    /// Mutably borrows the completed executor without exposing its backend.
    pub fn executor_mut(&mut self) -> GeometryExecutorMut<'_> {
        match self {
            Self::Regression(executor) => GeometryExecutorMut::Regression(executor),
            Self::Diffusion(executor) => GeometryExecutorMut::Diffusion(executor),
        }
    }

    /// Returns the result stream owned by the completed executor.
    pub fn cuda_stream(&self) -> &CudaStream {
        match self {
            Self::Regression(executor) => executor.cuda_stream(),
            Self::Diffusion(executor) => executor.cuda_stream(),
        }
    }

    /// Returns the exact shared audio accumulator injected into a track.
    pub fn audio_accumulator(&self, track: usize) -> Result<&Arc<AudioAccumulator>> {
        match self {
            Self::Regression(executor) => executor.audio_accumulator(track),
            Self::Diffusion(executor) => executor.audio_accumulator(track),
        }
    }

    /// Returns the exact shared emotion accumulator injected into a track.
    pub fn emotion_accumulator(&self, track: usize) -> Result<&Arc<EmotionAccumulator>> {
        match self {
            Self::Regression(executor) => executor.emotion_accumulator(track),
            Self::Diffusion(executor) => executor.emotion_accumulator(track),
        }
    }

    pub fn track_count(&self) -> usize {
        match self {
            Self::Regression(executor) => executor.track_count(),
            Self::Diffusion(executor) => executor.track_count(),
        }
    }

    /// Consumes geometry and installs host BlendShape solvers.
    ///
    /// Failure returns this exact bundle, including all model, stream, and
    /// accumulator resources, so the caller can correct parameters and retry.
    #[allow(clippy::result_large_err)]
    pub fn try_into_host_blendshape(
        self,
        parameters: HostBlendshapeSolveExecutorCreationParameters<'_>,
    ) -> std::result::Result<HostBlendshapeSolveExecutor, crate::audio2x::TransferError<Self>> {
        match self {
            Self::Regression(executor) => {
                executor
                    .try_into_host_blendshape(parameters)
                    .map_err(|failure| crate::audio2x::TransferError {
                        error: failure.error,
                        original: Self::Regression(failure.original),
                    })
            }
            Self::Diffusion(executor) => {
                executor
                    .try_into_host_blendshape(parameters)
                    .map_err(|failure| crate::audio2x::TransferError {
                        error: failure.error,
                        original: Self::Diffusion(failure.original),
                    })
            }
        }
    }

    /// Consumes geometry and installs device BlendShape solvers.
    #[allow(clippy::result_large_err)]
    pub fn try_into_device_blendshape(
        self,
        parameters: DeviceBlendshapeSolveExecutorCreationParameters<'_>,
    ) -> std::result::Result<DeviceBlendshapeSolveExecutor, crate::audio2x::TransferError<Self>>
    {
        match self {
            Self::Regression(executor) => {
                executor
                    .try_into_device_blendshape(parameters)
                    .map_err(|failure| crate::audio2x::TransferError {
                        error: failure.error,
                        original: Self::Regression(failure.original),
                    })
            }
            Self::Diffusion(executor) => {
                executor
                    .try_into_device_blendshape(parameters)
                    .map_err(|failure| crate::audio2x::TransferError {
                        error: failure.error,
                        original: Self::Diffusion(failure.original),
                    })
            }
        }
    }
}

impl BlendshapeExecutorBundle {
    pub fn executor(&self) -> BlendshapeExecutorRef<'_> {
        match self {
            Self::Host(executor) => BlendshapeExecutorRef::Host(executor),
            Self::Device(executor) => BlendshapeExecutorRef::Device(executor),
        }
    }

    pub fn executor_mut(&mut self) -> BlendshapeExecutorMut<'_> {
        match self {
            Self::Host(executor) => BlendshapeExecutorMut::Host(executor),
            Self::Device(executor) => BlendshapeExecutorMut::Device(executor),
        }
    }

    pub fn cuda_stream(&self) -> &CudaStream {
        match self {
            Self::Host(executor) => executor.cuda_stream(),
            Self::Device(executor) => executor.cuda_stream(),
        }
    }

    pub fn audio_accumulator(&self, track: usize) -> Result<&Arc<AudioAccumulator>> {
        match self {
            Self::Host(executor) => executor.audio_accumulator(track),
            Self::Device(executor) => executor.audio_accumulator(track),
        }
    }

    pub fn emotion_accumulator(&self, track: usize) -> Result<&Arc<EmotionAccumulator>> {
        match self {
            Self::Host(executor) => executor.emotion_accumulator(track),
            Self::Device(executor) => executor.emotion_accumulator(track),
        }
    }
}

impl From<HostBlendshapeSolveExecutor> for BlendshapeExecutorBundle {
    fn from(executor: HostBlendshapeSolveExecutor) -> Self {
        Self::Host(executor)
    }
}

impl From<DeviceBlendshapeSolveExecutor> for BlendshapeExecutorBundle {
    fn from(executor: DeviceBlendshapeSolveExecutor) -> Self {
        Self::Device(executor)
    }
}

/// Consumes a completed geometry bundle and creates a host BlendShape bundle.
#[allow(clippy::result_large_err)]
pub fn create_host_blendshape_solve_executor(
    geometry: GeometryExecutorBundle,
    parameters: HostBlendshapeSolveExecutorCreationParameters<'_>,
) -> std::result::Result<
    HostBlendshapeSolveExecutor,
    crate::audio2x::TransferError<GeometryExecutorBundle>,
> {
    geometry.try_into_host_blendshape(parameters)
}

/// Consumes a completed geometry bundle and creates a device BlendShape bundle.
#[allow(clippy::result_large_err)]
pub fn create_device_blendshape_solve_executor(
    geometry: GeometryExecutorBundle,
    parameters: DeviceBlendshapeSolveExecutorCreationParameters<'_>,
) -> std::result::Result<
    DeviceBlendshapeSolveExecutor,
    crate::audio2x::TransferError<GeometryExecutorBundle>,
> {
    geometry.try_into_device_blendshape(parameters)
}

/// Runtime-independent factory for completed geometry bundles.
pub struct GeometryExecutorBundleFactory;

impl GeometryExecutorBundleFactory {
    /// Loads one model-specific bundle on a worker thread.
    pub fn load(
        parameters: GeometryExecutorBundleCreationParameters,
    ) -> ExecutorFuture<'static, GeometryExecutorBundle> {
        Box::pin(async move {
            match parameters {
                GeometryExecutorBundleCreationParameters::Regression(parameters) => {
                    RegressionGeometryExecutorFactory::load(parameters)
                        .await
                        .map(GeometryExecutorBundle::Regression)
                }
                GeometryExecutorBundleCreationParameters::Diffusion(parameters) => {
                    DiffusionGeometryExecutorFactory::load(parameters)
                        .await
                        .map(GeometryExecutorBundle::Diffusion)
                }
            }
        })
    }

    pub fn regression(
        parameters: RegressionGeometryExecutorCreationParameters,
    ) -> ExecutorFuture<'static, GeometryExecutorBundle> {
        Self::load(GeometryExecutorBundleCreationParameters::Regression(
            parameters,
        ))
    }

    pub fn create_regression_bundle(
        parameters: RegressionGeometryExecutorCreationParameters,
    ) -> ExecutorFuture<'static, GeometryExecutorBundle> {
        Self::regression(parameters)
    }

    pub fn diffusion(
        parameters: DiffusionGeometryExecutorCreationParameters,
    ) -> ExecutorFuture<'static, GeometryExecutorBundle> {
        Self::load(GeometryExecutorBundleCreationParameters::Diffusion(
            parameters,
        ))
    }

    pub fn create_diffusion_bundle(
        parameters: DiffusionGeometryExecutorCreationParameters,
    ) -> ExecutorFuture<'static, GeometryExecutorBundle> {
        Self::diffusion(parameters)
    }
}

pub fn create_regression_bundle(
    parameters: RegressionGeometryExecutorCreationParameters,
) -> ExecutorFuture<'static, GeometryExecutorBundle> {
    GeometryExecutorBundleFactory::regression(parameters)
}

pub fn create_diffusion_bundle(
    parameters: DiffusionGeometryExecutorCreationParameters,
) -> ExecutorFuture<'static, GeometryExecutorBundle> {
    GeometryExecutorBundleFactory::diffusion(parameters)
}

impl std::fmt::Debug for GeometryExecutorBundleCreationParameters {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("GeometryExecutorBundleCreationParameters(..)")
    }
}

impl GeometryExecutorBundle {
    pub fn kind(&self) -> crate::ModelKind {
        match self {
            Self::Regression(_) => crate::ModelKind::Regression,
            Self::Diffusion(_) => crate::ModelKind::Diffusion,
        }
    }
}
