//! Public Audio2Face executor contracts.
//!
//! This module corresponds to the public interfaces in
//! `audio2face-sdk/include/audio2face/executor.h` and
//! `audio2face-sdk/include/audio2face/interactive_executor.h`.
//!
//! # Migration map
//!
//! | Existing Rust API | SDK-facing declaration |
//! |---|---|
//! | `animation::RegressionExecutor` | `regression::RegressionGeometryExecutor` |
//! | `animation::DiffusionExecutor` | `diffusion::DiffusionGeometryExecutor` |
//! | `animation::InteractiveRegressionExecutor<B, P>` | `regression::RegressionGeometryInteractiveExecutor` |
//! | `animation::InteractiveDiffusionExecutor<B, P>` | `diffusion::DiffusionGeometryInteractiveExecutor` |
//! | `animation::SkinAnimatorParams` | [`AnimatorSkinParams`] |
//! | `animation::TongueAnimatorParams` | [`AnimatorTongueParams`] |
//! | `animation::JawParameters` | [`AnimatorTeethParams`] |
//! | `animation::EyesAnimatorParams` | [`AnimatorEyesParams`] |
//! | `animation::BlendshapeSolverParameters` | [`BlendshapeSolverParams`] |
//!
//! Model-specific completed executors remain in their child modules so the
//! model family stays visible at call sites. Their backend and post-processor
//! implementation types are not part of this facade.

use std::ops::ControlFlow;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use std::cell::Cell;
#[cfg(feature = "cuda")]
use std::convert::Infallible;

#[cfg(feature = "tensorrt")]
use crate::Error;
use crate::audio2x::{
    AudioAccumulator, CallbackMetadata, DeviceComponentResults, EmotionAccumulator, Execution,
    Executor, ExecutorFuture, InteractiveExecutionReport, InteractiveExecutor, Result,
};
#[cfg(feature = "tensorrt")]
use crate::audio2x::{ExecutionReport, ExecutionState};

pub mod diffusion;
pub mod job_runner;
pub mod regression;

pub use crate::audio2x::RangeConfig;
pub use job_runner::{JobRunner, JobRunnerTask, ThreadPoolJobRunner};

/// Bit mask selecting the geometry components produced by an execution.
///
/// Corresponds to `nva2f::IGeometryExecutor::ExecutionOption` in
/// `audio2face/executor.h`.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[repr(transparent)]
pub struct GeometryExecutionOption(u32);

impl GeometryExecutionOption {
    pub const NONE: Self = Self(0);
    pub const SKIN: Self = Self(1 << 0);
    pub const TONGUE: Self = Self(1 << 1);
    pub const SKIN_TONGUE: Self = Self(Self::SKIN.0 | Self::TONGUE.0);
    pub const JAW: Self = Self(1 << 2);
    pub const EYES: Self = Self(1 << 3);
    pub const ALL: Self = Self(Self::SKIN_TONGUE.0 | Self::JAW.0 | Self::EYES.0);

    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl std::ops::BitOr for GeometryExecutionOption {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for GeometryExecutionOption {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Shared inputs owned by one geometry track.
#[derive(Clone, Debug)]
pub struct GeometryTrackResources {
    pub audio: Arc<AudioAccumulator>,
    pub emotions: Arc<EmotionAccumulator>,
}

/// Parameters common to all owning geometry executors.
///
/// Corresponds to `nva2f::GeometryExecutorCreationParameters` in
/// `audio2face/executor.h`. The raw accumulator arrays are represented by
/// owned `Arc`s, and the CUDA device is selected by ordinal.
#[derive(Clone, Debug)]
pub struct GeometryExecutorCreationParameters {
    pub tracks: Vec<GeometryTrackResources>,
    pub device_ordinal: i32,
    pub execution_option: GeometryExecutionOption,
}

/// Closed, history-preserving input resources for one interactive geometry executor.
///
/// Unlike the multi-track standard parameters, interactive execution owns one
/// random-access input timeline. Both accumulators must be closed and must not
/// have dropped history when the completed factory validates these parameters.
#[derive(Clone, Debug)]
pub struct GeometryInteractiveExecutorCreationParameters {
    pub audio: Arc<AudioAccumulator>,
    pub emotions: Arc<EmotionAccumulator>,
    pub device_ordinal: i32,
    pub execution_option: GeometryExecutionOption,
}

/// Metadata and device-resident emotions consumed for a face frame.
///
/// Corresponds to `nva2f::IFaceExecutor::Emotions` in
/// `audio2face-sdk/include/audio2face/executor.h`. The device view and stream
/// are valid only during the synchronous callback invocation.
pub struct FaceEmotions<'a> {
    pub metadata: CallbackMetadata,
    pub values: DeviceComponentResults<'a>,
}

/// Device-resident geometry produced for a frame.
///
/// A disabled component is represented by `None`. Each component carries the
/// stream that orders access to its borrowed device view. All borrowed views
/// expire when the callback returns; retaining data requires an explicit copy
/// to caller-owned storage ordered on the supplied stream.
///
/// Corresponds to `nva2f::IGeometryExecutor::Results` in
/// `audio2face-sdk/include/audio2face/executor.h`.
pub struct GeometryResults<'a> {
    pub metadata: CallbackMetadata,
    pub skin: Option<DeviceComponentResults<'a>>,
    pub tongue: Option<DeviceComponentResults<'a>>,
    pub jaw: Option<DeviceComponentResults<'a>>,
    pub eyes: Option<DeviceComponentResults<'a>>,
}

/// Borrowed synchronous callbacks for one geometry execution.
///
/// Both callbacks run on the thread calling [`GeometryExecutor::execute`]
/// before that method returns. `ControlFlow::Break` suppresses only subsequent
/// callbacks for the same track in this execution; other tracks and later
/// executions continue. The optional emotions callback runs immediately before
/// the matching geometry result callback.
pub struct GeometryCallbacks<'c> {
    pub results: &'c mut dyn for<'r> FnMut(GeometryResults<'r>) -> ControlFlow<()>,
    pub emotions: Option<&'c mut dyn for<'r> FnMut(FaceEmotions<'r>)>,
}

/// Common capabilities of Audio2Face executors.
///
/// Corresponds to `nva2f::IFaceExecutor` in
/// `audio2face-sdk/include/audio2face/executor.h`.
pub trait FaceExecutor: Executor {
    fn next_audio_sample_to_read(&self, track: usize) -> Result<usize>;
    fn next_emotion_timestamp_to_read(&self, track: usize) -> Result<i64>;
}

/// Common contract implemented by Regression and Diffusion geometry executors.
///
/// Corresponds to `nva2f::IGeometryExecutor` in
/// `audio2face-sdk/include/audio2face/executor.h`.
pub trait GeometryExecutor: FaceExecutor {
    fn execution_option(&self) -> GeometryExecutionOption;
    fn set_execution_option(&mut self, value: GeometryExecutionOption) -> Result<()>;
    fn skin_geometry_size(&self) -> usize;
    fn tongue_geometry_size(&self) -> usize;
    fn jaw_transform_size(&self) -> usize;
    fn eyes_rotation_size(&self) -> usize;
    fn execute(&mut self, callbacks: GeometryCallbacks<'_>) -> Result<Execution>;
}

/// Location of BlendShape solve results.
///
/// Corresponds to `nva2f::IBlendshapeExecutor::ResultsType` in
/// `audio2face-sdk/include/audio2face/executor.h`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BlendshapeResultKind {
    Unknown,
    Host,
    Device,
}

/// Common capabilities of host and device BlendShape executors.
///
/// Corresponds to `nva2f::IBlendshapeExecutor` in
/// `audio2face-sdk/include/audio2face/executor.h`. Host and device execution
/// entry points are attached to their concrete completed facades in Step 3,
/// because their callback ownership and completion boundaries differ.
pub trait BlendshapeExecutor: FaceExecutor {
    fn weight_count(&self) -> usize;
    fn result_kind(&self) -> BlendshapeResultKind;
}

/// Packed host BlendShape weights produced for one frame.
///
/// Corresponds to `nva2f::IBlendshapeExecutor::HostResults` in
/// `audio2face-sdk/include/audio2face/executor.h`. The weight slice is borrowed
/// only for the duration of the worker callback.
pub struct BlendshapeHostResults<'a> {
    pub metadata: CallbackMetadata,
    pub weights: &'a [f32],
}

/// Packed device BlendShape weights produced for one frame.
///
/// Corresponds to `nva2f::IBlendshapeExecutor::DeviceResults` in
/// `audio2face-sdk/include/audio2face/executor.h`. The device view and stream
/// are callback-scoped and do not imply CUDA completion.
pub struct BlendshapeDeviceResults<'a> {
    pub metadata: CallbackMetadata,
    pub weights: DeviceComponentResults<'a>,
}

/// A host BlendShape callback event, including track-local worker errors.
///
/// This combines the original callback's result and `std::error_code` arguments
/// into one typed Rust result.
pub type HostBlendshapeEvent<'a> = Result<BlendshapeHostResults<'a>>;

/// Callback retained by host BlendShape worker jobs.
///
/// Corresponds to `nva2f::IBlendshapeExecutor::host_results_callback_t` in
/// `audio2face-sdk/include/audio2face/executor.h`. It may be invoked concurrently
/// by multiple worker threads after `execute` returns. Each event's borrowed
/// weights expire when that invocation returns. Awaiting the returned
/// `Execution` (or its track wait) observes callback completion and worker
/// failures.
pub type HostBlendshapeCallback =
    Arc<dyn for<'r> Fn(HostBlendshapeEvent<'r>) + Send + Sync + 'static>;

/// Layer identifiers for interactive geometry invalidation.
///
/// Corresponds to the `kLayer*` constants of
/// `nva2f::IGeometryInteractiveExecutor` in
/// `audio2face-sdk/include/audio2face/interactive_executor.h`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(usize)]
pub enum GeometryInvalidationLayer {
    None = 0,
    All = 1,
    Inference = 2,
    Skin = 3,
    Tongue = 4,
    Teeth = 5,
    Eyes = 6,
}

/// Interactive Regression/Diffusion geometry contract.
///
/// Corresponds to `nva2f::IGeometryInteractiveExecutor` in
/// `audio2face-sdk/include/audio2face/interactive_executor.h`. Each compute
/// method returns a runtime-independent `Send` future borrowing both executor
/// and callback. `Ready` means all host computation and callbacks for the call
/// have finished; device completion remains ordered by the result streams.
/// A callback `Break` interrupts the entire current compute call and is reported
/// as `InteractiveExecutionStatus::Interrupted`.
pub trait GeometryInteractiveExecutor: InteractiveExecutor {
    fn invalidate_geometry(&mut self, layer: GeometryInvalidationLayer) -> Result<()>;
    fn is_geometry_valid(&self, layer: GeometryInvalidationLayer) -> bool;
    fn skin_geometry_size(&self) -> usize;
    fn tongue_geometry_size(&self) -> usize;
    fn jaw_transform_size(&self) -> usize;
    fn eyes_rotation_size(&self) -> usize;
    fn compute_frame<'a>(
        &'a mut self,
        frame: usize,
        callback: &'a mut (dyn for<'r> FnMut(GeometryResults<'r>) -> ControlFlow<()> + Send),
    ) -> ExecutorFuture<'a, InteractiveExecutionReport>;
    fn compute_all_frames<'a>(
        &'a mut self,
        callback: &'a mut (dyn for<'r> FnMut(GeometryResults<'r>) -> ControlFlow<()> + Send),
    ) -> ExecutorFuture<'a, InteractiveExecutionReport>;
}

/// Layer identifiers for interactive BlendShape invalidation.
///
/// Geometry layers remain addressable after ownership is transferred into a
/// BlendShape executor. Values 101--103 correspond to the additional `kLayer*`
/// constants of `nva2f::IBlendshapeInteractiveExecutor` in
/// `audio2face-sdk/include/audio2face/interactive_executor.h`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[repr(usize)]
pub enum BlendshapeInvalidationLayer {
    None = 0,
    All = 1,
    Inference = 2,
    Skin = 3,
    Tongue = 4,
    Teeth = 5,
    Eyes = 6,
    SkinSolverPrepare = 101,
    TongueSolverPrepare = 102,
    BlendshapeWeights = 103,
}

/// Common query and invalidation capabilities of interactive BlendShape executors.
///
/// Host and device compute methods remain result-kind-specific because their
/// callback threading and completion boundaries differ. Corresponds to
/// `nva2f::IBlendshapeInteractiveExecutor` in
/// `audio2face-sdk/include/audio2face/interactive_executor.h`.
pub trait BlendshapeInteractiveExecutor: InteractiveExecutor {
    fn invalidate_blendshape(&mut self, layer: BlendshapeInvalidationLayer) -> Result<()>;
    fn is_blendshape_valid(&self, layer: BlendshapeInvalidationLayer) -> bool;
    fn weight_count(&self) -> usize;
    fn result_kind(&self) -> BlendshapeResultKind;
}

/// Parameters for skin animation/post-processing.
///
/// Corresponds to `nva2f::AnimatorSkinParams` in
/// `audio2face-sdk/include/audio2face/animator.h` and replaces
/// `crate::animation::SkinAnimatorParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimatorSkinParams {
    pub lower_face_smoothing: f32,
    pub upper_face_smoothing: f32,
    pub lower_face_strength: f32,
    pub upper_face_strength: f32,
    pub face_mask_level: f32,
    pub face_mask_softness: f32,
    pub skin_strength: f32,
    pub blink_strength: f32,
    pub eyelid_open_offset: f32,
    pub lip_open_offset: f32,
    pub blink_offset: f32,
}

/// Parameters for tongue animation/post-processing.
///
/// Corresponds to `nva2f::AnimatorTongueParams` in
/// `audio2face-sdk/include/audio2face/animator.h` and replaces
/// `crate::animation::TongueAnimatorParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimatorTongueParams {
    pub tongue_strength: f32,
    pub tongue_height_offset: f32,
    pub tongue_depth_offset: f32,
}

/// Parameters for teeth animation/post-processing.
///
/// Corresponds to `nva2f::AnimatorTeethParams` in
/// `audio2face-sdk/include/audio2face/animator.h`; the existing
/// `crate::animation::JawParameters` remains a math-layer detail.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimatorTeethParams {
    pub lower_teeth_strength: f32,
    pub lower_teeth_height_offset: f32,
    pub lower_teeth_depth_offset: f32,
}

/// Parameters for eye animation/post-processing.
///
/// Corresponds to `nva2f::AnimatorEyesParams` in
/// `audio2face-sdk/include/audio2face/animator.h` and replaces
/// `crate::animation::EyesAnimatorParams`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimatorEyesParams {
    pub eyeballs_strength: f32,
    pub saccade_strength: f32,
    pub right_eyeball_rotation_offset_x: f32,
    pub right_eyeball_rotation_offset_y: f32,
    pub left_eyeball_rotation_offset_x: f32,
    pub left_eyeball_rotation_offset_y: f32,
    pub saccade_seed: f32,
}

/// Combined animator parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimatorParams {
    pub skin: AnimatorSkinParams,
    pub tongue: AnimatorTongueParams,
    pub teeth: AnimatorTeethParams,
    pub eyes: AnimatorEyesParams,
}

/// BlendShape solver regularization parameters.
///
/// Corresponds to `nva2f::BlendshapeSolverParams` in
/// `audio2face-sdk/include/audio2face/blendshape_solver.h` and replaces
/// `crate::animation::BlendshapeSolverParameters`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendshapeSolverParams {
    pub l1_regularization: f32,
    pub l2_regularization: f32,
    pub symmetry_regularization: f32,
    pub temporal_regularization: f32,
    pub template_bounding_box_size: f32,
    pub tolerance: f32,
}

impl Default for BlendshapeSolverParams {
    fn default() -> Self {
        Self {
            l1_regularization: 1.0,
            l2_regularization: 3.5,
            symmetry_regularization: 100.0,
            temporal_regularization: 0.0,
            template_bounding_box_size: 54.7,
            tolerance: 1.0e-10,
        }
    }
}

/// Borrowed host data used to initialize a BlendShape solver.
///
/// Corresponds to `nva2f::BlendshapeSolverDataView` in
/// `audio2face-sdk/include/audio2face/blendshape_solver.h`.
pub struct BlendshapeSolverDataView<'a> {
    pub neutral_pose: &'a [f32],
    pub delta_poses: &'a [f32],
    pub pose_mask: Option<&'a [usize]>,
    pub pose_names: &'a [&'a str],
}

/// Borrowed constraints and weight transforms for a BlendShape solver.
pub struct BlendshapeSolverConfigView<'a> {
    pub active_poses: &'a [i32],
    pub cancel_poses: &'a [i32],
    pub symmetry_poses: &'a [i32],
    pub multipliers: &'a [f32],
    pub offsets: &'a [f32],
}

/// Configuration of one optional BlendShape component.
pub struct BlendshapeSolveComponentParameters<'a> {
    pub params: BlendshapeSolverParams,
    pub config: BlendshapeSolverConfigView<'a>,
    pub data: BlendshapeSolverDataView<'a>,
}

/// Parameters common to host and device BlendShape solve executors.
pub struct BlendshapeSolveExecutorCreationParameters<'a> {
    pub skin: Option<BlendshapeSolveComponentParameters<'a>>,
    pub tongue: Option<BlendshapeSolveComponentParameters<'a>>,
}

/// Parameters for a host BlendShape solve executor.
pub struct HostBlendshapeSolveExecutorCreationParameters<'a> {
    pub components: BlendshapeSolveExecutorCreationParameters<'a>,
    pub job_runner: Option<Arc<dyn JobRunner>>,
}

/// Parameters for a device BlendShape solve executor.
pub struct DeviceBlendshapeSolveExecutorCreationParameters<'a> {
    pub components: BlendshapeSolveExecutorCreationParameters<'a>,
}

#[cfg(feature = "cuda")]
macro_rules! opaque_executor {
    ($name:ident, $original:literal) => {
        #[doc = concat!("Opaque owning facade corresponding to `", $original, "`.")]
        #[doc = " Construction and execution are connected in later implementation steps."]
        pub struct $name {
            _opaque: Infallible,
            _not_sync: Cell<()>,
        }
    };
}

#[cfg(feature = "cuda")]
opaque_executor!(
    HostBlendshapeSolveInteractiveExecutor,
    "nva2f::IBlendshapeInteractiveExecutor (host)"
);
#[cfg(feature = "cuda")]
opaque_executor!(
    DeviceBlendshapeSolveInteractiveExecutor,
    "nva2f::IBlendshapeInteractiveExecutor (device)"
);

/// Owning asynchronous host BlendShape executor.
#[cfg(feature = "cuda")]
pub struct HostBlendshapeSolveExecutor {
    #[cfg(feature = "tensorrt")]
    source: GeometrySource,
    #[cfg(feature = "tensorrt")]
    skin_solvers: Vec<Option<Arc<std::sync::Mutex<crate::animation::CpuBlendshapeSolver>>>>,
    #[cfg(feature = "tensorrt")]
    tongue_solvers: Vec<Option<Arc<std::sync::Mutex<crate::animation::CpuBlendshapeSolver>>>>,
    #[cfg(feature = "tensorrt")]
    runner: Arc<dyn JobRunner>,
    #[cfg(feature = "tensorrt")]
    pending: Vec<Arc<crate::audio2x::ExecutionCompletion>>,
    #[cfg(feature = "tensorrt")]
    weight_count: usize,
    #[cfg(not(feature = "tensorrt"))]
    _opaque: Infallible,
    _not_sync: std::marker::PhantomData<Cell<()>>,
}

/// Owning synchronous device-result BlendShape executor.
#[cfg(feature = "cuda")]
pub struct DeviceBlendshapeSolveExecutor {
    #[cfg(feature = "tensorrt")]
    source: GeometrySource,
    #[cfg(feature = "tensorrt")]
    skin_solvers: Vec<Option<crate::animation::GpuBlendshapeSolver>>,
    #[cfg(feature = "tensorrt")]
    tongue_solvers: Vec<Option<crate::animation::GpuBlendshapeSolver>>,
    #[cfg(feature = "tensorrt")]
    skin_weights: Option<crate::cuda::DeviceBuffer<f32>>,
    #[cfg(feature = "tensorrt")]
    tongue_weights: Option<crate::cuda::DeviceBuffer<f32>>,
    #[cfg(feature = "tensorrt")]
    skin_weight_count: usize,
    #[cfg(feature = "tensorrt")]
    output: crate::cuda::DeviceBuffer<f32>,
    #[cfg(feature = "tensorrt")]
    stream: crate::cuda::CudaStream,
    #[cfg(feature = "tensorrt")]
    device: Arc<crate::cuda::GpuDevice>,
    #[cfg(feature = "tensorrt")]
    weight_count: usize,
    #[cfg(not(feature = "tensorrt"))]
    _opaque: Infallible,
    _not_sync: std::marker::PhantomData<Cell<()>>,
}

#[cfg(feature = "tensorrt")]
#[allow(clippy::large_enum_variant)]
enum GeometrySource {
    Regression(regression::RegressionGeometryExecutor),
    Diffusion(diffusion::DiffusionGeometryExecutor),
}

#[cfg(feature = "tensorrt")]
impl GeometrySource {
    fn track_count(&self) -> usize {
        match self {
            Self::Regression(source) => source.track_count(),
            Self::Diffusion(source) => source.track_count(),
        }
    }

    fn device(&self) -> Arc<crate::cuda::GpuDevice> {
        match self {
            Self::Regression(source) => source.device_arc(),
            Self::Diffusion(source) => source.device_arc(),
        }
    }

    fn set_execution_option(&mut self, option: GeometryExecutionOption) -> Result<()> {
        match self {
            Self::Regression(source) => GeometryExecutor::set_execution_option(source, option),
            Self::Diffusion(source) => GeometryExecutor::set_execution_option(source, option),
        }
    }

    fn execute_device(
        &mut self,
        results: &mut dyn for<'a> FnMut(GeometryResults<'a>) -> ControlFlow<()>,
    ) -> Result<Execution> {
        match self {
            Self::Regression(source) => GeometryExecutor::execute(
                source,
                GeometryCallbacks {
                    results,
                    emotions: None,
                },
            ),
            Self::Diffusion(source) => GeometryExecutor::execute(
                source,
                GeometryCallbacks {
                    results,
                    emotions: None,
                },
            ),
        }
    }

    fn skin_geometry_size(&self) -> usize {
        match self {
            Self::Regression(source) => source.skin_geometry_size(),
            Self::Diffusion(source) => source.skin_geometry_size(),
        }
    }

    fn tongue_geometry_size(&self) -> usize {
        match self {
            Self::Regression(source) => source.tongue_geometry_size(),
            Self::Diffusion(source) => source.tongue_geometry_size(),
        }
    }

    fn execute_host(
        &mut self,
        mut callback: impl FnMut(
            CallbackMetadata,
            &crate::animation::RegressionGeometry,
        ) -> ControlFlow<()>,
    ) -> Result<(ExecutionState, usize)> {
        match self {
            Self::Regression(source) => {
                let mut active = vec![false; source.track_count()];
                let status = source.execute_host(|metadata, geometry| {
                    active[metadata.track] = true;
                    callback(
                        CallbackMetadata {
                            track_index: metadata.track,
                            frame_index: metadata.frame,
                            timestamp: metadata.timestamp,
                            next_timestamp: metadata.next_timestamp,
                        },
                        geometry,
                    )
                })?;
                let state = match status {
                    crate::animation::PumpStatus::AwaitingInput => ExecutionState::AwaitingInput,
                    crate::animation::PumpStatus::Complete => ExecutionState::Complete,
                    crate::animation::PumpStatus::Interrupted => ExecutionState::Progress,
                };
                Ok((state, active.iter().filter(|active| **active).count()))
            }
            Self::Diffusion(source) => {
                let status = source.execute_host(|metadata, geometry| {
                    callback(
                        CallbackMetadata {
                            track_index: metadata.track,
                            frame_index: metadata.frame,
                            timestamp: metadata.timestamp,
                            next_timestamp: metadata.next_timestamp,
                        },
                        geometry,
                    )
                })?;
                Ok(match status {
                    crate::animation::DiffusionExecutionStatus::AwaitingInput => {
                        (ExecutionState::AwaitingInput, 0)
                    }
                    crate::animation::DiffusionExecutionStatus::Complete => {
                        (ExecutionState::Complete, 0)
                    }
                    crate::animation::DiffusionExecutionStatus::Executed { tracks } => {
                        (ExecutionState::Progress, tracks)
                    }
                })
            }
        }
    }

    fn reset_track(&mut self, track: usize) -> Result<()> {
        match self {
            Self::Regression(source) => Executor::reset_track(source, track),
            Self::Diffusion(source) => Executor::reset_track(source, track),
        }
    }

    fn has_execution_started(&self, track: usize) -> Result<bool> {
        match self {
            Self::Regression(source) => Executor::has_execution_started(source, track),
            Self::Diffusion(source) => Executor::has_execution_started(source, track),
        }
    }

    fn available_execution_count(&self, track: usize) -> Result<usize> {
        match self {
            Self::Regression(source) => Executor::available_execution_count(source, track),
            Self::Diffusion(source) => Executor::available_execution_count(source, track),
        }
    }

    fn ready_track_count(&self) -> usize {
        match self {
            Self::Regression(source) => Executor::ready_track_count(source),
            Self::Diffusion(source) => Executor::ready_track_count(source),
        }
    }

    fn total_frame_count(&self, track: usize) -> Result<Option<usize>> {
        match self {
            Self::Regression(source) => Executor::total_frame_count(source, track),
            Self::Diffusion(source) => Executor::total_frame_count(source, track),
        }
    }

    fn sample_rate(&self) -> usize {
        match self {
            Self::Regression(source) => source.sample_rate(),
            Self::Diffusion(source) => source.sample_rate(),
        }
    }

    fn frame_rate(&self) -> crate::audio2x::FrameRate {
        match self {
            Self::Regression(source) => source.frame_rate(),
            Self::Diffusion(source) => source.frame_rate(),
        }
    }

    fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        match self {
            Self::Regression(source) => Executor::frame_timestamp(source, frame),
            Self::Diffusion(source) => Executor::frame_timestamp(source, frame),
        }
    }

    fn next_audio_sample_to_read(&self, track: usize) -> Result<usize> {
        match self {
            Self::Regression(source) => FaceExecutor::next_audio_sample_to_read(source, track),
            Self::Diffusion(source) => FaceExecutor::next_audio_sample_to_read(source, track),
        }
    }

    fn next_emotion_timestamp_to_read(&self, track: usize) -> Result<i64> {
        match self {
            Self::Regression(source) => FaceExecutor::next_emotion_timestamp_to_read(source, track),
            Self::Diffusion(source) => FaceExecutor::next_emotion_timestamp_to_read(source, track),
        }
    }
}

#[cfg(feature = "tensorrt")]
fn owned_blendshape_data(
    component: &BlendshapeSolveComponentParameters<'_>,
) -> crate::animation::BlendshapeData {
    crate::animation::BlendshapeData {
        neutral_pose: component.data.neutral_pose.to_vec(),
        delta_poses: component.data.delta_poses.to_vec(),
        pose_names: component
            .data
            .pose_names
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        pose_mask: component.data.pose_mask.map(<[usize]>::to_vec),
    }
}

#[cfg(feature = "tensorrt")]
fn owned_blendshape_config(
    component: &BlendshapeSolveComponentParameters<'_>,
) -> crate::common::BlendshapeConfig {
    crate::common::BlendshapeConfig {
        l2_regularization: component.params.l2_regularization,
        temporal_regularization: component.params.temporal_regularization,
        l1_regularization: component.params.l1_regularization,
        symmetry_regularization: component.params.symmetry_regularization,
        num_poses: component.data.pose_names.len(),
        active_poses: component.config.active_poses.to_vec(),
        cancel_poses: component.config.cancel_poses.to_vec(),
        symmetry_poses: component.config.symmetry_poses.to_vec(),
        multipliers: component.config.multipliers.to_vec(),
        offsets: component.config.offsets.to_vec(),
        template_bb_size: component.params.template_bounding_box_size,
        tolerance: component.params.tolerance,
    }
}

#[cfg(feature = "tensorrt")]
fn create_cpu_solver(
    component: &BlendshapeSolveComponentParameters<'_>,
) -> Result<crate::animation::CpuBlendshapeSolver> {
    let data = owned_blendshape_data(component);
    let mut solver = crate::animation::CpuBlendshapeSolver::new(data)?;
    solver.set_parameters(crate::animation::BlendshapeSolverParameters {
        l1_regularization: component.params.l1_regularization,
        l2_regularization: component.params.l2_regularization,
        symmetry_regularization: component.params.symmetry_regularization,
        temporal_regularization: component.params.temporal_regularization,
        template_bb_size: component.params.template_bounding_box_size,
        tolerance: component.params.tolerance,
    })?;
    solver.set_active_poses(component.config.active_poses.to_vec())?;
    solver.set_cancel_poses(component.config.cancel_poses.to_vec())?;
    solver.set_symmetry_poses(component.config.symmetry_poses.to_vec())?;
    solver.set_multipliers(component.config.multipliers.to_vec())?;
    solver.set_offsets(component.config.offsets.to_vec())?;
    solver.prepare()?;
    Ok(solver)
}

#[cfg(feature = "tensorrt")]
fn create_solvers(
    source: &GeometrySource,
    components: &BlendshapeSolveExecutorCreationParameters<'_>,
) -> Result<(
    Option<crate::animation::CpuBlendshapeSolver>,
    Option<crate::animation::CpuBlendshapeSolver>,
    usize,
)> {
    let skin = components
        .skin
        .as_ref()
        .map(create_cpu_solver)
        .transpose()?;
    let tongue = components
        .tongue
        .as_ref()
        .map(create_cpu_solver)
        .transpose()?;
    if skin.is_none() && tongue.is_none() {
        return Err(Error::InvalidArgument {
            field: "components",
            reason: "at least one BlendShape component is required".into(),
        });
    }
    if skin
        .as_ref()
        .is_some_and(|solver| solver.data().neutral_pose.len() != source.skin_geometry_size())
    {
        return Err(Error::SizeMismatch {
            field: "skin neutral pose",
            expected: source.skin_geometry_size(),
            actual: skin.as_ref().unwrap().data().neutral_pose.len(),
        });
    }
    if tongue
        .as_ref()
        .is_some_and(|solver| solver.data().neutral_pose.len() != source.tongue_geometry_size())
    {
        return Err(Error::SizeMismatch {
            field: "tongue neutral pose",
            expected: source.tongue_geometry_size(),
            actual: tongue.as_ref().unwrap().data().neutral_pose.len(),
        });
    }
    let count = skin.as_ref().map_or(0, |solver| solver.data().pose_count())
        + tongue
            .as_ref()
            .map_or(0, |solver| solver.data().pose_count());
    Ok((skin, tongue, count))
}

#[cfg(feature = "tensorrt")]
fn create_gpu_solver(
    component: &BlendshapeSolveComponentParameters<'_>,
    device: &Arc<crate::cuda::GpuDevice>,
    stream: &crate::cuda::CudaStream,
) -> Result<crate::animation::GpuBlendshapeSolver> {
    crate::animation::GpuBlendshapeSolver::new(
        device,
        stream,
        owned_blendshape_data(component),
        &owned_blendshape_config(component),
    )
}

#[cfg(feature = "tensorrt")]
impl HostBlendshapeSolveExecutor {
    // The error must retain the unique owning geometry source for recovery.
    #[allow(clippy::result_large_err)]
    fn from_source(
        source: GeometrySource,
        parameters: HostBlendshapeSolveExecutorCreationParameters<'_>,
    ) -> std::result::Result<Self, (Error, GeometrySource)> {
        let (skin, tongue, weight_count) = match create_solvers(&source, &parameters.components) {
            Ok(solvers) => solvers,
            Err(error) => return Err((error, source)),
        };
        let component_count = usize::from(skin.is_some()) + usize::from(tongue.is_some());
        let runner = match parameters.job_runner {
            Some(runner) => runner,
            None => match ThreadPoolJobRunner::for_components(component_count) {
                Ok(runner) => Arc::new(runner),
                Err(error) => return Err((error, source)),
            },
        };
        let track_count = source.track_count();
        Ok(Self {
            source,
            skin_solvers: (0..track_count)
                .map(|_| {
                    skin.clone()
                        .map(|solver| Arc::new(std::sync::Mutex::new(solver)))
                })
                .collect(),
            tongue_solvers: (0..track_count)
                .map(|_| {
                    tongue
                        .clone()
                        .map(|solver| Arc::new(std::sync::Mutex::new(solver)))
                })
                .collect(),
            runner,
            pending: Vec::new(),
            weight_count,
            _not_sync: std::marker::PhantomData,
        })
    }

    /// Schedules CPU solve jobs and returns a call-local completion handle.
    pub fn execute(&mut self, callback: HostBlendshapeCallback) -> Result<Execution> {
        let track_count = self.source.track_count();
        let (execution, completion) = Execution::pending(track_count);
        let runner = Arc::clone(&self.runner);
        let skin_solvers = &self.skin_solvers;
        let tongue_solvers = &self.tongue_solvers;
        let mut emitted_frames = 0;
        let run = self.source.execute_host(|metadata, geometry| {
            let skin = geometry.skin.clone();
            let tongue = geometry.tongue.clone();
            let skin_solver = skin_solvers[metadata.track_index].clone();
            let tongue_solver = tongue_solvers[metadata.track_index].clone();
            let task_callback = Arc::clone(&callback);
            if let Err(error) = completion.add_task(metadata.track_index) {
                tracing::warn!(error = %error, "failed to register BlendShape task");
                return ControlFlow::Break(());
            }
            let task =
                JobRunnerTask::new(Arc::clone(&completion), metadata.track_index, move || {
                    let solve = (|| -> Result<Vec<f32>> {
                        let mut weights = Vec::new();
                        if let Some(solver) = skin_solver {
                            weights.extend(
                                solver
                                    .lock()
                                    .map_err(|_| Error::Poisoned {
                                        resource: "skin BlendShape solver",
                                    })?
                                    .solve(&skin)?,
                            );
                        }
                        if let Some(solver) = tongue_solver {
                            weights.extend(
                                solver
                                    .lock()
                                    .map_err(|_| Error::Poisoned {
                                        resource: "tongue BlendShape solver",
                                    })?
                                    .solve(&tongue)?,
                            );
                        }
                        Ok(weights)
                    })();
                    match solve {
                        Ok(weights) => {
                            task_callback(Ok(BlendshapeHostResults {
                                metadata,
                                weights: &weights,
                            }));
                            Ok(())
                        }
                        Err(error) => {
                            task_callback(Err(error.clone()));
                            Err(error)
                        }
                    }
                });
            emitted_frames += 1;
            if let Err(error) = runner.enqueue(task) {
                callback(Err(error));
            }
            ControlFlow::Continue(())
        });
        let (state, executed_tracks) = match run {
            Ok(report) => report,
            Err(error) => {
                completion.finish_schedule(ExecutionReport {
                    state: ExecutionState::Progress,
                    executed_tracks: 0,
                    emitted_frames,
                });
                return Err(error);
            }
        };
        completion.finish_schedule(ExecutionReport {
            state,
            executed_tracks,
            emitted_frames,
        });
        self.pending.push(completion);
        Ok(execution)
    }
}

#[cfg(feature = "tensorrt")]
impl Drop for HostBlendshapeSolveExecutor {
    fn drop(&mut self) {
        for completion in &self.pending {
            completion.wait_blocking();
        }
    }
}

#[cfg(feature = "tensorrt")]
impl DeviceBlendshapeSolveExecutor {
    // The error must retain the unique owning geometry source for recovery.
    #[allow(clippy::result_large_err)]
    fn from_source(
        mut source: GeometrySource,
        parameters: DeviceBlendshapeSolveExecutorCreationParameters<'_>,
    ) -> std::result::Result<Self, (Error, GeometrySource)> {
        let (_, _, weight_count) = match create_solvers(&source, &parameters.components) {
            Ok(solvers) => solvers,
            Err(error) => return Err((error, source)),
        };
        let device = source.device();
        let stream = match device.create_stream() {
            Ok(stream) => stream,
            Err(error) => return Err((error, source)),
        };
        let mut option = GeometryExecutionOption::NONE;
        if parameters.components.skin.is_some() {
            option |= GeometryExecutionOption::SKIN;
        }
        if parameters.components.tongue.is_some() {
            option |= GeometryExecutionOption::TONGUE;
        }
        if let Err(error) = source.set_execution_option(option) {
            return Err((error, source));
        }
        let track_count = source.track_count();
        let built = (|| -> Result<_> {
            let skin_weight_count = parameters
                .components
                .skin
                .as_ref()
                .map_or(0, |component| component.data.pose_names.len());
            let skin_solvers = (0..track_count)
                .map(|_| {
                    parameters
                        .components
                        .skin
                        .as_ref()
                        .map(|component| create_gpu_solver(component, &device, &stream))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let tongue_solvers = (0..track_count)
                .map(|_| {
                    parameters
                        .components
                        .tongue
                        .as_ref()
                        .map(|component| create_gpu_solver(component, &device, &stream))
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            let skin_weights = (skin_weight_count != 0)
                .then(|| device.allocate(skin_weight_count))
                .transpose()?;
            let tongue_weight_count = weight_count.saturating_sub(skin_weight_count);
            let tongue_weights = (tongue_weight_count != 0)
                .then(|| device.allocate(tongue_weight_count))
                .transpose()?;
            let output = device.allocate(weight_count)?;
            Ok((
                skin_solvers,
                tongue_solvers,
                skin_weights,
                tongue_weights,
                skin_weight_count,
                output,
            ))
        })();
        let (skin_solvers, tongue_solvers, skin_weights, tongue_weights, skin_weight_count, output) =
            match built {
                Ok(built) => built,
                Err(error) => return Err((error, source)),
            };
        Ok(Self {
            source,
            skin_solvers,
            tongue_solvers,
            skin_weights,
            tongue_weights,
            skin_weight_count,
            output,
            stream,
            device,
            weight_count,
            _not_sync: std::marker::PhantomData,
        })
    }

    /// Solves synchronously and exposes callback-scoped device weights.
    pub fn execute(
        &mut self,
        callback: &mut dyn for<'r> FnMut(BlendshapeDeviceResults<'r>) -> ControlFlow<()>,
    ) -> Result<Execution> {
        let skin_solvers = &mut self.skin_solvers;
        let tongue_solvers = &mut self.tongue_solvers;
        let skin_weights = &mut self.skin_weights;
        let tongue_weights = &mut self.tongue_weights;
        let skin_weight_count = self.skin_weight_count;
        let output = &mut self.output;
        let stream = &self.stream;
        let device = &self.device;
        let mut callback_error = None;
        let mut geometry_callback = |geometry: GeometryResults<'_>| {
            if callback_error.is_some() {
                return ControlFlow::Break(());
            }
            let metadata = geometry.metadata;
            let solve = (|| -> Result<()> {
                let producer = geometry
                    .skin
                    .as_ref()
                    .map(|component| component.stream)
                    .or_else(|| geometry.tongue.as_ref().map(|component| component.stream))
                    .ok_or(Error::InvalidState {
                        operation: "device BlendShape solve",
                        state: "geometry component is unavailable",
                    })?;
                device.synchronize_borrowed_stream(producer)?;
                if let (Some(solver), Some(component), Some(weights)) = (
                    &mut skin_solvers[metadata.track_index],
                    geometry.skin,
                    skin_weights.as_mut(),
                ) {
                    let fence = solver.solve_async_view(component.values, weights, stream)?;
                    fence.synchronize()?;
                    drop(fence);
                    output.copy_from_device_range(0, weights, 0, weights.len(), stream)?;
                }
                if let (Some(solver), Some(component), Some(weights)) = (
                    &mut tongue_solvers[metadata.track_index],
                    geometry.tongue,
                    tongue_weights.as_mut(),
                ) {
                    let fence = solver.solve_async_view(component.values, weights, stream)?;
                    fence.synchronize()?;
                    drop(fence);
                    output.copy_from_device_range(
                        skin_weight_count,
                        weights,
                        0,
                        weights.len(),
                        stream,
                    )?;
                }
                Ok(())
            })();
            if let Err(error) = solve {
                callback_error = Some(error);
                return ControlFlow::Break(());
            }
            callback(BlendshapeDeviceResults {
                metadata,
                weights: DeviceComponentResults {
                    values: output.view(),
                    stream: stream.as_ref(),
                },
            })
        };
        let execution = self.source.execute_device(&mut geometry_callback)?;
        if let Some(error) = callback_error {
            return Err(error);
        }
        Ok(execution)
    }
}

#[cfg(feature = "tensorrt")]
impl Executor for HostBlendshapeSolveExecutor {
    fn track_count(&self) -> usize {
        self.source.track_count()
    }

    fn reset_track(&mut self, track: usize) -> Result<()> {
        if self
            .pending
            .iter()
            .any(|completion| completion.track_pending(track).unwrap_or(false))
        {
            return Err(Error::InvalidState {
                operation: "reset host BlendShape track",
                state: "track jobs are pending",
            });
        }
        self.source.reset_track(track)?;
        if let Some(solver) = self.skin_solvers.get(track).ok_or(Error::OutOfBounds {
            field: "track",
            index: track,
            len: self.source.track_count(),
        })? {
            solver
                .lock()
                .map_err(|_| Error::Poisoned {
                    resource: "skin BlendShape solver",
                })?
                .reset();
        }
        if let Some(solver) = &self.tongue_solvers[track] {
            solver
                .lock()
                .map_err(|_| Error::Poisoned {
                    resource: "tongue BlendShape solver",
                })?
                .reset();
        }
        Ok(())
    }

    fn has_execution_started(&self, track: usize) -> Result<bool> {
        self.source.has_execution_started(track)
    }

    fn available_execution_count(&self, track: usize) -> Result<usize> {
        self.source.available_execution_count(track)
    }

    fn ready_track_count(&self) -> usize {
        self.source.ready_track_count()
    }

    fn total_frame_count(&self, track: usize) -> Result<Option<usize>> {
        self.source.total_frame_count(track)
    }

    fn sample_rate(&self) -> usize {
        self.source.sample_rate()
    }

    fn frame_rate(&self) -> crate::audio2x::FrameRate {
        self.source.frame_rate()
    }

    fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        self.source.frame_timestamp(frame)
    }
}

#[cfg(feature = "tensorrt")]
impl FaceExecutor for HostBlendshapeSolveExecutor {
    fn next_audio_sample_to_read(&self, track: usize) -> Result<usize> {
        self.source.next_audio_sample_to_read(track)
    }

    fn next_emotion_timestamp_to_read(&self, track: usize) -> Result<i64> {
        self.source.next_emotion_timestamp_to_read(track)
    }
}

#[cfg(feature = "tensorrt")]
impl BlendshapeExecutor for HostBlendshapeSolveExecutor {
    fn weight_count(&self) -> usize {
        self.weight_count
    }

    fn result_kind(&self) -> BlendshapeResultKind {
        BlendshapeResultKind::Host
    }
}

#[cfg(feature = "tensorrt")]
impl Executor for DeviceBlendshapeSolveExecutor {
    fn track_count(&self) -> usize {
        self.source.track_count()
    }

    fn reset_track(&mut self, track: usize) -> Result<()> {
        self.source.reset_track(track)?;
        let len = self.source.track_count();
        if let Some(solver) = self.skin_solvers.get_mut(track).ok_or(Error::OutOfBounds {
            field: "track",
            index: track,
            len,
        })? {
            solver.reset(&self.stream)?;
        }
        if let Some(solver) = &mut self.tongue_solvers[track] {
            solver.reset(&self.stream)?;
        }
        Ok(())
    }

    fn has_execution_started(&self, track: usize) -> Result<bool> {
        self.source.has_execution_started(track)
    }

    fn available_execution_count(&self, track: usize) -> Result<usize> {
        self.source.available_execution_count(track)
    }

    fn ready_track_count(&self) -> usize {
        self.source.ready_track_count()
    }

    fn total_frame_count(&self, track: usize) -> Result<Option<usize>> {
        self.source.total_frame_count(track)
    }

    fn sample_rate(&self) -> usize {
        self.source.sample_rate()
    }

    fn frame_rate(&self) -> crate::audio2x::FrameRate {
        self.source.frame_rate()
    }

    fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        self.source.frame_timestamp(frame)
    }
}

#[cfg(feature = "tensorrt")]
impl FaceExecutor for DeviceBlendshapeSolveExecutor {
    fn next_audio_sample_to_read(&self, track: usize) -> Result<usize> {
        self.source.next_audio_sample_to_read(track)
    }

    fn next_emotion_timestamp_to_read(&self, track: usize) -> Result<i64> {
        self.source.next_emotion_timestamp_to_read(track)
    }
}

#[cfg(feature = "tensorrt")]
impl BlendshapeExecutor for DeviceBlendshapeSolveExecutor {
    fn weight_count(&self) -> usize {
        self.weight_count
    }

    fn result_kind(&self) -> BlendshapeResultKind {
        BlendshapeResultKind::Device
    }
}
