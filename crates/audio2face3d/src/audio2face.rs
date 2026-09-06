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
use std::sync::{Mutex, MutexGuard};

#[cfg(feature = "cuda")]
use std::cell::Cell;
#[cfg(all(feature = "cuda", not(feature = "tensorrt")))]
use std::convert::Infallible;

#[cfg(feature = "tensorrt")]
use crate::Error;
#[cfg(feature = "cuda")]
use crate::animation::{
    InteractiveBlendshapeLayer, InteractiveBlendshapeWeights, RegressionGeometry,
};
#[cfg(feature = "cuda")]
use crate::audio2x::InteractiveInterruptHandle;
use crate::audio2x::{
    AudioAccumulator, CallbackMetadata, DeviceComponentResults, EmotionAccumulator, Execution,
    Executor, ExecutorFuture, InteractiveExecutionReport, InteractiveExecutor, Result,
};
#[cfg(feature = "tensorrt")]
use crate::audio2x::{ExecutionReport, ExecutionState};

pub mod animator;
pub mod blendshape_solver;
#[cfg(feature = "tensorrt")]
pub mod bundle;
pub mod diffusion;
pub mod job_runner;
#[cfg(feature = "cuda")]
pub mod noise;
pub mod regression;

pub use crate::animation::CpuBlendshapeSolver;
#[cfg(feature = "cuda")]
pub use crate::animation::GpuPhiloxNoise;
pub use crate::audio2x::RangeConfig;
pub use animator::{
    AnimatorEyes, AnimatorPcaReconstruction, AnimatorSkin, AnimatorTeeth, AnimatorTongue,
    create_animator_eyes, create_animator_pca_reconstruction, create_animator_skin,
    create_animator_teeth, create_animator_tongue,
};
pub use blendshape_solver::create_blendshape_solver;
#[cfg(feature = "tensorrt")]
pub use bundle::{
    BlendshapeExecutorBundle, BlendshapeExecutorMut, BlendshapeExecutorRef, GeometryExecutorBundle,
    GeometryExecutorBundleCreationParameters, GeometryExecutorBundleFactory, GeometryExecutorMut,
    GeometryExecutorRef, create_device_blendshape_solve_executor, create_diffusion_bundle,
    create_host_blendshape_solve_executor, create_regression_bundle,
};
pub use job_runner::{
    JobRunner, JobRunnerTask, ThreadPoolJobRunner, create_thread_pool_job_runner,
};
#[cfg(feature = "cuda")]
pub use noise::create_noise_generator;

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
/// Owning CPU interactive BlendShape facade corresponding to
/// `nva2f::IBlendshapeInteractiveExecutor` host results.
#[cfg(feature = "cuda")]
pub struct HostBlendshapeSolveInteractiveExecutor {
    layer: Arc<Mutex<InteractiveBlendshapeLayer>>,
    runner: Arc<dyn JobRunner>,
    total_frames: Option<usize>,
    sample_rate: usize,
    frame_rate: crate::audio2x::FrameRate,
    interrupt: InteractiveInterruptHandle,
    _not_sync: Cell<()>,
}

/// Owning GPU interactive BlendShape facade corresponding to
/// `nva2f::IBlendshapeInteractiveExecutor` device results.
#[cfg(feature = "cuda")]
pub struct DeviceBlendshapeSolveInteractiveExecutor {
    layer: crate::animation::InteractiveGpuBlendshapeLayer,
    total_frames: Option<usize>,
    sample_rate: usize,
    frame_rate: crate::audio2x::FrameRate,
    interrupt: InteractiveInterruptHandle,
    _not_sync: Cell<()>,
}

#[cfg(feature = "cuda")]
impl HostBlendshapeSolveInteractiveExecutor {
    /// Wraps an already prepared CPU interactive layer.
    pub fn from_layer(
        layer: InteractiveBlendshapeLayer,
        sample_rate: usize,
        frame_rate: crate::audio2x::FrameRate,
    ) -> Result<Self> {
        Self::from_layer_with_runner(layer, sample_rate, frame_rate, None)
    }

    /// Wraps a CPU layer and optionally shares a caller-provided job runner.
    pub fn from_layer_with_runner(
        layer: InteractiveBlendshapeLayer,
        sample_rate: usize,
        frame_rate: crate::audio2x::FrameRate,
        runner: Option<Arc<dyn JobRunner>>,
    ) -> Result<Self> {
        let component_count = usize::from(layer.skin_solver().is_some())
            + usize::from(layer.tongue_solver().is_some());
        let runner = match runner {
            Some(runner) => runner,
            None => Arc::new(ThreadPoolJobRunner::new_for_components(component_count)?),
        };
        Ok(Self {
            layer: Arc::new(Mutex::new(layer)),
            runner,
            total_frames: None,
            sample_rate,
            frame_rate,
            interrupt: InteractiveInterruptHandle::new(),
            _not_sync: Cell::new(()),
        })
    }

    pub fn layer(&self) -> Result<MutexGuard<'_, InteractiveBlendshapeLayer>> {
        self.layer.lock().map_err(|_| crate::Error::Poisoned {
            resource: "interactive BlendShape layer",
        })
    }

    pub fn layer_mut(&mut self) -> Result<MutexGuard<'_, InteractiveBlendshapeLayer>> {
        self.layer()
    }

    /// Computes one random-access frame. The callback runs during one future
    /// poll and receives an owned snapshot, so it may retain the values.
    pub fn compute_frame<'a, C>(
        &'a mut self,
        frame: usize,
        total_frames: usize,
        geometry: &'a RegressionGeometry,
        callback: C,
    ) -> ExecutorFuture<'a, InteractiveExecutionReport>
    where
        C: FnMut(&InteractiveBlendshapeWeights) -> bool + Send + Unpin + 'a,
    {
        self.total_frames = Some(total_frames);
        Box::pin(HostBlendshapeInteractiveFuture {
            executor: self,
            geometry: Some(geometry),
            all_geometry: None,
            frame,
            total_frames,
            callback,
            next_frame: 0,
            pending: None,
            pending_result: None,
            finished: false,
            generation_started: false,
            generation: 0,
            emitted_frames: 0,
        })
    }

    /// Computes one frame per poll for an ordered pass.
    pub fn compute_all_frames<'a, C>(
        &'a mut self,
        geometry: &'a [RegressionGeometry],
        callback: C,
    ) -> ExecutorFuture<'a, InteractiveExecutionReport>
    where
        C: FnMut(&InteractiveBlendshapeWeights) -> bool + Send + Unpin + 'a,
    {
        self.total_frames = Some(geometry.len());
        Box::pin(HostBlendshapeInteractiveFuture {
            executor: self,
            geometry: None,
            all_geometry: Some(geometry),
            frame: 0,
            total_frames: geometry.len(),
            callback,
            next_frame: 0,
            pending: None,
            pending_result: None,
            finished: false,
            generation_started: false,
            generation: 0,
            emitted_frames: 0,
        })
    }

    fn timestamp(&self, frame: usize) -> Result<i64> {
        let numerator = self.frame_rate.numerator() as u128;
        let denominator = self.frame_rate.denominator() as u128;
        let samples = (frame as u128)
            .checked_mul(self.sample_rate as u128)
            .and_then(|value| value.checked_mul(denominator))
            .and_then(|value| value.checked_div(numerator))
            .ok_or(crate::Error::InvalidArgument {
                field: "frame",
                reason: "frame timestamp overflow".into(),
            })?;
        i64::try_from(samples).map_err(|_| crate::Error::InvalidArgument {
            field: "frame",
            reason: "frame timestamp exceeds sample range".into(),
        })
    }
}

#[cfg(feature = "cuda")]
struct HostBlendshapeInteractiveFuture<'a, C> {
    executor: &'a mut HostBlendshapeSolveInteractiveExecutor,
    geometry: Option<&'a RegressionGeometry>,
    all_geometry: Option<&'a [RegressionGeometry]>,
    frame: usize,
    total_frames: usize,
    callback: C,
    next_frame: usize,
    pending: Option<Execution>,
    pending_result: Option<Arc<Mutex<Option<Result<InteractiveBlendshapeWeights>>>>>,
    finished: bool,
    generation_started: bool,
    generation: u64,
    emitted_frames: usize,
}

#[cfg(feature = "cuda")]
impl<C> std::future::Future for HostBlendshapeInteractiveFuture<'_, C>
where
    C: FnMut(&InteractiveBlendshapeWeights) -> bool + Unpin,
{
    type Output = Result<InteractiveExecutionReport>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        if this.finished {
            panic!("interactive BlendShape future polled after completion");
        }
        if !this.generation_started {
            this.generation = this.executor.interrupt.generation();
            this.generation_started = true;
        }
        if this
            .executor
            .interrupt
            .is_interrupted_since(this.generation)
        {
            this.finished = true;
            return std::task::Poll::Ready(Ok(InteractiveExecutionReport {
                status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                emitted_frames: this.emitted_frames,
            }));
        }

        if let Some(execution) = &mut this.pending {
            match std::future::Future::poll(std::pin::Pin::new(execution), cx) {
                std::task::Poll::Pending => return std::task::Poll::Pending,
                std::task::Poll::Ready(Err(error)) => {
                    this.finished = true;
                    return std::task::Poll::Ready(Err(error));
                }
                std::task::Poll::Ready(Ok(_)) => {}
            }
            this.pending = None;
            let Some(result) = this.pending_result.take() else {
                this.finished = true;
                return std::task::Poll::Ready(Err(crate::Error::InvalidState {
                    operation: "complete interactive BlendShape job",
                    state: "worker result storage is missing",
                }));
            };
            let weights = match result.lock() {
                Ok(mut result) => match result.take() {
                    Some(Ok(weights)) => weights,
                    Some(Err(error)) => {
                        this.finished = true;
                        return std::task::Poll::Ready(Err(error));
                    }
                    None => {
                        this.finished = true;
                        return std::task::Poll::Ready(Err(crate::Error::InvalidState {
                            operation: "complete interactive BlendShape job",
                            state: "worker completed without a result",
                        }));
                    }
                },
                Err(_) => {
                    this.finished = true;
                    return std::task::Poll::Ready(Err(crate::Error::Poisoned {
                        resource: "interactive BlendShape result",
                    }));
                }
            };

            let keep_going = (this.callback)(&weights);
            this.emitted_frames += 1;
            if this.all_geometry.is_some() {
                this.next_frame += 1;
            }
            if !keep_going
                || this
                    .executor
                    .interrupt
                    .is_interrupted_since(this.generation)
                || this.geometry.is_some()
                || this.next_frame >= this.total_frames
            {
                this.finished = true;
                let status = if !keep_going
                    || this
                        .executor
                        .interrupt
                        .is_interrupted_since(this.generation)
                {
                    crate::audio2x::InteractiveExecutionStatus::Interrupted
                } else {
                    crate::audio2x::InteractiveExecutionStatus::Complete
                };
                return std::task::Poll::Ready(Ok(InteractiveExecutionReport {
                    status,
                    emitted_frames: this.emitted_frames,
                }));
            }
        }

        let (frame, geometry, ordered) = if let Some(geometry) = this.geometry {
            (this.frame, geometry.clone(), false)
        } else if let Some(geometry) = this.all_geometry {
            if this.next_frame >= geometry.len() {
                this.finished = true;
                return std::task::Poll::Ready(Ok(InteractiveExecutionReport {
                    status: crate::audio2x::InteractiveExecutionStatus::Complete,
                    emitted_frames: this.emitted_frames,
                }));
            }
            (this.next_frame, geometry[this.next_frame].clone(), true)
        } else {
            this.finished = true;
            return std::task::Poll::Ready(Err(crate::Error::InvalidState {
                operation: "interactive BlendShape compute",
                state: "no geometry was supplied",
            }));
        };
        let result = Arc::new(Mutex::new(None));
        let result_for_job = Arc::clone(&result);
        let layer = Arc::clone(&this.executor.layer);
        let total_frames = this.total_frames;
        let (execution, completion) = Execution::pending(1);
        if let Err(error) = completion.add_task(0) {
            this.finished = true;
            return std::task::Poll::Ready(Err(error));
        }
        let task = JobRunnerTask::new(Arc::clone(&completion), 0, move || {
            let computation = (|| {
                let mut layer = layer.lock().map_err(|_| crate::Error::Poisoned {
                    resource: "interactive BlendShape layer",
                })?;
                if ordered {
                    if frame == 0 {
                        layer.begin_all_frames(total_frames)?;
                    }
                    layer.compute_next_frame(frame, &geometry)
                } else {
                    layer.compute_frame(frame, total_frames, &geometry)
                }
            })();
            let mut output = result_for_job.lock().map_err(|_| crate::Error::Poisoned {
                resource: "interactive BlendShape result",
            })?;
            *output = Some(computation);
            Ok(())
        });
        if let Err(error) = this.executor.runner.enqueue(task) {
            completion.finish_schedule(crate::audio2x::ExecutionReport {
                state: crate::audio2x::ExecutionState::Progress,
                executed_tracks: 1,
                emitted_frames: 0,
            });
            this.finished = true;
            return std::task::Poll::Ready(Err(error));
        }
        completion.finish_schedule(crate::audio2x::ExecutionReport {
            state: crate::audio2x::ExecutionState::Progress,
            executed_tracks: 1,
            emitted_frames: 1,
        });
        this.pending = Some(execution);
        this.pending_result = Some(result);
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    }
}

#[cfg(feature = "cuda")]
impl<C> Drop for HostBlendshapeInteractiveFuture<'_, C> {
    fn drop(&mut self) {
        // An ordered pass keeps completed frame cache entries while its
        // remaining entries stay empty, making validity false after drop.
        // Clearing all entries here would discard useful completed work.
    }
}

#[cfg(feature = "cuda")]
impl InteractiveExecutor for HostBlendshapeSolveInteractiveExecutor {
    fn invalidate_all(&mut self) -> Result<()> {
        self.layer()?
            .invalidate(crate::animation::BlendshapeInvalidationLayer::All);
        Ok(())
    }

    fn is_fully_valid(&self) -> bool {
        self.layer()
            .is_ok_and(|layer| layer.is_valid(crate::animation::BlendshapeInvalidationLayer::All))
    }

    fn total_frame_count(&self) -> Result<usize> {
        self.total_frames.ok_or(crate::Error::InvalidState {
            operation: "query interactive BlendShape frame count",
            state: "no computation has established the input timeline",
        })
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn frame_rate(&self) -> crate::audio2x::FrameRate {
        self.frame_rate
    }

    fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        let total = self.total_frame_count()?;
        if frame >= total {
            return Err(crate::Error::OutOfBounds {
                field: "frame",
                index: frame,
                len: total,
            });
        }
        self.timestamp(frame)
    }

    fn interrupt_handle(&self) -> InteractiveInterruptHandle {
        self.interrupt.clone()
    }
}

#[cfg(feature = "cuda")]
impl BlendshapeInteractiveExecutor for HostBlendshapeSolveInteractiveExecutor {
    fn invalidate_blendshape(&mut self, layer: BlendshapeInvalidationLayer) -> Result<()> {
        match layer {
            BlendshapeInvalidationLayer::None => {}
            BlendshapeInvalidationLayer::SkinSolverPrepare => self
                .layer()?
                .invalidate(crate::animation::BlendshapeInvalidationLayer::SkinSolverPrepare),
            BlendshapeInvalidationLayer::TongueSolverPrepare => self
                .layer()?
                .invalidate(crate::animation::BlendshapeInvalidationLayer::TongueSolverPrepare),
            _ => self
                .layer()?
                .invalidate(crate::animation::BlendshapeInvalidationLayer::Weights),
        }
        Ok(())
    }

    fn is_blendshape_valid(&self, layer: BlendshapeInvalidationLayer) -> bool {
        let Ok(layer_guard) = self.layer() else {
            return false;
        };
        match layer {
            BlendshapeInvalidationLayer::SkinSolverPrepare => layer_guard
                .is_valid(crate::animation::BlendshapeInvalidationLayer::SkinSolverPrepare),
            BlendshapeInvalidationLayer::TongueSolverPrepare => layer_guard
                .is_valid(crate::animation::BlendshapeInvalidationLayer::TongueSolverPrepare),
            BlendshapeInvalidationLayer::None => true,
            _ => layer_guard.is_valid(crate::animation::BlendshapeInvalidationLayer::Weights),
        }
    }

    fn weight_count(&self) -> usize {
        let Ok(layer) = self.layer() else {
            return 0;
        };
        layer
            .skin_solver()
            .map_or(0, |solver| solver.data().pose_count())
            + layer
                .tongue_solver()
                .map_or(0, |solver| solver.data().pose_count())
    }

    fn result_kind(&self) -> BlendshapeResultKind {
        BlendshapeResultKind::Host
    }
}

#[cfg(feature = "cuda")]
impl DeviceBlendshapeSolveInteractiveExecutor {
    /// Wraps an already prepared GPU interactive layer. Device views yielded
    /// to the callback are valid until the callback returns; dependent work
    /// must remain ordered on the supplied layer stream.
    pub fn from_layer(
        layer: crate::animation::InteractiveGpuBlendshapeLayer,
        sample_rate: usize,
        frame_rate: crate::audio2x::FrameRate,
    ) -> Self {
        Self {
            layer,
            total_frames: None,
            sample_rate,
            frame_rate,
            interrupt: InteractiveInterruptHandle::new(),
            _not_sync: Cell::new(()),
        }
    }

    pub fn layer(&self) -> &crate::animation::InteractiveGpuBlendshapeLayer {
        &self.layer
    }

    pub fn layer_mut(&mut self) -> &mut crate::animation::InteractiveGpuBlendshapeLayer {
        &mut self.layer
    }

    pub fn compute_frame<'a, C>(
        &'a mut self,
        frame: usize,
        total_frames: usize,
        geometry: &'a RegressionGeometry,
        callback: C,
    ) -> ExecutorFuture<'a, InteractiveExecutionReport>
    where
        C: for<'r> FnMut(crate::animation::InteractiveGpuBlendshapeOutput<'r>) -> bool
            + Send
            + Unpin
            + 'a,
    {
        self.total_frames = Some(total_frames);
        Box::pin(DeviceBlendshapeInteractiveFuture {
            executor: self,
            geometry: Some(geometry),
            all_geometry: None,
            frame,
            total_frames,
            callback,
            next_frame: 0,
            started: false,
            finished: false,
            generation_started: false,
            generation: 0,
            emitted_frames: 0,
        })
    }

    pub fn compute_all_frames<'a, C>(
        &'a mut self,
        geometry: &'a [RegressionGeometry],
        callback: C,
    ) -> ExecutorFuture<'a, InteractiveExecutionReport>
    where
        C: for<'r> FnMut(crate::animation::InteractiveGpuBlendshapeOutput<'r>) -> bool
            + Send
            + Unpin
            + 'a,
    {
        self.total_frames = Some(geometry.len());
        Box::pin(DeviceBlendshapeInteractiveFuture {
            executor: self,
            geometry: None,
            all_geometry: Some(geometry),
            frame: 0,
            total_frames: geometry.len(),
            callback,
            next_frame: 0,
            started: false,
            finished: false,
            generation_started: false,
            generation: 0,
            emitted_frames: 0,
        })
    }

    fn timestamp(&self, frame: usize) -> Result<i64> {
        let numerator = self.frame_rate.numerator() as u128;
        let denominator = self.frame_rate.denominator() as u128;
        let samples = (frame as u128)
            .checked_mul(self.sample_rate as u128)
            .and_then(|value| value.checked_mul(denominator))
            .and_then(|value| value.checked_div(numerator))
            .ok_or(crate::Error::InvalidArgument {
                field: "frame",
                reason: "frame timestamp overflow".into(),
            })?;
        i64::try_from(samples).map_err(|_| crate::Error::InvalidArgument {
            field: "frame",
            reason: "frame timestamp exceeds sample range".into(),
        })
    }
}

#[cfg(feature = "cuda")]
struct DeviceBlendshapeInteractiveFuture<'a, C> {
    executor: &'a mut DeviceBlendshapeSolveInteractiveExecutor,
    geometry: Option<&'a RegressionGeometry>,
    all_geometry: Option<&'a [RegressionGeometry]>,
    frame: usize,
    total_frames: usize,
    callback: C,
    next_frame: usize,
    started: bool,
    finished: bool,
    generation_started: bool,
    generation: u64,
    emitted_frames: usize,
}

#[cfg(feature = "cuda")]
impl<C> std::future::Future for DeviceBlendshapeInteractiveFuture<'_, C>
where
    C: for<'r> FnMut(crate::animation::InteractiveGpuBlendshapeOutput<'r>) -> bool + Unpin,
{
    type Output = Result<InteractiveExecutionReport>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        if this.finished {
            panic!("interactive GPU BlendShape future polled after completion");
        }
        if !this.generation_started {
            this.generation = this.executor.interrupt.generation();
            this.generation_started = true;
        }
        if this
            .executor
            .interrupt
            .is_interrupted_since(this.generation)
        {
            this.finished = true;
            return std::task::Poll::Ready(Ok(InteractiveExecutionReport {
                status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                emitted_frames: this.emitted_frames,
            }));
        }
        if this.all_geometry.is_some() && !this.started {
            if let Err(error) = this.executor.layer.begin_all_frames(this.total_frames) {
                this.finished = true;
                return std::task::Poll::Ready(Err(error));
            }
            this.started = true;
        }
        let result = if let Some(geometry) = this.geometry {
            this.executor
                .layer
                .compute_frame(this.frame, this.total_frames, geometry, |output| {
                    (this.callback)(output)
                })
        } else if let Some(geometry) = this.all_geometry {
            if this.next_frame >= geometry.len() {
                this.finished = true;
                return std::task::Poll::Ready(Ok(InteractiveExecutionReport {
                    status: crate::audio2x::InteractiveExecutionStatus::Complete,
                    emitted_frames: this.emitted_frames,
                }));
            }
            let frame = this.next_frame;
            let result =
                this.executor
                    .layer
                    .compute_next_frame(frame, &geometry[frame], |output| (this.callback)(output));
            this.next_frame += 1;
            result
        } else {
            Err(crate::Error::InvalidState {
                operation: "interactive GPU BlendShape compute",
                state: "no geometry was supplied",
            })
        };
        let keep_going = match result {
            Ok(keep_going) => keep_going,
            Err(error) => {
                this.finished = true;
                return std::task::Poll::Ready(Err(error));
            }
        };
        this.emitted_frames += 1;
        if !keep_going
            || this
                .executor
                .interrupt
                .is_interrupted_since(this.generation)
            || this.geometry.is_some()
            || this.next_frame >= this.total_frames
        {
            this.finished = true;
            let status = if !keep_going
                || this
                    .executor
                    .interrupt
                    .is_interrupted_since(this.generation)
            {
                crate::audio2x::InteractiveExecutionStatus::Interrupted
            } else {
                crate::audio2x::InteractiveExecutionStatus::Complete
            };
            return std::task::Poll::Ready(Ok(InteractiveExecutionReport {
                status,
                emitted_frames: this.emitted_frames,
            }));
        }
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    }
}

#[cfg(feature = "cuda")]
impl<C> Drop for DeviceBlendshapeInteractiveFuture<'_, C> {
    fn drop(&mut self) {
        // The GPU layer keeps cached device buffers alive; unfinished ordered
        // frames remain absent and therefore report invalidity.
    }
}

#[cfg(feature = "cuda")]
impl InteractiveExecutor for DeviceBlendshapeSolveInteractiveExecutor {
    fn invalidate_all(&mut self) -> Result<()> {
        self.layer
            .invalidate(crate::animation::BlendshapeInvalidationLayer::All);
        Ok(())
    }

    fn is_fully_valid(&self) -> bool {
        self.layer
            .is_valid(crate::animation::BlendshapeInvalidationLayer::All)
    }

    fn total_frame_count(&self) -> Result<usize> {
        self.total_frames.ok_or(crate::Error::InvalidState {
            operation: "query interactive GPU BlendShape frame count",
            state: "no computation has established the input timeline",
        })
    }

    fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    fn frame_rate(&self) -> crate::audio2x::FrameRate {
        self.frame_rate
    }

    fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        let total = self.total_frame_count()?;
        if frame >= total {
            return Err(crate::Error::OutOfBounds {
                field: "frame",
                index: frame,
                len: total,
            });
        }
        self.timestamp(frame)
    }

    fn interrupt_handle(&self) -> InteractiveInterruptHandle {
        self.interrupt.clone()
    }
}

#[cfg(feature = "cuda")]
impl BlendshapeInteractiveExecutor for DeviceBlendshapeSolveInteractiveExecutor {
    fn invalidate_blendshape(&mut self, layer: BlendshapeInvalidationLayer) -> Result<()> {
        match layer {
            BlendshapeInvalidationLayer::None => {}
            BlendshapeInvalidationLayer::SkinSolverPrepare => self
                .layer
                .invalidate(crate::animation::BlendshapeInvalidationLayer::SkinSolverPrepare),
            BlendshapeInvalidationLayer::TongueSolverPrepare => self
                .layer
                .invalidate(crate::animation::BlendshapeInvalidationLayer::TongueSolverPrepare),
            _ => self
                .layer
                .invalidate(crate::animation::BlendshapeInvalidationLayer::Weights),
        }
        Ok(())
    }

    fn is_blendshape_valid(&self, layer: BlendshapeInvalidationLayer) -> bool {
        match layer {
            BlendshapeInvalidationLayer::SkinSolverPrepare => self
                .layer
                .is_valid(crate::animation::BlendshapeInvalidationLayer::SkinSolverPrepare),
            BlendshapeInvalidationLayer::TongueSolverPrepare => self
                .layer
                .is_valid(crate::animation::BlendshapeInvalidationLayer::TongueSolverPrepare),
            BlendshapeInvalidationLayer::None => true,
            _ => self
                .layer
                .is_valid(crate::animation::BlendshapeInvalidationLayer::Weights),
        }
    }

    fn weight_count(&self) -> usize {
        self.layer
            .skin_solver()
            .map_or(0, crate::animation::GpuBlendshapeSolver::pose_count)
            + self
                .layer
                .tongue_solver()
                .map_or(0, crate::animation::GpuBlendshapeSolver::pose_count)
    }

    fn result_kind(&self) -> BlendshapeResultKind {
        BlendshapeResultKind::Device
    }
}

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

    fn cuda_stream(&self) -> &crate::cuda::CudaStream {
        match self {
            Self::Regression(source) => source.cuda_stream(),
            Self::Diffusion(source) => source.cuda_stream(),
        }
    }

    fn audio_accumulator(&self, track: usize) -> Result<&Arc<AudioAccumulator>> {
        match self {
            Self::Regression(source) => source.audio_accumulator(track),
            Self::Diffusion(source) => source.audio_accumulator(track),
        }
    }

    fn emotion_accumulator(&self, track: usize) -> Result<&Arc<EmotionAccumulator>> {
        match self {
            Self::Regression(source) => source.emotion_accumulator(track),
            Self::Diffusion(source) => source.emotion_accumulator(track),
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

    /// Returns the geometry result stream retained by this owning executor.
    pub fn cuda_stream(&self) -> &crate::cuda::CudaStream {
        self.source.cuda_stream()
    }

    /// Returns the exact shared audio accumulator transferred with geometry.
    pub fn audio_accumulator(&self, track: usize) -> Result<&Arc<AudioAccumulator>> {
        self.source.audio_accumulator(track)
    }

    /// Returns the exact shared emotion accumulator transferred with geometry.
    pub fn emotion_accumulator(&self, track: usize) -> Result<&Arc<EmotionAccumulator>> {
        self.source.emotion_accumulator(track)
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

    /// Returns the solver stream owned by this device-result executor.
    pub fn cuda_stream(&self) -> &crate::cuda::CudaStream {
        &self.stream
    }

    /// Returns the exact shared audio accumulator transferred with geometry.
    pub fn audio_accumulator(&self, track: usize) -> Result<&Arc<AudioAccumulator>> {
        self.source.audio_accumulator(track)
    }

    /// Returns the exact shared emotion accumulator transferred with geometry.
    pub fn emotion_accumulator(&self, track: usize) -> Result<&Arc<EmotionAccumulator>> {
        self.source.emotion_accumulator(track)
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

#[cfg(all(test, feature = "cuda"))]
mod interactive_blendshape_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Waker};

    #[derive(Default)]
    struct HoldingRunner {
        tasks: Mutex<Vec<JobRunnerTask>>,
    }

    impl HoldingRunner {
        fn run_one(&self) {
            self.tasks.lock().unwrap().remove(0).run();
        }
    }

    impl JobRunner for HoldingRunner {
        fn enqueue(&self, task: JobRunnerTask) -> Result<()> {
            self.tasks.lock().unwrap().push(task);
            Ok(())
        }
    }

    fn geometry() -> RegressionGeometry {
        RegressionGeometry {
            skin: Vec::new(),
            tongue: Vec::new(),
            jaw_transform: [0.0; 16],
            eyes_rotation: crate::animation::EyesRotation {
                right: [0.0; 3],
                left: [0.0; 3],
            },
        }
    }

    fn executor(runner: Arc<HoldingRunner>) -> HostBlendshapeSolveInteractiveExecutor {
        let shared: Arc<dyn JobRunner> = runner;
        HostBlendshapeSolveInteractiveExecutor::from_layer_with_runner(
            InteractiveBlendshapeLayer::new(None, None),
            48_000,
            crate::audio2x::FrameRate::new(30, 1).unwrap(),
            Some(shared),
        )
        .unwrap()
    }

    #[test]
    fn host_future_waits_for_its_job_and_callback() {
        let runner = Arc::new(HoldingRunner::default());
        let mut executor = executor(Arc::clone(&runner));
        let callbacks = Arc::new(AtomicUsize::new(0));
        let callbacks_for_call = Arc::clone(&callbacks);
        let frame = geometry();
        let mut future = executor.compute_frame(0, 1, &frame, move |_| {
            callbacks_for_call.fetch_add(1, Ordering::SeqCst);
            true
        });
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(callbacks.load(Ordering::SeqCst), 0);
        runner.run_one();
        assert!(matches!(
            future.as_mut().poll(&mut context),
            Poll::Ready(Ok(InteractiveExecutionReport {
                status: crate::audio2x::InteractiveExecutionStatus::Complete,
                emitted_frames: 1,
            }))
        ));
        assert_eq!(callbacks.load(Ordering::SeqCst), 1);
        drop(future);
        assert!(executor.is_fully_valid());
    }

    #[test]
    fn dropping_host_future_leaves_unfinished_frames_invalid_and_no_callback() {
        let runner = Arc::new(HoldingRunner::default());
        let mut executor = executor(Arc::clone(&runner));
        let callbacks = Arc::new(AtomicUsize::new(0));
        let callbacks_for_call = Arc::clone(&callbacks);
        let frames = [geometry(), geometry()];
        let mut future = executor.compute_all_frames(&frames, move |_| {
            callbacks_for_call.fetch_add(1, Ordering::SeqCst);
            true
        });
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        drop(future);
        runner.run_one();
        assert_eq!(callbacks.load(Ordering::SeqCst), 0);
        assert!(!executor.is_fully_valid());
    }

    #[test]
    fn host_interrupt_is_generation_scoped_and_checked_after_final_callback() {
        let runner = Arc::new(HoldingRunner::default());
        let mut executor = executor(Arc::clone(&runner));
        let interrupt = executor.interrupt_handle();
        interrupt.interrupt();
        let interrupt_from_callback = interrupt.clone();
        let frame = geometry();
        let mut future = executor.compute_frame(0, 1, &frame, move |_| {
            interrupt_from_callback.interrupt();
            true
        });
        let waker = Waker::noop();
        let mut context = Context::from_waker(waker);
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        runner.run_one();
        assert!(matches!(
            future.as_mut().poll(&mut context),
            Poll::Ready(Ok(InteractiveExecutionReport {
                status: crate::audio2x::InteractiveExecutionStatus::Interrupted,
                emitted_frames: 1,
            }))
        ));
    }
}
