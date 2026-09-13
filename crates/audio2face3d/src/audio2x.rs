//! Portable executor contracts corresponding to the SDK's `nva2x` namespace.
//!
//! These contracts are available without CUDA, TensorRT, or an async runtime.
//! Standard execution starts synchronously and returns [`Execution`]; interactive
//! computation returns [`ExecutorFuture`]. Callbacks remain synchronous closures.
//!
//! Standard executor construction remains model-specific, while this module
//! provides the runtime-independent completion primitive used by all owning
//! facades. No async runtime is required to poll an [`Execution`].
//!
//! # Migration and correspondence
//!
//! | Existing Rust API | SDK-facing contract | SDK header |
//! |---|---|---|
//! | `common::AudioAccumulator` | [`AudioAccumulator`] | `audio2x/audio_accumulator.h` |
//! | `common::EmotionAccumulator` | [`EmotionAccumulator`] | `audio2x/emotion_accumulator.h` |
//! | `common::FloatAccumulator` | [`FloatAccumulator`] | `audio2x/float_accumulator.h` |
//! | Separate scheduler query methods | [`Executor`] | `audio2x/executor.h` |
//! | Separate interactive query methods | [`InteractiveExecutor`] | `audio2x/interactive_executor.h` |
//!
//! Accumulators are shared inputs; completed executors uniquely own their state.
//!
//! Completion can be awaited on any runtime. The trait remains dyn-compatible:
//!
//! ```
//! use audio2face3d::audio2x::{Execution, Executor, ExecutionReport, Result};
//!
//! fn track_count(executor: &dyn Executor) -> usize {
//!     executor.track_count()
//! }
//!
//! async fn finish(mut execution: Execution) -> Result<ExecutionReport> {
//!     // Host completion for track zero, not a CUDA device synchronization.
//!     execution.wait_track(0).await?;
//!     execution.await
//! }
//! ```

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

pub use crate::common::{
    AudioAccumulator, EmotionAccumulator, Error, FloatAccumulator, Result, Timestamp,
};
pub use crate::cuda::{CudaStreamRef, DeviceView};

/// A runtime-independent, dynamically dispatched, sendable completion future.
///
/// The borrow covers the executor and callback, not an individual callback result.
/// Corresponds to the host completion boundary of SDK `ComputeFrame`/`ComputeAllFrames`
/// and `Wait` methods; C++ does not expose a Future type.
pub type ExecutorFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T>> + Send + 'a>>;

/// Starts a model factory on a dedicated worker and returns a runtime-neutral
/// completion future.
///
/// Model file I/O, TensorRT deserialization, and CUDA initialization are
/// deliberately kept out of [`Future::poll`]. Dropping the returned future
/// only drops the observation handle; the worker keeps owning its closure and
/// therefore releases any partially-created resources when construction
/// finishes or fails.
#[allow(dead_code)]
pub(crate) fn spawn_blocking_factory<T, F>(load: F) -> ExecutorFuture<'static, T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    let state = Arc::new(FactoryState {
        state: Mutex::new(FactoryCompletionState {
            result: None,
            waker: None,
        }),
    });
    let worker_state = Arc::clone(&state);
    if let Err(error) = std::thread::Builder::new()
        .name("audio2face3d-factory".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(load))
                .unwrap_or_else(|panic| {
                    let message = panic
                        .downcast_ref::<&str>()
                        .map(|message| (*message).to_owned())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "unknown panic payload".to_owned());
                    Err(Error::Io {
                        operation: "run factory worker",
                        message: format!("factory worker panicked: {message}"),
                    })
                });
            let waker = {
                let mut state = worker_state
                    .state
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state.result = Some(result);
                state.waker.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        })
    {
        state
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .result = Some(Err(Error::Io {
            operation: "spawn factory worker",
            message: error.to_string(),
        }));
    }
    Box::pin(FactoryFuture { state })
}

#[allow(dead_code)]
struct FactoryState<T> {
    state: Mutex<FactoryCompletionState<T>>,
}

#[allow(dead_code)]
struct FactoryCompletionState<T> {
    result: Option<Result<T>>,
    waker: Option<Waker>,
}

#[allow(dead_code)]
struct FactoryFuture<T> {
    state: Arc<FactoryState<T>>,
}

impl<T> Future for FactoryFuture<T> {
    type Output = Result<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let state = &self.get_mut().state;
        let mut completion = state
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(result) = completion.result.take() {
            return Poll::Ready(result);
        }
        completion.waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

/// A positive rational frame rate in frames per second.
///
/// Corresponds to `nva2x::IExecutor::GetFrameRate` in
/// `audio2x-common/include/audio2x/executor.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    numerator: usize,
    denominator: usize,
}

impl FrameRate {
    /// Constructs a rate without rounding fractional frame rates.
    pub fn new(numerator: usize, denominator: usize) -> Result<Self> {
        if numerator == 0 || denominator == 0 {
            return Err(Error::InvalidArgument {
                field: "frame_rate",
                reason: "numerator and denominator must be non-zero".into(),
            });
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub const fn numerator(self) -> usize {
        self.numerator
    }
    pub const fn denominator(self) -> usize {
        self.denominator
    }
}

/// A parameter's default, bounds, and human-readable description.
///
/// Corresponds to `nva2f::RangeConfig<T>` in
/// `audio2face-sdk/include/audio2face/animator.h` and is also exported by `audio2face`.
#[derive(Debug, Clone, Copy)]
pub struct RangeConfig<T> {
    pub default_value: T,
    pub minimum: T,
    pub maximum: T,
    pub description: &'static str,
}

impl<T: PartialEq> PartialEq for RangeConfig<T> {
    fn eq(&self, other: &Self) -> bool {
        self.default_value == other.default_value
            && self.minimum == other.minimum
            && self.maximum == other.maximum
    }
}

impl<T: Eq> Eq for RangeConfig<T> {}

/// Common track, readiness, reset, and time queries for standard execution.
///
/// Corresponds to `nva2x::IExecutor` in `audio2x-common/include/audio2x/executor.h`.
/// Completed SDK executors are uniquely owned, `Send`, and not `Sync` or `Clone`.
/// Execution callbacks belong to specialized traits, so this trait is dyn-compatible.
pub trait Executor: Send {
    fn track_count(&self) -> usize;

    /// Resets only this track's executor state, preserving accumulated input.
    /// Pending host work must complete before reset; dropped input history cannot
    /// be restored by resetting an executor.
    fn reset_track(&mut self, track: usize) -> Result<()>;
    fn has_execution_started(&self, track: usize) -> Result<bool>;

    /// A snapshot lower bound; concurrent producers may add more input.
    fn available_execution_count(&self, track: usize) -> Result<usize>;
    fn ready_track_count(&self) -> usize;

    /// `None` while input is open, including when zero frames are currently ready.
    fn total_frame_count(&self, track: usize) -> Result<Option<usize>>;
    fn sample_rate(&self) -> usize;
    fn frame_rate(&self) -> FrameRate;

    /// Timestamp in signed audio sample units, not milliseconds.
    fn frame_timestamp(&self, frame: usize) -> Result<i64>;
}

/// Common synchronous queries and invalidation for interactive execution.
///
/// Corresponds to `nva2x::IInteractiveExecutor` in
/// `audio2x-common/include/audio2x/interactive_executor.h`.
/// Inputs must be closed and retain the required history. Specialized traits
/// provide output-specific compute Futures and typed layer invalidation.
pub trait InteractiveExecutor: Send {
    fn invalidate_all(&mut self) -> Result<()>;
    fn is_fully_valid(&self) -> bool;
    fn total_frame_count(&self) -> Result<usize>;
    fn sample_rate(&self) -> usize;
    fn frame_rate(&self) -> FrameRate;
    fn frame_timestamp(&self, frame: usize) -> Result<i64>;
    fn interrupt_handle(&self) -> InteractiveInterruptHandle;
}

/// Normal state after one standard execution; these states are not failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionState {
    Progress,
    AwaitingInput,
    Complete,
}

/// Host completion report for one call, independent of GPU completion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionReport {
    pub state: ExecutionState,
    /// Active tracks executed, not the fixed TensorRT batch size.
    pub executed_tracks: usize,
    pub emitted_frames: usize,
}

/// Normal interactive completion or cooperative interruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InteractiveExecutionStatus {
    Complete,
    Interrupted,
}

/// Report returned after the requested host computation and callbacks finish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InteractiveExecutionReport {
    pub status: InteractiveExecutionStatus,
    pub emitted_frames: usize,
}

/// Common original-track metadata, never a packed batch slot index.
///
/// Corresponds to callback track/frame timestamps in
/// `audio2face-sdk/include/audio2face/executor.h` and
/// `audio2emotion-sdk/include/audio2emotion/executor.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CallbackMetadata {
    pub track_index: usize,
    pub frame_index: usize,
    /// Signed sample index in the executor's sampling rate.
    pub timestamp: i64,
    pub next_timestamp: i64,
}

/// One component's borrowed device values and producer stream.
///
/// Corresponds to component tensor/stream pairs in `nva2f::IGeometryExecutor`
/// (`audio2face-sdk/include/audio2face/executor.h`). No host copy or stream
/// synchronization is implied. Both borrows end with the result callback;
/// continued use requires an explicit copy to caller-owned storage and a fence.
pub struct DeviceComponentResults<'a> {
    pub values: DeviceView<'a, f32>,
    pub stream: CudaStreamRef<'a>,
}

#[cfg(feature = "cuda")]
impl DeviceComponentResults<'_> {
    /// Copies this callback-scoped device result into caller-owned host memory.
    ///
    /// The copy is enqueued on the result's producer stream and synchronized
    /// before returning, so `destination` is immediately ready for CPU use.
    pub fn copy_to(&self, destination: &mut [f32]) -> Result<()> {
        crate::cuda::copy_device_view_to_host(self.values, destination, self.stream)
    }
}

/// Completion handle for a single synchronously started execution.
///
/// Corresponds to `Execute`/host BlendShape `Wait(track)` completion in
/// `audio2face-sdk/include/audio2face/executor.h` (with runtime-independent await).
/// Awaiting this handle waits for all host jobs/callbacks belonging to its call,
/// not unrelated jobs on a shared runner and not CUDA stream completion.
/// Dropping it detaches observation; it does not cancel started work. The executor
/// retains work/resources and drains outstanding jobs before its own destruction.
/// Worker errors are returned by await and must not unwind across worker/FFI boundaries.
///
/// The handle only observes a call. Dropping it detaches observation; the
/// completion state remains owned by the executor/runner resources that
/// created it.
#[must_use = "await execution to observe host completion and worker errors"]
pub struct Execution {
    completion: Arc<ExecutionCompletion>,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl Execution {
    /// Geometry device execution completes its callbacks before returning.
    #[cfg(feature = "tensorrt")]
    pub(crate) fn into_ready_report(mut self) -> Result<ExecutionReport> {
        let mut context = Context::from_waker(Waker::noop());
        match Pin::new(&mut self).poll(&mut context) {
            Poll::Ready(result) => result,
            Poll::Pending => Err(Error::InvalidSchema(
                "device execution unexpectedly pending".into(),
            )),
        }
    }
    /// Creates an execution whose schedule is finished by
    /// [`ExecutionCompletion::finish_schedule`]. This is crate-private because
    /// only completed owning executors may create public execution handles.
    #[allow(dead_code)]
    pub(crate) fn pending(track_count: usize) -> (Self, Arc<ExecutionCompletion>) {
        let completion = Arc::new(ExecutionCompletion::new(track_count));
        (
            Self {
                completion: Arc::clone(&completion),
                _not_sync: PhantomData,
            },
            completion,
        )
    }

    /// Creates an already completed execution for synchronous/device paths.
    #[allow(dead_code)]
    pub(crate) fn ready(report: ExecutionReport) -> Self {
        let (execution, completion) = Self::pending(0);
        completion.finish_schedule(report);
        execution
    }

    /// Waits only for this call's jobs and callbacks for `track`.
    /// Out-of-range tracks are reported as typed failures by the completion layer.
    pub fn wait_track(&mut self, track: usize) -> ExecutorFuture<'_, ()> {
        Box::pin(TrackCompletionFuture {
            completion: Arc::clone(&self.completion),
            track,
            checked: false,
        })
    }

    /// Equivalent to awaiting this handle directly; consumes the call observer.
    pub fn wait_all(self) -> ExecutorFuture<'static, ExecutionReport> {
        Box::pin(self)
    }
}

impl Future for Execution {
    type Output = Result<ExecutionReport>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.get_mut().completion.poll_all(cx)
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        self.completion.detach();
    }
}

/// Shared completion state used by standard executions and host worker tasks.
///
/// It deliberately uses only `Mutex` and `Waker`; no executor/runtime is
/// involved. Registration and completion both hold the same lock while doing
/// their state transition, so a completion racing with `poll` cannot lose a
/// wakeup.
pub(crate) struct ExecutionCompletion {
    state: Mutex<CompletionState>,
    changed: Condvar,
}

struct CompletionState {
    pending_by_track: Vec<usize>,
    errors_by_track: Vec<Option<Error>>,
    report: Option<ExecutionReport>,
    detached: bool,
    all_observed: bool,
    all_wakers: Vec<Waker>,
    track_wakers: Vec<Vec<Waker>>,
}

#[allow(dead_code)]
impl ExecutionCompletion {
    fn new(track_count: usize) -> Self {
        Self {
            state: Mutex::new(CompletionState {
                pending_by_track: vec![0; track_count],
                errors_by_track: vec![None; track_count],
                report: None,
                detached: false,
                all_observed: false,
                all_wakers: Vec::new(),
                track_wakers: (0..track_count).map(|_| Vec::new()).collect(),
            }),
            changed: Condvar::new(),
        }
    }

    pub(crate) fn add_task(&self, track: usize) -> Result<()> {
        let mut state = self.lock()?;
        if track >= state.pending_by_track.len() {
            return Err(Error::OutOfBounds {
                field: "track",
                index: track,
                len: state.pending_by_track.len(),
            });
        }
        state.pending_by_track[track] += 1;
        Ok(())
    }

    /// Marks enqueueing complete. The report is held until all accepted tasks
    /// complete, allowing a task to finish before the scheduler returns.
    pub(crate) fn finish_schedule(&self, report: ExecutionReport) {
        let (all, tracks) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.report = Some(report);
            self.ready_wakers_locked(&mut state)
        };
        self.changed.notify_all();
        wake_all(all, tracks);
    }

    pub(crate) fn complete_task(&self, track: usize, result: Result<()>) {
        let (all, tracks) = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let Some(pending) = state.pending_by_track.get_mut(track) else {
                return;
            };
            // A task is expected to be registered before it is handed to a
            // runner. Ignore an unregistered task rather than allowing it to
            // poison an unrelated execution.
            if *pending == 0 {
                return;
            }
            *pending -= 1;
            if let Err(error) = result {
                if state.detached {
                    tracing::warn!(track, error = %error, "detached execution worker failed");
                }
                if state.errors_by_track[track].is_none() {
                    state.errors_by_track[track] = Some(error);
                }
            }
            self.ready_wakers_locked(&mut state)
        };
        self.changed.notify_all();
        wake_all(all, tracks);
    }

    fn poll_all(&self, cx: &mut Context<'_>) -> Poll<Result<ExecutionReport>> {
        let mut state = self.lock().expect("completion mutex poisoned");
        if all_ready(&state) {
            state.all_observed = true;
            return Poll::Ready(all_result(&state));
        }
        state.all_wakers.push(cx.waker().clone());
        // The lock is still held here. A completing worker either observes the
        // newly registered waker or waits until registration is complete.
        if all_ready(&state) {
            let waker = state.all_wakers.pop();
            drop(state);
            if let Some(waker) = waker {
                waker.wake();
            }
        }
        Poll::Pending
    }

    fn detach(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.all_observed || state.detached {
            return;
        }
        state.detached = true;
        for (track, error) in state.errors_by_track.iter().enumerate() {
            if let Some(error) = error {
                tracing::warn!(track, error = %error, "execution dropped with an unobserved worker error");
            }
        }
    }

    pub(crate) fn track_pending(&self, track: usize) -> Result<bool> {
        let state = self.lock()?;
        state
            .pending_by_track
            .get(track)
            .map(|pending| *pending != 0)
            .ok_or(Error::OutOfBounds {
                field: "track",
                index: track,
                len: state.pending_by_track.len(),
            })
    }

    pub(crate) fn wait_blocking(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while !all_ready(&state) {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }

    fn poll_track(&self, track: usize, cx: &mut Context<'_>) -> Poll<Result<()>> {
        let mut state = self.lock().expect("completion mutex poisoned");
        if track >= state.pending_by_track.len() {
            return Poll::Ready(Err(Error::OutOfBounds {
                field: "track",
                index: track,
                len: state.pending_by_track.len(),
            }));
        }
        if track_ready(&state, track) {
            return Poll::Ready(track_result(&state, track));
        }
        state.track_wakers[track].push(cx.waker().clone());
        if track_ready(&state, track) {
            let waker = state.track_wakers[track].pop();
            drop(state);
            if let Some(waker) = waker {
                waker.wake();
            }
        }
        Poll::Pending
    }

    fn lock(&self) -> Result<MutexGuard<'_, CompletionState>> {
        self.state.lock().map_err(|_| Error::Poisoned {
            resource: "execution completion",
        })
    }

    fn ready_wakers_locked(&self, state: &mut CompletionState) -> (Vec<Waker>, Vec<Waker>) {
        let mut all = Vec::new();
        let mut tracks = Vec::new();
        if all_ready(state) {
            all.append(&mut state.all_wakers);
        }
        for track in 0..state.pending_by_track.len() {
            if track_ready(state, track) {
                tracks.append(&mut state.track_wakers[track]);
            }
        }
        (all, tracks)
    }
}

fn all_ready(state: &CompletionState) -> bool {
    state.report.is_some() && state.pending_by_track.iter().all(|pending| *pending == 0)
}

fn track_ready(state: &CompletionState, track: usize) -> bool {
    state.report.is_some() && state.pending_by_track[track] == 0
}

fn track_result(state: &CompletionState, track: usize) -> Result<()> {
    state.errors_by_track[track].clone().map_or(Ok(()), Err)
}

fn all_result(state: &CompletionState) -> Result<ExecutionReport> {
    if let Some(error) = state.errors_by_track.iter().flatten().next() {
        return Err(error.clone());
    }
    state.report.ok_or(Error::InvalidState {
        operation: "wait for execution",
        state: "schedule is not finished",
    })
}

#[allow(dead_code)]
fn wake_all(all: Vec<Waker>, tracks: Vec<Waker>) {
    for waker in all.into_iter().chain(tracks) {
        waker.wake();
    }
}

struct TrackCompletionFuture {
    completion: Arc<ExecutionCompletion>,
    track: usize,
    checked: bool,
}

impl Future for TrackCompletionFuture {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.checked {
            // A completed track future is never polled again by compliant
            // callers. Re-polling remains deterministic and non-blocking.
            return self.completion.poll_track(self.track, cx);
        }
        self.checked = true;
        self.completion.poll_track(self.track, cx)
    }
}

/// A clonable thread-safe request handle for the currently active compute call.
///
/// Corresponds to `nva2x::IInteractiveExecutor::Interrupt` in
/// `audio2x-common/include/audio2x/interactive_executor.h`.
/// Each request advances a generation: calls ignore requests preceding their
/// start and observe generation changes at safe inference/layer boundaries.
/// The completed executor creates the handle; consumers only clone and interrupt it.
/// A future captures the internal generation when it starts; requests issued
/// before that capture are intentionally ignored by that call.
#[derive(Debug, Clone)]
pub struct InteractiveInterruptHandle {
    generation: Arc<AtomicU64>,
}

#[allow(dead_code)]
impl InteractiveInterruptHandle {
    pub(crate) fn new() -> Self {
        Self {
            generation: Arc::new(AtomicU64::new(0)),
        }
    }

    pub fn interrupt(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Returns the generation observed when a call begins.
    ///
    /// Requests made before this value is captured do not carry over into the
    /// new call. A computation should retain this snapshot for its lifetime
    /// and use [`Self::is_interrupted_since`] at safe boundaries.
    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Returns whether an interrupt was requested after `generation`.
    pub(crate) fn is_interrupted_since(&self, generation: u64) -> bool {
        self.generation.load(Ordering::Acquire) != generation
    }
}

/// Failed ownership transfer, returning the original executor to the caller.
///
/// Corresponds to the Geometry-to-BlendShape ownership-transfer factories in
/// `audio2face-sdk/include/audio2face/audio2face.h`. No high-cost model reload
/// is required to recover the original owner after a failed conversion.
#[derive(Debug)]
pub struct TransferError<T> {
    pub error: Error,
    pub original: T,
}

impl<T> TransferError<T> {
    pub fn into_parts(self) -> (Error, T) {
        (self.error, self.original)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    struct CountingWaker(Arc<AtomicUsize>);

    impl Wake for CountingWaker {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }

        fn wake_by_ref(self: &Arc<Self>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn report() -> ExecutionReport {
        ExecutionReport {
            state: ExecutionState::Progress,
            executed_tracks: 2,
            emitted_frames: 2,
        }
    }

    #[test]
    fn frame_rate_preserves_fraction_and_rejects_zero() {
        let rate = FrameRate::new(30_000, 1_001).unwrap();
        assert_eq!((rate.numerator(), rate.denominator()), (30_000, 1_001));
        assert!(matches!(
            FrameRate::new(0, 1),
            Err(Error::InvalidArgument { .. })
        ));
        assert!(matches!(
            FrameRate::new(1, 0),
            Err(Error::InvalidArgument { .. })
        ));
    }

    #[test]
    fn transfer_failure_returns_unique_original_owner() {
        let owner = Box::new(42);
        let address = std::ptr::from_ref(owner.as_ref());
        let failure = TransferError {
            error: Error::InvalidState {
                operation: "transfer",
                state: "pending work",
            },
            original: owner,
        };
        let (_, returned) = failure.into_parts();
        assert_eq!(std::ptr::from_ref(returned.as_ref()), address);
    }

    #[test]
    fn track_wait_is_local_while_all_waits_for_every_track() {
        let (mut execution, completion) = Execution::pending(2);
        completion.add_task(0).unwrap();
        completion.add_task(0).unwrap();
        completion.add_task(1).unwrap();
        completion.finish_schedule(report());

        let wake_count = Arc::new(AtomicUsize::new(0));
        let waker = Waker::from(Arc::new(CountingWaker(Arc::clone(&wake_count))));
        let mut cx = Context::from_waker(&waker);
        let mut track_zero = execution.wait_track(0);
        assert!(matches!(
            Pin::new(&mut track_zero).poll(&mut cx),
            Poll::Pending
        ));

        completion.complete_task(0, Ok(()));
        assert!(matches!(
            Pin::new(&mut track_zero).poll(&mut cx),
            Poll::Pending
        ));
        completion.complete_task(0, Ok(()));
        assert!(matches!(
            Pin::new(&mut track_zero).poll(&mut cx),
            Poll::Ready(Ok(()))
        ));
        drop(track_zero);
        assert!(matches!(
            Pin::new(&mut execution).poll(&mut cx),
            Poll::Pending
        ));

        completion.complete_task(1, Ok(()));
        assert!(
            matches!(Pin::new(&mut execution).poll(&mut cx), Poll::Ready(Ok(value)) if value == report())
        );
        assert!(wake_count.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn completion_before_poll_is_ready_and_completion_after_poll_wakes() {
        let (mut ready_execution, ready_completion) = Execution::pending(1);
        ready_completion.finish_schedule(report());
        let counter = Arc::new(AtomicUsize::new(0));
        let waker = Waker::from(Arc::new(CountingWaker(Arc::clone(&counter))));
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(
            Pin::new(&mut ready_execution).poll(&mut cx),
            Poll::Ready(Ok(_))
        ));

        let (mut pending_execution, pending_completion) = Execution::pending(1);
        pending_completion.add_task(0).unwrap();
        pending_completion.finish_schedule(report());
        assert!(matches!(
            Pin::new(&mut pending_execution).poll(&mut cx),
            Poll::Pending
        ));
        let completion_for_thread = Arc::clone(&pending_completion);
        std::thread::spawn(move || completion_for_thread.complete_task(0, Ok(())))
            .join()
            .unwrap();
        assert!(counter.load(Ordering::SeqCst) > 0);
        assert!(matches!(
            Pin::new(&mut pending_execution).poll(&mut cx),
            Poll::Ready(Ok(_))
        ));
    }

    #[tokio::test]
    async fn tokio_spawn_waits_for_execution_completed_by_std_thread() {
        let (execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        completion.finish_schedule(report());
        let (release, released) = std::sync::mpsc::channel();
        let completion_for_thread = Arc::clone(&completion);
        let worker = std::thread::spawn(move || {
            released.recv().unwrap();
            completion_for_thread.complete_task(0, Ok(()));
        });

        let task = tokio::spawn(execution);
        tokio::task::yield_now().await;
        release.send(()).unwrap();
        let completed = tokio::select! {
            result = task => result.expect("execution task panicked").unwrap(),
            _ = std::future::pending::<()>() => unreachable!(),
        };
        worker.join().unwrap();
        assert_eq!(completed, report());
    }

    #[tokio::test]
    async fn tokio_select_can_drop_observer_without_cancelling_completion() {
        let (mut execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        completion.finish_schedule(report());

        tokio::select! {
            biased;
            _ = std::future::ready(()) => {}
            result = &mut execution => panic!("pending execution unexpectedly completed: {result:?}"),
        }
        drop(execution);
        assert!(completion.state.lock().unwrap().detached);

        let completion_for_thread = Arc::clone(&completion);
        std::thread::spawn(move || completion_for_thread.complete_task(0, Ok(())))
            .join()
            .unwrap();
        assert!(!completion.track_pending(0).unwrap());
    }

    #[test]
    fn worker_error_is_retained_for_both_track_and_all_waits() {
        let (mut execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        completion.finish_schedule(report());
        completion.complete_task(
            0,
            Err(Error::Worker {
                track: 0,
                message: "injected failure".into(),
            }),
        );
        let waker = Waker::from(Arc::new(CountingWaker(Arc::new(AtomicUsize::new(0)))));
        let mut cx = Context::from_waker(&waker);
        let mut track_wait = execution.wait_track(0);
        assert!(matches!(
            Pin::new(&mut track_wait).poll(&mut cx),
            Poll::Ready(Err(Error::Worker { track: 0, .. }))
        ));
        drop(track_wait);
        assert!(matches!(
            Pin::new(&mut execution).poll(&mut cx),
            Poll::Ready(Err(Error::Worker { track: 0, .. }))
        ));
    }

    #[test]
    fn interactive_interrupt_is_generation_scoped() {
        let handle = InteractiveInterruptHandle::new();
        let before_call = handle.generation();
        handle.interrupt();
        let call_generation = handle.generation();
        assert!(handle.is_interrupted_since(before_call));
        assert!(!handle.is_interrupted_since(call_generation));
        handle.interrupt();
        assert!(handle.is_interrupted_since(call_generation));
    }

    fn wait_for_factory<T>(future: &mut ExecutorFuture<'static, T>) -> Result<T> {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let mut context = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(result) = future.as_mut().poll(&mut context) {
                return result;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "factory worker did not complete"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn factory_poll_is_non_blocking_and_worker_wakes_observer() {
        let (release, released) = std::sync::mpsc::channel();
        let mut future = spawn_blocking_factory(move || {
            released.recv().unwrap();
            Ok(17)
        });
        let wake_count = Arc::new(AtomicUsize::new(0));
        let waker = Waker::from(Arc::new(CountingWaker(Arc::clone(&wake_count))));
        let mut context = Context::from_waker(&waker);
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        release.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while wake_count.load(Ordering::SeqCst) == 0 {
            assert!(
                std::time::Instant::now() < deadline,
                "factory worker did not wake its observer"
            );
            std::thread::yield_now();
        }
        assert!(wake_count.load(Ordering::SeqCst) > 0);
        assert_eq!(wait_for_factory(&mut future).unwrap(), 17);
    }

    #[tokio::test]
    async fn tokio_spawn_and_select_wait_for_factory_worker() {
        let (release, released) = std::sync::mpsc::channel();
        let factory = spawn_blocking_factory(move || {
            released.recv().unwrap();
            Ok(23)
        });
        let task = tokio::spawn(factory);
        tokio::task::yield_now().await;
        release.send(()).unwrap();

        let value = tokio::select! {
            result = task => result.expect("factory task panicked").unwrap(),
            _ = std::future::pending::<()>() => unreachable!(),
        };
        assert_eq!(value, 23);
    }

    #[test]
    fn dropped_factory_future_releases_unobserved_result() {
        struct DropSignal(std::sync::mpsc::Sender<()>);
        impl Drop for DropSignal {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }

        let (release, released) = std::sync::mpsc::channel();
        let (dropped, observed_drop) = std::sync::mpsc::channel();
        let future = spawn_blocking_factory(move || {
            released.recv().unwrap();
            Ok(DropSignal(dropped))
        });
        drop(future);
        release.send(()).unwrap();
        observed_drop
            .recv_timeout(std::time::Duration::from_secs(5))
            .expect("worker must release a successful but unobserved result");
    }

    #[test]
    fn factory_worker_panic_completes_with_an_error() {
        let mut future = spawn_blocking_factory::<(), _>(|| panic!("injected factory panic"));
        assert!(matches!(
            wait_for_factory(&mut future),
            Err(Error::Io {
                operation: "run factory worker",
                ..
            })
        ));
    }
}
