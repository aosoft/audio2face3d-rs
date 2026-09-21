//! Host BlendShape job scheduling contract.
//!
//! Corresponds to `audio2face-sdk/include/audio2face/job_runner.h`.
//! Scheduling is runtime-independent: the default runner uses standard
//! threads and a condition-variable queue, while callers may inject any
//! `Arc<dyn JobRunner>` implementation.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::audio2x::{Error, ExecutionCompletion, Result};

type JobAction = Box<dyn FnOnce() -> Result<()> + Send + 'static>;

/// An opaque, owning, at-most-once job.
///
/// A runner accepts ownership through [`JobRunner::enqueue`]. Dropping an
/// accepted task without calling [`JobRunnerTask::run`] records cancellation
/// in the task's private completion state, so an execution cannot remain
/// pending forever when a custom runner rejects or silently drops work.
pub struct JobRunnerTask {
    scope: crate::logging::integration::LogScope,
    action: Option<JobAction>,
    completion: Option<Arc<ExecutionCompletion>>,
    track: usize,
}

impl JobRunnerTask {
    /// Runs this task at most once, consuming it.
    pub fn run(mut self) {
        let _scope = self.scope.activate();
        let Some(action) = self.action.take() else {
            return;
        };
        let result = match catch_unwind(AssertUnwindSafe(action)) {
            Ok(result) => result,
            Err(payload) => Err(Error::Worker {
                track: self.track,
                message: panic_message(payload),
            }),
        };
        if let Some(completion) = &self.completion {
            completion.complete_task(self.track, result);
        }
    }

    /// Creates a task for an owning executor. The public API intentionally
    /// exposes no callback/task constructor; only executor implementations can
    /// attach a completion state to a job.
    #[allow(dead_code)]
    pub(crate) fn new<F>(completion: Arc<ExecutionCompletion>, track: usize, action: F) -> Self
    where
        F: FnOnce() -> Result<()> + Send + 'static,
    {
        Self {
            scope: crate::logging::integration::LogScope::capture(),
            action: Some(Box::new(action)),
            completion: Some(completion),
            track,
        }
    }
}

impl Drop for JobRunnerTask {
    fn drop(&mut self) {
        let _scope = self.scope.activate();
        if self.action.take().is_some()
            && let Some(completion) = &self.completion
        {
            completion.complete_task(
                self.track,
                Err(Error::Worker {
                    track: self.track,
                    message: "job task was dropped before execution".into(),
                }),
            );
        }
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "worker panicked with a non-string payload".into()
    }
}

/// Object-safe scheduler shared by host BlendShape executors.
pub trait JobRunner: Send + Sync + 'static {
    /// Takes ownership of a task for asynchronous execution.
    fn enqueue(&self, task: JobRunnerTask) -> Result<()>;
}

/// Preserves a track's temporal solver state even with an unordered runner.
/// Only the drain task is scheduled; workers never wait on another worker.
#[cfg(any(test, feature = "tensorrt"))]
#[derive(Default)]
pub(crate) struct SerialJobQueue {
    state: Mutex<SerialQueueState>,
}

#[cfg(any(test, feature = "tensorrt"))]
#[derive(Default)]
struct SerialQueueState {
    tasks: VecDeque<JobRunnerTask>,
    scheduled: bool,
}

#[cfg(any(test, feature = "tensorrt"))]
impl SerialJobQueue {
    pub(crate) fn enqueue(
        self: &Arc<Self>,
        runner: &dyn JobRunner,
        task: JobRunnerTask,
    ) -> Result<()> {
        let mut state = self.state.lock().map_err(|_| Error::Poisoned {
            resource: "serial job queue",
        })?;
        state.tasks.push_back(task);
        if state.scheduled {
            return Ok(());
        }
        state.scheduled = true;
        drop(state);
        let guard = SerialDrainGuard {
            queue: Arc::clone(self),
            armed: true,
        };
        runner.enqueue(JobRunnerTask {
            scope: crate::logging::integration::LogScope::capture(),
            action: Some(Box::new(move || guard.run())),
            completion: None,
            track: 0,
        })
    }
}

#[cfg(any(test, feature = "tensorrt"))]
struct SerialDrainGuard {
    queue: Arc<SerialJobQueue>,
    armed: bool,
}

#[cfg(any(test, feature = "tensorrt"))]
impl SerialDrainGuard {
    fn run(mut self) -> Result<()> {
        loop {
            let task = {
                let mut state = self.queue.state.lock().map_err(|_| Error::Poisoned {
                    resource: "serial job queue",
                })?;
                match state.tasks.pop_front() {
                    Some(task) => task,
                    None => {
                        state.scheduled = false;
                        self.armed = false;
                        return Ok(());
                    }
                }
            };
            task.run();
        }
    }
}

#[cfg(any(test, feature = "tensorrt"))]
impl Drop for SerialDrainGuard {
    fn drop(&mut self) {
        if self.armed {
            let tasks = {
                let mut state = self
                    .queue
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                state.scheduled = false;
                std::mem::take(&mut state.tasks)
            };
            // Cancellation wakes observers; never invoke them under the queue lock.
            drop(tasks);
        }
    }
}

struct QueueState {
    tasks: VecDeque<JobRunnerTask>,
    shutdown: bool,
}

struct PoolInner {
    state: Mutex<QueueState>,
    available: Condvar,
}

/// Opaque default thread-pool runner.
///
/// The pool owns only scheduling resources. Solver state and completion are
/// owned by the task supplied by an executor. Drop signals shutdown, drains
/// queued work, and joins workers; it never stops an external/shared runner.
pub struct ThreadPoolJobRunner {
    scope: crate::logging::integration::LogScope,
    inner: Arc<PoolInner>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl ThreadPoolJobRunner {
    /// Creates a pool with exactly `thread_count` workers.
    pub fn new(thread_count: usize) -> Result<Self> {
        if thread_count == 0 {
            return Err(Error::InvalidArgument {
                field: "thread_count",
                reason: "must be greater than zero".into(),
            });
        }

        let inner = Arc::new(PoolInner {
            state: Mutex::new(QueueState {
                tasks: VecDeque::new(),
                shutdown: false,
            }),
            available: Condvar::new(),
        });
        let mut workers = Vec::with_capacity(thread_count);
        for index in 0..thread_count {
            let scope = crate::logging::integration::LogScope::capture();
            let worker_inner = Arc::clone(&inner);
            let worker = thread::Builder::new()
                .name(format!("audio2face-job-{index}"))
                .spawn(move || scope.in_scope(|| worker_loop(worker_inner)))
                .map_err(|error| Error::Io {
                    operation: "spawn job runner worker",
                    message: error.to_string(),
                });
            match worker {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    request_shutdown(&inner);
                    for worker in workers {
                        let _ = worker.join();
                    }
                    return Err(error);
                }
            }
        }

        Ok(Self {
            scope: crate::logging::integration::LogScope::capture(),
            inner,
            workers: Mutex::new(workers),
        })
    }

    /// Creates a pool sized for the active solver components, capped by the
    /// available hardware parallelism. At least one worker is retained so a
    /// task cannot be accepted into a pool that has no consumer.
    pub fn new_for_components(component_count: usize) -> Result<Self> {
        let hardware = thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        Self::new(component_count.max(1).min(hardware))
    }

    /// Alias matching the terminology used by creation-parameter builders.
    pub fn for_components(component_count: usize) -> Result<Self> {
        Self::new_for_components(component_count)
    }

    /// Returns the number of workers currently owned by this pool.
    pub fn worker_count(&self) -> usize {
        self.workers
            .lock()
            .map(|workers| workers.len())
            .unwrap_or(0)
    }
}

impl Default for ThreadPoolJobRunner {
    fn default() -> Self {
        let hardware = thread::available_parallelism()
            .map(|parallelism| parallelism.get())
            .unwrap_or(1);
        Self::new(hardware).expect("hardware parallelism is always non-zero")
    }
}

impl JobRunner for ThreadPoolJobRunner {
    fn enqueue(&self, task: JobRunnerTask) -> Result<()> {
        let mut state = self.inner.state.lock().map_err(|_| Error::Poisoned {
            resource: "job runner queue",
        })?;
        if state.shutdown {
            return Err(Error::InvalidState {
                operation: "enqueue job",
                state: "runner is shutting down",
            });
        }
        state.tasks.push_back(task);
        self.inner.available.notify_one();
        Ok(())
    }
}

/// Creates the standard shared host job runner used by BlendShape executors.
///
/// Returning the concrete runner keeps the standard factory free of a public
/// implementation trait object while callers may still coerce the `Arc` to
/// `Arc<dyn JobRunner>` when injecting it into creation parameters.
pub fn create_thread_pool_job_runner(thread_count: usize) -> Result<Arc<ThreadPoolJobRunner>> {
    Ok(Arc::new(ThreadPoolJobRunner::new(thread_count)?))
}

impl Drop for ThreadPoolJobRunner {
    fn drop(&mut self) {
        let _scope = self.scope.activate();
        request_shutdown(&self.inner);
        if let Ok(mut workers) = self.workers.lock() {
            while let Some(worker) = workers.pop() {
                let _ = worker.join();
            }
        }
    }
}

fn request_shutdown(inner: &Arc<PoolInner>) {
    if let Ok(mut state) = inner.state.lock() {
        state.shutdown = true;
        inner.available.notify_all();
    }
}

fn worker_loop(inner: Arc<PoolInner>) {
    loop {
        let task = {
            let mut state = match inner.state.lock() {
                Ok(state) => state,
                Err(poisoned) => poisoned.into_inner(),
            };
            while state.tasks.is_empty() && !state.shutdown {
                state = match inner.available.wait(state) {
                    Ok(state) => state,
                    Err(poisoned) => poisoned.into_inner(),
                };
            }
            if state.tasks.is_empty() && state.shutdown {
                return;
            }
            state.tasks.pop_front()
        };
        if let Some(task) = task {
            task.run();
        }
    }
}

impl ThreadPoolJobRunner {
    pub fn new_with_context(
        thread_count: usize,
        context: crate::Audio2Face3DContext,
    ) -> Result<Self> {
        crate::logging::integration::LogScope::new(context).in_scope(|| Self::new(thread_count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio2x::{Execution, ExecutionReport, ExecutionState};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    struct CounterWaker(Arc<AtomicUsize>);

    impl Wake for CounterWaker {
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
            executed_tracks: 1,
            emitted_frames: 1,
        }
    }

    #[test]
    fn task_run_is_at_most_once_and_completes() {
        let (execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        let runs = Arc::new(AtomicUsize::new(0));
        let runs_for_task = Arc::clone(&runs);
        let task = JobRunnerTask::new(completion.clone(), 0, move || {
            runs_for_task.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
        task.run();
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        completion.finish_schedule(report());
        assert_eq!(futures_poll(execution).unwrap(), report());
    }

    #[test]
    fn dropped_task_records_worker_error_and_wakes_track_waiter() {
        let (mut execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        let wake_count = Arc::new(AtomicUsize::new(0));
        let waker = Waker::from(Arc::new(CounterWaker(Arc::clone(&wake_count))));
        let mut cx = Context::from_waker(&waker);
        let mut track_wait = execution.wait_track(0);
        assert!(matches!(
            Pin::new(&mut track_wait).poll(&mut cx),
            Poll::Pending
        ));
        // The accepted task is then dropped without being run.
        drop(JobRunnerTask::new(completion.clone(), 0, || Ok(())));
        completion.finish_schedule(report());
        assert!(wake_count.load(Ordering::SeqCst) > 0);
        assert!(matches!(
            Pin::new(&mut track_wait).poll(&mut cx),
            Poll::Ready(Err(Error::Worker { .. }))
        ));
    }

    #[test]
    fn panic_is_saved_as_worker_error() {
        let (execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        let pool = ThreadPoolJobRunner::new(1).unwrap();
        pool.enqueue(JobRunnerTask::new(completion.clone(), 0, || {
            panic!("callback panic")
        }))
        .unwrap();
        completion.finish_schedule(report());
        drop(pool);
        assert!(
            matches!(futures_poll(execution), Err(Error::Worker { track: 0, message }) if message.contains("callback panic"))
        );
    }

    struct RejectingRunner;

    #[derive(Default)]
    struct HoldingRunner(Mutex<Vec<JobRunnerTask>>);

    impl JobRunner for HoldingRunner {
        fn enqueue(&self, task: JobRunnerTask) -> Result<()> {
            self.0.lock().unwrap().push(task);
            Ok(())
        }
    }

    #[test]
    fn serial_queues_preserve_track_order_across_calls_with_reverse_runner() {
        let runner = HoldingRunner::default();
        let queues = [
            Arc::new(SerialJobQueue::default()),
            Arc::new(SerialJobQueue::default()),
        ];
        let observed = Arc::new(Mutex::new([Vec::new(), Vec::new()]));
        let mut executions = Vec::new();
        for frame in 0..8 {
            let (execution, completion) = Execution::pending(2);
            for (track, queue) in queues.iter().enumerate() {
                completion.add_task(track).unwrap();
                let observed = Arc::clone(&observed);
                queue
                    .enqueue(
                        &runner,
                        JobRunnerTask::new(completion.clone(), track, move || {
                            observed.lock().unwrap()[track].push(frame);
                            Ok(())
                        }),
                    )
                    .unwrap();
            }
            completion.finish_schedule(report());
            executions.push(execution);
        }
        let mut dispatches = std::mem::take(&mut *runner.0.lock().unwrap());
        assert_eq!(dispatches.len(), 2);
        while let Some(task) = dispatches.pop() {
            task.run();
        }
        assert_eq!(
            *observed.lock().unwrap(),
            [Vec::from_iter(0..8), Vec::from_iter(0..8)]
        );
        for execution in executions {
            assert_ready(execution, false);
        }
    }

    #[test]
    fn serial_queue_cancels_dropped_and_rejected_dispatches_and_can_be_reused() {
        let queue = Arc::new(SerialJobQueue::default());
        let runner = HoldingRunner::default();
        let (execution, completion) = Execution::pending(1);
        for _ in 0..3 {
            completion.add_task(0).unwrap();
            queue
                .enqueue(
                    &runner,
                    JobRunnerTask::new(completion.clone(), 0, || Ok(())),
                )
                .unwrap();
        }
        completion.finish_schedule(report());
        drop(std::mem::take(&mut *runner.0.lock().unwrap()));
        assert_ready(execution, true);

        let (execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        assert!(
            queue
                .enqueue(
                    &RejectingRunner,
                    JobRunnerTask::new(completion.clone(), 0, || Ok(()))
                )
                .is_err()
        );
        completion.finish_schedule(report());
        assert_ready(execution, true);

        let (execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        queue
            .enqueue(
                &runner,
                JobRunnerTask::new(completion.clone(), 0, || Ok(())),
            )
            .unwrap();
        completion.finish_schedule(report());
        runner.0.lock().unwrap().pop().unwrap().run();
        assert_ready(execution, false);
    }

    #[test]
    fn serial_queue_continues_after_task_panic() {
        let queue = Arc::new(SerialJobQueue::default());
        let runner = HoldingRunner::default();
        let (failed, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        queue
            .enqueue(
                &runner,
                JobRunnerTask::new(completion.clone(), 0, || panic!("serial panic")),
            )
            .unwrap();
        completion.finish_schedule(report());
        let (succeeded, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        queue
            .enqueue(
                &runner,
                JobRunnerTask::new(completion.clone(), 0, || Ok(())),
            )
            .unwrap();
        completion.finish_schedule(report());
        runner.0.lock().unwrap().pop().unwrap().run();
        assert_ready(failed, true);
        assert_ready(succeeded, false);
    }

    #[test]
    fn serial_queue_preserves_order_on_multi_worker_pool() {
        let queue = Arc::new(SerialJobQueue::default());
        let runner = ThreadPoolJobRunner::new(4).unwrap();
        let observed = Arc::new(Mutex::new(Vec::new()));
        let (execution, completion) = Execution::pending(1);
        for frame in 0..1_000 {
            completion.add_task(0).unwrap();
            let observed = Arc::clone(&observed);
            queue
                .enqueue(
                    &runner,
                    JobRunnerTask::new(completion.clone(), 0, move || {
                        thread::yield_now();
                        observed.lock().unwrap().push(frame);
                        Ok(())
                    }),
                )
                .unwrap();
            thread::yield_now();
        }
        completion.finish_schedule(report());
        drop(runner);
        assert_ready(execution, false);
        assert_eq!(*observed.lock().unwrap(), Vec::from_iter(0..1_000));
    }

    fn assert_ready(mut execution: Execution, worker_error: bool) {
        let mut cx = Context::from_waker(Waker::noop());
        match Pin::new(&mut execution).poll(&mut cx) {
            Poll::Ready(Err(Error::Worker { .. })) if worker_error => {}
            Poll::Ready(Ok(_)) if !worker_error => {}
            result => panic!("unexpected completion: {result:?}"),
        }
    }

    impl JobRunner for RejectingRunner {
        fn enqueue(&self, task: JobRunnerTask) -> Result<()> {
            drop(task);
            Err(Error::InvalidState {
                operation: "enqueue job",
                state: "test runner rejected task",
            })
        }
    }

    #[test]
    fn rejected_runner_task_cannot_leave_execution_pending() {
        let (execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        let runner = RejectingRunner;
        let enqueue_error = runner
            .enqueue(JobRunnerTask::new(completion.clone(), 0, || Ok(())))
            .unwrap_err();
        assert!(matches!(enqueue_error, Error::InvalidState { .. }));
        completion.finish_schedule(report());
        assert!(matches!(
            futures_poll(execution),
            Err(Error::Worker { track: 0, .. })
        ));
    }

    #[test]
    fn pool_executes_tasks_and_drains_on_drop() {
        let pool = ThreadPoolJobRunner::new(2).unwrap();
        assert_eq!(pool.worker_count(), 2);
        let (execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        let called = Arc::new(AtomicUsize::new(0));
        let called_for_task = Arc::clone(&called);
        pool.enqueue(JobRunnerTask::new(completion.clone(), 0, move || {
            called_for_task.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }))
        .unwrap();
        completion.finish_schedule(report());
        drop(pool);
        assert_eq!(called.load(Ordering::SeqCst), 1);
        assert_eq!(futures_poll(execution).unwrap(), report());
    }

    #[test]
    fn dropping_execution_detaches_observation_but_task_still_runs() {
        let pool = ThreadPoolJobRunner::new(1).unwrap();
        let (execution, completion) = Execution::pending(1);
        completion.add_task(0).unwrap();
        let called = Arc::new(AtomicUsize::new(0));
        let called_for_task = Arc::clone(&called);
        pool.enqueue(JobRunnerTask::new(completion.clone(), 0, move || {
            called_for_task.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }))
        .unwrap();
        completion.finish_schedule(report());
        drop(execution);
        drop(pool);
        assert_eq!(called.load(Ordering::SeqCst), 1);
    }

    fn futures_poll(mut execution: Execution) -> Result<ExecutionReport> {
        let waker = Waker::from(Arc::new(CounterWaker(Arc::new(AtomicUsize::new(0)))));
        let mut cx = Context::from_waker(&waker);
        loop {
            match Pin::new(&mut execution).poll(&mut cx) {
                Poll::Ready(result) => return result,
                Poll::Pending => thread::yield_now(),
            }
        }
    }
}

#[cfg(test)]
mod logging_tests {
    use super::*;
    use crate::{
        Audio2Face3DContext,
        logging::{LogLevel, Logger, integration::LogScope},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Sink(AtomicUsize);
    impl Logger for Sink {
        fn log_level(&self) -> LogLevel {
            LogLevel::Trace
        }
        fn write_log(&self, _: LogLevel, _: crate::logging::LogRecord) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    struct Resource;
    impl Drop for Resource {
        fn drop(&mut self) {
            tracing::info!("resource dropped");
        }
    }
    #[test]
    fn task_run_and_unexecuted_drop_retain_original_context() {
        for run in [false, true] {
            let sink = Arc::new(Sink(AtomicUsize::new(0)));
            let context = Audio2Face3DContext::builder().logger(sink.clone()).build();
            let scope = LogScope::new(context);
            let task = scope.in_scope(|| {
                let (_, completion) = crate::audio2x::Execution::pending(1);
                completion.add_task(0).unwrap();
                let resource = Resource;
                JobRunnerTask::new(completion, 0, move || {
                    drop(resource);
                    Ok(())
                })
            });
            drop(scope);
            std::thread::spawn(move || if run { task.run() } else { drop(task) })
                .join()
                .unwrap();
            assert!(sink.0.load(Ordering::SeqCst) >= 1);
        }
    }
}
