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
    action: Option<JobAction>,
    completion: Arc<ExecutionCompletion>,
    track: usize,
}

impl JobRunnerTask {
    /// Runs this task at most once, consuming it.
    pub fn run(mut self) {
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
        self.completion.complete_task(self.track, result);
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
            action: Some(Box::new(action)),
            completion,
            track,
        }
    }
}

impl Drop for JobRunnerTask {
    fn drop(&mut self) {
        if self.action.take().is_some() {
            self.completion.complete_task(
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
            let worker_inner = Arc::clone(&inner);
            let worker = thread::Builder::new()
                .name(format!("audio2face-job-{index}"))
                .spawn(move || worker_loop(worker_inner))
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

impl Drop for ThreadPoolJobRunner {
    fn drop(&mut self) {
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
