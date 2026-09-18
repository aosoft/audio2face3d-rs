use crate::client::{Limits, notify::Notify, types::*};
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex, OnceLock, Weak},
    thread::{self, Thread},
    time::Instant,
};

pub(crate) struct Core {
    pub scope: crate::logging::integration::LogScope,
    pub limits: Limits,
    pub state: Mutex<Global>,
    pub notify: Notify,
    timer: OnceLock<Thread>,
}
pub(crate) struct Global {
    pub closing: bool,
    pub driver_closed: Option<Result<()>>,
    pub requests: BTreeMap<u64, Weak<Request>>,
    next: u64,
    pub bytes: usize,
}
impl Core {
    pub fn new(limits: Limits) -> Result<Arc<Self>> {
        let core = Arc::new(Self {
            scope: crate::logging::integration::LogScope::capture(),
            limits,
            state: Mutex::new(Global {
                closing: false,
                driver_closed: None,
                requests: BTreeMap::new(),
                next: 1,
                bytes: 0,
            }),
            notify: Notify::default(),
            timer: OnceLock::new(),
        });
        let scope = core.scope.clone();
        let weak = Arc::downgrade(&core);
        let worker = thread::Builder::new()
            .name("a2f-client-deadlines".into())
            .spawn(move || {
                let _scope = scope.enter();
                loop {
                    let Some(core) = weak.upgrade() else {
                        break;
                    };
                    let requests = core.requests();
                    let mut next = None;
                    for request in requests {
                        if let Some(deadline) = request.deadline {
                            if deadline <= Instant::now() {
                                request.fail(Error::new(
                                    ErrorKind::DeadlineExceeded,
                                    "request deadline exceeded",
                                ));
                            } else if !request.has_failed() {
                                next =
                                    Some(next.map_or(deadline, |old: Instant| old.min(deadline)));
                            }
                        }
                    }
                    let done = {
                        let state = core.state.lock().unwrap();
                        state.closing && state.requests.is_empty()
                    };
                    drop(core);
                    if done {
                        break;
                    }
                    match next {
                        Some(t) => {
                            thread::park_timeout(t.saturating_duration_since(Instant::now()))
                        }
                        None => thread::park(),
                    }
                }
            })
            .map_err(|e| Error::new(ErrorKind::RuntimeUnavailable, e.to_string()))?;
        core.timer.set(worker.thread().clone()).expect("new timer");
        Ok(core)
    }
    fn requests(&self) -> Vec<Arc<Request>> {
        self.state
            .lock()
            .unwrap()
            .requests
            .values()
            .filter_map(Weak::upgrade)
            .collect()
    }
    pub fn poke(&self) {
        if let Some(thread) = self.timer.get() {
            thread.unpark();
        }
    }
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Result<Lease> {
        let mut state = self.state.lock().unwrap();
        if bytes > self.limits.max_buffered_bytes.saturating_sub(state.bytes) {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "client storage budget exhausted",
            ));
        }
        state.bytes += bytes;
        Ok(Lease {
            core: self.clone(),
            bytes,
        })
    }
    pub fn begin_shutdown(&self) -> bool {
        {
            let mut state = self.state.lock().unwrap();
            if state.closing {
                return false;
            }
            state.closing = true;
        }
        for request in self.requests() {
            request.fail(Error::new(ErrorKind::ShuttingDown, "client shutting down"));
        }
        self.poke();
        self.notify.wake();
        true
    }
    pub fn driver_closed(&self, result: Result<()>) {
        let result = result.map_err(compact_error);
        self.state
            .lock()
            .unwrap()
            .driver_closed
            .get_or_insert(result);
        self.notify.wake();
        self.poke();
    }
}
impl Drop for Core {
    fn drop(&mut self) {
        self.poke();
    }
}
pub(crate) struct Lease {
    core: Arc<Core>,
    pub bytes: usize,
}
impl Drop for Lease {
    fn drop(&mut self) {
        self.core.state.lock().unwrap().bytes -= self.bytes;
    }
}
pub(crate) struct Buffered<T> {
    pub value: T,
    pub lease: Lease,
}
impl<T> Buffered<T> {
    pub fn bytes(&self) -> usize {
        self.lease.bytes
    }
}
pub(crate) enum Outcome {
    Running,
    Success,
    Failed(Error),
}
pub(crate) struct State {
    pub input: VecDeque<Buffered<InputChunk>>,
    pub pending_input: Option<Buffered<InputChunk>>,
    pub input_bytes: usize,
    pub output: VecDeque<Buffered<OutputEvent>>,
    pub pending_output: Option<Buffered<OutputEvent>>,
    pub output_bytes: usize,
    pub input_finished: bool,
    pub input_eof: bool,
    pub outcome: Outcome,
    pub closed: Option<Result<()>>,
    pub terminal_seen: bool,
    retired: bool,
    pub progress: Progress,
    pub processing_finished: bool,
}
impl State {
    pub fn error(&self) -> Option<Error> {
        match &self.outcome {
            Outcome::Failed(e) => Some(e.clone()),
            _ => None,
        }
    }
}
pub(crate) struct Request {
    pub id: RequestId,
    pub core: Arc<Core>,
    pub format: AudioFormat,
    pub deadline: Option<Instant>,
    pub state: Mutex<State>,
    pub notify: Notify,
}
impl Request {
    pub fn register(
        core: Arc<Core>,
        format: AudioFormat,
        deadline: Option<Instant>,
        bytes: usize,
    ) -> Result<(Arc<Self>, Lease)> {
        if bytes > core.limits.max_request_bytes {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "request settings too large",
            ));
        }
        let mut global = core.state.lock().unwrap();
        if global.closing {
            return Err(Error::new(ErrorKind::ShuttingDown, "client shutting down"));
        }
        if global.requests.len() >= core.limits.max_requests {
            return Err(Error::new(
                ErrorKind::QueueFull,
                "local request limit reached",
            ));
        }
        if bytes > core.limits.max_buffered_bytes.saturating_sub(global.bytes) {
            return Err(Error::new(
                ErrorKind::LimitExceeded,
                "client storage budget exhausted",
            ));
        }
        let id = RequestId(global.next);
        global.next = global
            .next
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded, "request identifiers exhausted"))?;
        let request = Arc::new(Self {
            id,
            core: core.clone(),
            format,
            deadline,
            notify: Notify::default(),
            state: Mutex::new(State {
                input: VecDeque::new(),
                pending_input: None,
                input_bytes: 0,
                output: VecDeque::new(),
                pending_output: None,
                output_bytes: 0,
                input_finished: false,
                input_eof: false,
                outcome: Outcome::Running,
                closed: None,
                terminal_seen: false,
                retired: false,
                progress: Progress::default(),
                processing_finished: false,
            }),
        });
        global.bytes += bytes;
        global.requests.insert(id.0, Arc::downgrade(&request));
        drop(global);
        core.poke();
        Ok((request, Lease { core, bytes }))
    }
    pub fn has_failed(&self) -> bool {
        self.state.lock().unwrap().error().is_some()
    }
    pub fn fail(&self, error: Error) {
        let error = compact_error(error);
        let discarded = {
            let mut state = self.state.lock().unwrap();
            if state.error().is_some() || state.terminal_seen {
                return;
            }
            state.outcome =
                Outcome::Failed(error.with_request(self.id).with_progress(state.progress));
            state.input_bytes = 0;
            state.output_bytes = 0;
            (
                std::mem::take(&mut state.input),
                state.pending_input.take(),
                std::mem::take(&mut state.output),
                state.pending_output.take(),
            )
        };
        drop(discarded);
        self.changed();
        self.core.poke();
    }
    pub fn closed(&self, result: Result<()>) {
        if let Err(error) = result {
            self.fail(error);
        } else {
            let complete = {
                let state = self.state.lock().unwrap();
                state.input_finished && state.input_eof && state.pending_output.is_none()
            };
            if !complete {
                self.fail(Error::new(
                    ErrorKind::IncompleteResponse,
                    "backend ended before input/output handoff completed",
                ));
            }
        }
        {
            let mut state = self.state.lock().unwrap();
            if state.closed.is_some() {
                return;
            }
            state.closed = Some(state.error().map_or(Ok(()), Err));
            if state.error().is_none() {
                state.outcome = Outcome::Success;
            }
        }
        self.changed();
    }
    pub fn changed(&self) {
        let retire = {
            let mut state = self.state.lock().unwrap();
            let ready = state.closed.is_some() && (state.error().is_some() || state.terminal_seen);
            if ready && !state.retired {
                state.retired = true;
                true
            } else {
                false
            }
        };
        if retire {
            self.core.state.lock().unwrap().requests.remove(&self.id.0);
            self.core.notify.wake();
            self.core.poke();
        }
        self.notify.wake();
    }
    pub fn context(&self, error: Error) -> Error {
        error
            .with_request(self.id)
            .with_progress(self.state.lock().unwrap().progress)
    }
}

/// Terminal notifications bypass the data queue, but their text remains bounded.
fn compact_error(error: Error) -> Error {
    let mut end = error.message().len().min(4096);
    while !error.message().is_char_boundary(end) {
        end -= 1;
    }
    Error::new(error.kind(), &error.message()[..end])
}
