//! FIFO execution admission with RAII permits and one optional deadline thread per queue.
use crate::inference::{Cancellation, cancellation::Registration};
use crate::types::{Error, ErrorKind, Result};
use std::{
    collections::VecDeque,
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex, Weak},
    task::{Context, Poll, Waker},
    thread::{self, Thread},
    time::{Duration, Instant},
};

#[derive(Debug)]
enum Outcome {
    Waiting,
    Granted,
    Failed(Error),
    Consumed,
}
struct WaitState {
    outcome: Outcome,
    waker: Option<Waker>,
}
struct Waiter {
    state: Mutex<WaitState>,
    deadline: Option<Instant>,
}
struct State {
    active: usize,
    closed: bool,
    queue: VecDeque<Arc<Waiter>>,
    grants: Vec<Weak<Waiter>>,
}
struct Inner {
    state: Mutex<State>,
    limit: usize,
    queued: usize,
    timeout: Duration,
    timer: Mutex<Option<Thread>>,
}

/// Clone shares the same execution capacity. Admission starts when acquire is called.
#[derive(Clone)]
pub struct Admission {
    inner: Arc<Inner>,
}
impl Admission {
    pub fn new(active: usize, queued: usize, timeout: Duration) -> Result<Self> {
        if active == 0 {
            return Err(Error::invalid("execution capacity must be positive"));
        }
        if !timeout.is_zero() && Instant::now().checked_add(timeout).is_none() {
            return Err(Error::invalid("queue timeout exceeds clock range"));
        }
        let inner = Arc::new(Inner {
            state: Mutex::new(State {
                active: 0,
                closed: false,
                queue: VecDeque::new(),
                grants: vec![],
            }),
            limit: active,
            queued,
            timeout,
            timer: Mutex::new(None),
        });
        if !timeout.is_zero() {
            let weak = Arc::downgrade(&inner);
            let handle = thread::Builder::new()
                .name("a2f-deadlines".into())
                .spawn(move || {
                    loop {
                        let Some(inner) = weak.upgrade() else {
                            break;
                        };
                        let (deadline, closed, wakes) = {
                            let mut state = inner.state.lock().unwrap();
                            let wakes = inner.dispatch(&mut state);
                            (
                                state.queue.iter().filter_map(|w| w.deadline).min(),
                                state.closed,
                                wakes,
                            )
                        };
                        // Drop strong ownership before parking, so the queue can be destroyed.
                        drop(inner);
                        wake(wakes);
                        if closed {
                            break;
                        }
                        match deadline {
                            Some(t) => {
                                thread::park_timeout(t.saturating_duration_since(Instant::now()))
                            }
                            None => thread::park(),
                        }
                    }
                })
                .map_err(|e| Error::new(ErrorKind::Inference, e.to_string()))?;
            *inner.timer.lock().unwrap() = Some(handle.thread().clone());
        }
        Ok(Self { inner })
    }
    pub fn acquire(&self, cancel: &Cancellation) -> Acquire {
        let waiter = Arc::new(Waiter {
            state: Mutex::new(WaitState {
                outcome: Outcome::Waiting,
                waker: None,
            }),
            deadline: if self.inner.timeout.is_zero() {
                None
            } else {
                Instant::now().checked_add(self.inner.timeout)
            },
        });
        let wakes = {
            let mut state = self.inner.state.lock().unwrap();
            let wakes = self.inner.dispatch(&mut state);
            if state.closed {
                fail(
                    &waiter,
                    Error::new(ErrorKind::ShuttingDown, "execution queue closed"),
                );
            } else if cancel.is_cancelled() {
                fail(&waiter, cancelled());
            } else if state.active < self.inner.limit {
                grant(&mut state, &waiter);
            } else if state.queue.len() >= self.inner.queued {
                fail(
                    &waiter,
                    Error::new(ErrorKind::QueueFull, "request queue full"),
                );
            } else {
                state.queue.push_back(waiter.clone());
            }
            wakes
        };
        wake(wakes);
        self.inner.notify_timer();
        let inner = Arc::downgrade(&self.inner);
        let weak_waiter = Arc::downgrade(&waiter);
        let registration = cancel.register(Arc::new(move || {
            if let (Some(inner), Some(waiter)) = (inner.upgrade(), weak_waiter.upgrade()) {
                inner.remove(&waiter, cancelled());
            }
        }));
        Acquire {
            inner: self.inner.clone(),
            waiter,
            _registration: registration,
        }
    }
    /// Reject new/queued admissions. Already delivered permits retain capacity until dropped.
    pub fn close(&self) {
        let wakes = {
            let mut state = self.inner.state.lock().unwrap();
            state.closed = true;
            let mut wakes = vec![];
            for waiter in state.queue.drain(..) {
                if let Some(w) = fail(
                    &waiter,
                    Error::new(ErrorKind::ShuttingDown, "execution queue closed"),
                ) {
                    wakes.push(w);
                }
            }
            let grants = std::mem::take(&mut state.grants);
            for waiter in grants.into_iter().filter_map(|w| w.upgrade()) {
                if matches!(waiter.state.lock().unwrap().outcome, Outcome::Granted) {
                    state.active -= 1;
                    if let Some(w) = fail(
                        &waiter,
                        Error::new(ErrorKind::ShuttingDown, "execution queue closed"),
                    ) {
                        wakes.push(w);
                    }
                }
            }
            wakes
        };
        wake(wakes);
        self.inner.notify_timer();
    }
    pub fn available_permits(&self) -> usize {
        self.inner.limit - self.inner.state.lock().unwrap().active
    }
    pub fn waiting_count(&self) -> usize {
        self.inner.state.lock().unwrap().queue.len()
    }
}
fn cancelled() -> Error {
    Error::new(ErrorKind::Cancelled, "execution queue wait cancelled")
}
fn wake(wakers: Vec<Waker>) {
    for waker in wakers {
        waker.wake();
    }
}
fn fail(waiter: &Waiter, error: Error) -> Option<Waker> {
    let mut state = waiter.state.lock().unwrap();
    state.outcome = Outcome::Failed(error);
    state.waker.take()
}
fn grant(state: &mut State, waiter: &Arc<Waiter>) -> Option<Waker> {
    state.grants.retain(|w| {
        w.upgrade()
            .is_some_and(|w| matches!(w.state.lock().unwrap().outcome, Outcome::Granted))
    });
    state.active += 1;
    state.grants.push(Arc::downgrade(waiter));
    let mut result = waiter.state.lock().unwrap();
    result.outcome = Outcome::Granted;
    result.waker.take()
}
impl Inner {
    fn notify_timer(&self) {
        if let Some(timer) = &*self.timer.lock().unwrap() {
            timer.unpark();
        }
    }
    /// Called with state locked. Expiry precedes grant, even if the timer has not run yet.
    fn dispatch(&self, state: &mut State) -> Vec<Waker> {
        let now = Instant::now();
        let mut wakes = vec![];
        state.queue.retain(|waiter| {
            if waiter.deadline.is_some_and(|t| t <= now) {
                if let Some(w) = fail(
                    waiter,
                    Error::new(ErrorKind::DeadlineExceeded, "request queue wait timed out"),
                ) {
                    wakes.push(w);
                }
                false
            } else {
                true
            }
        });
        while !state.closed && state.active < self.limit {
            let Some(waiter) = state.queue.pop_front() else {
                break;
            };
            if let Some(w) = grant(state, &waiter) {
                wakes.push(w);
            }
        }
        wakes
    }
    fn remove(&self, waiter: &Arc<Waiter>, error: Error) {
        let wakes = {
            let mut state = self.state.lock().unwrap();
            let mut result = waiter.state.lock().unwrap();
            match result.outcome {
                Outcome::Waiting => state.queue.retain(|w| !Arc::ptr_eq(w, waiter)),
                Outcome::Granted => state.active -= 1,
                Outcome::Consumed | Outcome::Failed(_) => return,
            }
            result.outcome = Outcome::Failed(error);
            let mut wakes: Vec<_> = result.waker.take().into_iter().collect();
            drop(result);
            wakes.extend(self.dispatch(&mut state));
            wakes
        };
        wake(wakes);
        self.notify_timer();
    }
}
impl Drop for Inner {
    fn drop(&mut self) {
        self.notify_timer();
    }
}
/// Dropping an unconsumed acquire removes it, including an already reserved slot.
pub struct Acquire {
    inner: Arc<Inner>,
    waiter: Arc<Waiter>,
    _registration: Registration,
}
impl Future for Acquire {
    type Output = Result<Permit>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let _state = self.inner.state.lock().unwrap();
        let mut state = self.waiter.state.lock().unwrap();
        match std::mem::replace(&mut state.outcome, Outcome::Consumed) {
            Outcome::Waiting => {
                state.outcome = Outcome::Waiting;
                state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
            Outcome::Granted => {
                state.waker = None;
                Poll::Ready(Ok(Permit {
                    inner: self.inner.clone(),
                }))
            }
            Outcome::Failed(error) => {
                state.waker = None;
                Poll::Ready(Err(error))
            }
            Outcome::Consumed => panic!("acquire polled after completion"),
        }
    }
}
impl Drop for Acquire {
    fn drop(&mut self) {
        self.inner.remove(&self.waiter, cancelled());
    }
}
/// Release only after native cleanup AND application output ownership have ended.
pub struct Permit {
    inner: Arc<Inner>,
}
impl std::fmt::Debug for Permit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Permit").finish_non_exhaustive()
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        let wakes = {
            let mut state = self.inner.state.lock().unwrap();
            state.active -= 1;
            self.inner.dispatch(&mut state)
        };
        wake(wakes);
        self.inner.notify_timer();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::worker::wait;
    fn pending(future: &mut Acquire) {
        assert!(
            Pin::new(future)
                .poll(&mut Context::from_waker(Waker::noop()))
                .is_pending()
        );
    }
    #[test]
    fn fifo_uses_any_free_slot_for_one_or_multiple_engines() {
        for n in [1, 2, 4] {
            let queue = Admission::new(n, 3, Duration::ZERO).unwrap();
            let cancel = Cancellation::new();
            let mut running = (0..n)
                .map(|_| wait(queue.acquire(&cancel)).unwrap())
                .collect::<Vec<_>>();
            let first = queue.acquire(&cancel);
            let mut second = queue.acquire(&cancel);
            let mut third = queue.acquire(&cancel);
            assert_eq!(
                wait(queue.acquire(&cancel)).unwrap_err().kind(),
                ErrorKind::QueueFull
            );
            drop(running.pop()); // FIFO reservation is made even if the oldest Future is not polled.
            pending(&mut second);
            pending(&mut third);
            let mut later = queue.acquire(&cancel);
            pending(&mut later);
            let one = wait(first).unwrap();
            pending(&mut second);
            drop(one);
            let two = wait(second).unwrap();
            pending(&mut third);
            drop(two);
            drop(wait(third).unwrap());
            drop(wait(later).unwrap());
            drop(running);
            assert_eq!(queue.available_permits(), n);
            assert_eq!(queue.waiting_count(), 0);
        }
    }
    #[test]
    fn cancelling_without_poll_and_dropping_granted_future_return_capacity() {
        let queue = Admission::new(2, 2, Duration::ZERO).unwrap();
        let cancel = Cancellation::new();
        let a = wait(queue.acquire(&cancel)).unwrap();
        let b = wait(queue.acquire(&cancel)).unwrap();
        let abandoned = Cancellation::new();
        let first = queue.acquire(&abandoned);
        let second = queue.acquire(&cancel);
        abandoned.cancel();
        assert_eq!(queue.waiting_count(), 1);
        assert_eq!(wait(first).unwrap_err().kind(), ErrorKind::Cancelled);
        let third = queue.acquire(&cancel);
        drop(a); // second is granted but not consumed
        drop(second); // third is granted immediately; b still occupies its slot
        drop(wait(third).unwrap());
        drop(b);
        assert_eq!(queue.available_permits(), 2);
        let granted = queue.acquire(&abandoned);
        assert_eq!(wait(granted).unwrap_err().kind(), ErrorKind::Cancelled);
        let late_cancel = Cancellation::new();
        let granted = queue.acquire(&late_cancel);
        late_cancel.cancel();
        assert_eq!(queue.available_permits(), 2);
        assert_eq!(wait(granted).unwrap_err().kind(), ErrorKind::Cancelled);
    }
    #[test]
    fn deadline_frees_queue_without_polling_and_shutdown_keeps_active_permits() {
        let queue = Admission::new(1, 1, Duration::from_millis(20)).unwrap();
        let cancel = Cancellation::new();
        let active = wait(queue.acquire(&cancel)).unwrap();
        let expired = queue.acquire(&cancel);
        let deadline = Instant::now() + Duration::from_secs(5);
        while queue.waiting_count() != 0 {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(
            wait(expired).unwrap_err().kind(),
            ErrorKind::DeadlineExceeded
        );
        let stopped = queue.acquire(&cancel);
        queue.close();
        assert_eq!(wait(stopped).unwrap_err().kind(), ErrorKind::ShuttingDown);
        assert_eq!(queue.available_permits(), 0);
        drop(active);
        assert_eq!(queue.available_permits(), 1);
        assert_eq!(
            wait(queue.acquire(&cancel)).unwrap_err().kind(),
            ErrorKind::ShuttingDown
        );
    }
    #[test]
    fn cancel_grant_race_and_wakers_can_reenter_queue() {
        struct Reenter(Admission);
        impl std::task::Wake for Reenter {
            fn wake(self: Arc<Self>) {
                let _ = self.0.available_permits();
            }
        }
        for _ in 0..100 {
            let queue = Admission::new(1, 1, Duration::ZERO).unwrap();
            let cancel = Cancellation::new();
            let active = wait(queue.acquire(&Cancellation::new())).unwrap();
            let mut waiting = queue.acquire(&cancel);
            let waker = Waker::from(Arc::new(Reenter(queue.clone())));
            assert!(
                Pin::new(&mut waiting)
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending()
            );
            thread::scope(|s| {
                s.spawn(|| cancel.cancel());
                s.spawn(|| drop(active));
            });
            assert_eq!(wait(waiting).unwrap_err().kind(), ErrorKind::Cancelled);
            assert_eq!(queue.available_permits(), 1);
            assert_eq!(queue.waiting_count(), 0);
        }
    }
    #[test]
    fn closing_revokes_unconsumed_grants_and_zero_wait_capacity_is_valid() {
        let queue = Admission::new(2, 0, Duration::ZERO).unwrap();
        let cancel = Cancellation::new();
        let a = queue.acquire(&cancel);
        let b = queue.acquire(&cancel);
        assert_eq!(
            wait(queue.acquire(&cancel)).unwrap_err().kind(),
            ErrorKind::QueueFull
        );
        queue.close();
        assert_eq!(wait(a).unwrap_err().kind(), ErrorKind::ShuttingDown);
        assert_eq!(wait(b).unwrap_err().kind(), ErrorKind::ShuttingDown);
        assert_eq!(queue.available_permits(), 2);
    }
}
