//! Serialized initialization without holding the registry lock across foreign code.
use super::{NativeRuntimeError, NativeRuntimeErrorKind};
use std::{
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{Arc, Condvar, Mutex},
    thread::{self, ThreadId},
};

pub(crate) struct LoadAttempt {
    started: bool,
}
impl LoadAttempt {
    pub(crate) fn begin(&mut self) {
        self.started = true;
    }
}
enum State<K, T> {
    Empty,
    Loading(ThreadId),
    Ready(K, Arc<T>),
    Failed(NativeRuntimeError),
}
pub(crate) struct Registry<K, T> {
    state: Mutex<State<K, T>>,
    changed: Condvar,
}
impl<K: PartialEq, T> Registry<K, T> {
    pub(crate) fn prior_failure(&self) -> Option<NativeRuntimeError> {
        match &*self.state.lock().unwrap_or_else(|e| e.into_inner()) {
            State::Failed(error) => Some(error.clone()),
            _ => None,
        }
    }
    pub(crate) const fn new() -> Self {
        Self {
            state: Mutex::new(State::Empty),
            changed: Condvar::new(),
        }
    }
    pub(crate) fn initialize(
        &self,
        key: K,
        initialize: impl FnOnce(&mut LoadAttempt) -> Result<T, NativeRuntimeError>,
    ) -> Result<Arc<T>, NativeRuntimeError> {
        let owner = thread::current().id();
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match &*state {
                State::Empty => {
                    *state = State::Loading(owner);
                    break;
                }
                State::Loading(id) if *id == owner => {
                    return Err(NativeRuntimeError::new(
                        NativeRuntimeErrorKind::InitializationReentered,
                        "native initialization reentered on the initializing thread",
                    ));
                }
                State::Loading(_) => {
                    state = self.changed.wait(state).unwrap_or_else(|e| e.into_inner())
                }
                State::Ready(existing, value) if *existing == key => return Ok(Arc::clone(value)),
                State::Ready(_, _) => {
                    return Err(NativeRuntimeError::new(
                        NativeRuntimeErrorKind::RuntimeConflict,
                        "a different native configuration is already loaded",
                    ));
                }
                State::Failed(error) => return Err(error.clone()),
            }
        }
        drop(state);
        let mut attempt = LoadAttempt { started: false };
        let result = catch_unwind(AssertUnwindSafe(|| initialize(&mut attempt)));
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let output = match result {
            Ok(Ok(value)) => {
                let value = Arc::new(value);
                *state = State::Ready(key, Arc::clone(&value));
                Ok(Ok(value))
            }
            Ok(Err(error)) => {
                let error = if attempt.started {
                    error.after_load()
                } else {
                    error
                };
                *state = if attempt.started {
                    State::Failed(error.clone())
                } else {
                    State::Empty
                };
                Ok(Err(error))
            }
            Err(payload) => {
                *state = if attempt.started {
                    State::Failed(
                        NativeRuntimeError::new(
                            NativeRuntimeErrorKind::InitializationFailed,
                            "native initialization panicked after loading began",
                        )
                        .after_load(),
                    )
                } else {
                    State::Empty
                };
                Err(payload)
            }
        };
        self.changed.notify_all();
        drop(state);
        match output {
            Ok(result) => result,
            Err(payload) => resume_unwind(payload),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    fn failure() -> NativeRuntimeError {
        NativeRuntimeError::new(NativeRuntimeErrorKind::SymbolMissing, "fixture")
    }
    #[test]
    fn concurrent_callers_share_one_initialization_and_conflicts_are_rejected() {
        let registry = Registry::<u8, u32>::new();
        let barrier = Barrier::new(8);
        let calls = AtomicUsize::new(0);
        thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    assert_eq!(
                        *registry
                            .initialize(1, |_| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                Ok(42)
                            })
                            .unwrap(),
                        42
                    );
                });
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            registry.initialize(2, |_| Ok(0)).unwrap_err().kind(),
            NativeRuntimeErrorKind::RuntimeConflict
        );
    }
    #[test]
    fn reentry_is_reported_without_deadlock() {
        let registry = Registry::<u8, u32>::new();
        registry
            .initialize(1, |_| {
                assert_eq!(
                    registry.initialize(1, |_| Ok(0)).unwrap_err().kind(),
                    NativeRuntimeErrorKind::InitializationReentered
                );
                Ok(42)
            })
            .unwrap();
    }
    #[test]
    fn preload_failure_can_retry_but_postload_failure_cannot() {
        let registry = Registry::<u8, u32>::new();
        assert!(
            !registry
                .initialize(1, |_| Err(failure()))
                .unwrap_err()
                .restart_required()
        );
        assert!(
            registry
                .initialize(2, |attempt| {
                    attempt.begin();
                    Err(failure())
                })
                .unwrap_err()
                .restart_required()
        );
        assert!(
            registry
                .initialize(2, |_| panic!("must not retry"))
                .unwrap_err()
                .restart_required()
        );
    }
    #[test]
    fn panics_release_waiters_and_preserve_failed_after_load_state() {
        let registry = Registry::<u8, u32>::new();
        assert!(
            catch_unwind(AssertUnwindSafe(
                || registry.initialize(1, |_| panic!("before load"))
            ))
            .is_err()
        );
        assert!(
            catch_unwind(AssertUnwindSafe(|| registry.initialize(1, |attempt| {
                attempt.begin();
                panic!("after load")
            })))
            .is_err()
        );
        assert_eq!(
            registry.initialize(1, |_| Ok(0)).unwrap_err().kind(),
            NativeRuntimeErrorKind::InitializationFailed
        );
    }
}

#[cfg(test)]
mod callback_tests {
    use super::*;
    use crate::{
        Audio2Face3DContext,
        logging::{LogLevel, Logger},
        runtime::{NativeVersion, version::verify},
    };
    struct Reentrant(Arc<Registry<u8, ()>>);
    impl Logger for Reentrant {
        fn log_level(&self) -> LogLevel {
            LogLevel::Warn
        }
        fn write_log(&self, _: LogLevel, _: crate::logging::LogRecord) {
            assert_eq!(
                self.0.initialize(1, |_| Ok(())).unwrap_err().kind(),
                NativeRuntimeErrorKind::InitializationReentered
            );
        }
    }
    #[test]
    fn logger_callback_can_reenter_without_holding_registry_lock() {
        let registry = Arc::new(Registry::new());
        let context = Audio2Face3DContext::builder()
            .logger(Arc::new(Reentrant(registry.clone())))
            .build();
        registry
            .initialize(1, |_| {
                verify(
                    &context,
                    "fixture",
                    NativeVersion::new(10, 16, None, None),
                    NativeVersion::new(10, 17, None, None),
                )
            })
            .unwrap();
    }
    #[test]
    fn concurrent_postload_failure_releases_every_waiter_without_retry() {
        use std::sync::{
            Barrier,
            atomic::{AtomicUsize, Ordering},
        };
        let registry = Registry::<u8, ()>::new();
        let barrier = Barrier::new(8);
        let calls = AtomicUsize::new(0);
        thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    barrier.wait();
                    let error = registry
                        .initialize(1, |attempt| {
                            attempt.begin();
                            calls.fetch_add(1, Ordering::SeqCst);
                            Err(NativeRuntimeError::new(
                                NativeRuntimeErrorKind::InitializationFailed,
                                "fixture initialization failure",
                            ))
                        })
                        .unwrap_err();
                    assert!(error.restart_required());
                    assert_eq!(error.kind(), NativeRuntimeErrorKind::InitializationFailed);
                });
            }
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
