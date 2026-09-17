use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex, Weak},
};

type Listener = Arc<dyn Fn() + Send + Sync>;
#[derive(Default)]
struct State {
    cancelled: bool,
    next: u64,
    listeners: BTreeMap<u64, Listener>,
}
/// Cooperative cancellation. Native work already started must still be drained.
#[derive(Clone, Default)]
pub struct Cancellation {
    state: Arc<Mutex<State>>,
}
impl Cancellation {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn is_cancelled(&self) -> bool {
        self.state.lock().unwrap().cancelled
    }
    pub fn cancel(&self) {
        let listeners = {
            let mut state = self.state.lock().unwrap();
            state.cancelled = true;
            std::mem::take(&mut state.listeners)
        };
        // Never invoke user/executor callbacks with a lock held.
        for listener in listeners.into_values() {
            listener();
        }
    }
    pub(crate) fn register(&self, listener: Listener) -> Registration {
        let mut state = self.state.lock().unwrap();
        let id = state.next;
        state.next += 1;
        if state.cancelled {
            drop(state);
            listener();
        } else {
            state.listeners.insert(id, listener);
        }
        Registration {
            state: Arc::downgrade(&self.state),
            id,
        }
    }
}
pub(crate) struct Registration {
    state: Weak<Mutex<State>>,
    id: u64,
}
impl Drop for Registration {
    fn drop(&mut self) {
        if let Some(state) = self.state.upgrade() {
            state.lock().unwrap().listeners.remove(&self.id);
        }
    }
}
