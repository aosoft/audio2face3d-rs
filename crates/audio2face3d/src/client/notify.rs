use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    task::Waker,
};

#[derive(Clone, Default)]
pub(crate) struct Notify(Arc<Mutex<State>>);
#[derive(Default)]
struct State {
    next: u64,
    wakers: BTreeMap<u64, Waker>,
}
pub(crate) struct Subscription {
    notify: Notify,
    id: u64,
}
impl Notify {
    pub fn subscribe(&self) -> Subscription {
        let mut state = self.0.lock().unwrap();
        let id = state.next;
        state.next = state
            .next
            .checked_add(1)
            .expect("notification identifier exhausted");
        Subscription {
            notify: self.clone(),
            id,
        }
    }
    pub fn wake(&self) {
        let wakers = std::mem::take(&mut self.0.lock().unwrap().wakers);
        for waker in wakers.into_values() {
            waker.wake();
        }
    }
}
impl Subscription {
    pub fn register(&self, waker: &Waker) {
        // RawWaker clone/drop callbacks may reenter: run them outside our mutex.
        let waker = waker.clone();
        let previous = self.notify.0.lock().unwrap().wakers.insert(self.id, waker);
        drop(previous);
    }
}
impl Drop for Subscription {
    fn drop(&mut self) {
        let previous = self.notify.0.lock().unwrap().wakers.remove(&self.id);
        drop(previous);
    }
}
