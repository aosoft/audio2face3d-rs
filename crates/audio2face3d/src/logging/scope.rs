use crate::Audio2Face3DContext;
use std::{
    cell::RefCell,
    future::Future,
    marker::PhantomData,
    mem::ManuallyDrop,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll},
};
thread_local! {static CURRENT:RefCell<Option<LogScope>>=const{RefCell::new(None)};}
#[derive(Clone)]
pub struct LogScope {
    context: Audio2Face3DContext,
    fields: Vec<(String, super::LogValue)>,
}
impl LogScope {
    pub fn new(context: Audio2Face3DContext) -> Self {
        Self {
            context,
            fields: Vec::new(),
        }
    }
    pub fn capture() -> Self {
        CURRENT
            .with(|v| v.borrow().clone())
            .unwrap_or_else(|| Self::new(Audio2Face3DContext::default()))
    }
    /// Adds request-local context without modifying shared application resources.
    pub fn field(mut self, key: impl Into<String>, value: impl Into<super::LogValue>) -> Self {
        let record = super::LogRecord {
            message: String::new(),
            fields: self.fields,
        }
        .field(key, value);
        self.fields = record.fields;
        self
    }
    pub fn log(&self, level: super::LogLevel, make_record: impl FnOnce() -> super::LogRecord) {
        use super::Logger;
        self.context.logger().log(level, || {
            let mut record = make_record();
            for (key, value) in &self.fields {
                if !record.fields.iter().any(|(k, _)| k == key) {
                    record.fields.push((key.clone(), value.clone()));
                }
            }
            record
        });
    }
    pub fn for_current(&self) -> Self {
        let current = Self::capture();
        if self.fields.is_empty() && current.context.shares_resources(&self.context) {
            current
        } else {
            self.clone()
        }
    }
    pub fn activate(&self) -> ScopeGuard {
        self.for_current().enter()
    }
    pub fn context(&self) -> &Audio2Face3DContext {
        &self.context
    }
    pub fn enter(&self) -> ScopeGuard {
        let previous = CURRENT.with(|v| v.replace(Some(self.clone())));
        ScopeGuard {
            previous,
            not_send: PhantomData,
        }
    }
    pub fn in_scope<T>(&self, f: impl FnOnce() -> T) -> T {
        let _guard = self.enter();
        f()
    }
    pub fn wrap_future<F: Future>(&self, future: F) -> Scoped<F> {
        Scoped {
            future: ManuallyDrop::new(future),
            scope: self.clone(),
        }
    }
}
pub struct ScopeGuard {
    previous: Option<LogScope>,
    not_send: PhantomData<Rc<()>>,
}
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        CURRENT.with(|v| {
            v.replace(self.previous.take());
        });
    }
}
/// Restores the owning scope for each poll and for cancellation/drop.
pub struct Scoped<F> {
    future: ManuallyDrop<F>,
    scope: LogScope,
}
impl<F: Future> Future for Scoped<F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: future is never moved after pinning; Drop destroys it in place.
        let this = unsafe { self.get_unchecked_mut() };
        let _guard = this.scope.enter();
        // SAFETY: projecting only the structurally pinned future; no move or replacement.
        unsafe { Pin::new_unchecked(&mut *this.future) }.poll(cx)
    }
}
impl<F> Drop for Scoped<F> {
    fn drop(&mut self) {
        let _guard = self.scope.enter();
        // SAFETY: future is manually dropped exactly once, in place, under its scope.
        unsafe {
            ManuallyDrop::drop(&mut self.future);
        }
    }
}

impl std::fmt::Debug for LogScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogScope").finish_non_exhaustive()
    }
}
