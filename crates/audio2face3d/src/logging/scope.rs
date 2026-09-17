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
    #[cfg(feature = "tracing")]
    dispatch: tracing::Dispatch,
    #[cfg(feature = "tracing")]
    span: tracing::Span,
}
impl LogScope {
    pub fn new(context: Audio2Face3DContext) -> Self {
        #[cfg(feature = "tracing")]
        let dispatch = if context.is_legacy() {
            tracing::dispatcher::get_default(Clone::clone)
        } else {
            super::bridge::dispatch(context.clone())
        };
        Self {
            context,
            #[cfg(feature = "tracing")]
            dispatch,
            #[cfg(feature = "tracing")]
            span: tracing::Span::none(),
        }
    }
    pub fn capture() -> Self {
        let scope = CURRENT
            .with(|v| v.borrow().clone())
            .unwrap_or_else(|| Self::new(Audio2Face3DContext::legacy()));
        #[cfg(feature = "tracing")]
        {
            let mut scope = scope;
            scope.span = tracing::Span::current();
            scope
        }
        #[cfg(not(feature = "tracing"))]
        {
            scope
        }
    }
    pub fn for_current(&self) -> Self {
        let current = Self::capture();
        if current.context.shares_resources(&self.context) {
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
        #[cfg(feature = "tracing")]
        let dispatch = tracing::dispatcher::set_default(&self.dispatch);
        ScopeGuard {
            previous,
            #[cfg(feature = "tracing")]
            dispatch: Some(dispatch),
            #[cfg(feature = "tracing")]
            span: Some(self.span.clone().entered()),
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
    #[cfg(feature = "tracing")]
    dispatch: Option<tracing::dispatcher::DefaultGuard>,
    #[cfg(feature = "tracing")]
    span: Option<tracing::span::EnteredSpan>,
    not_send: PhantomData<Rc<()>>,
}
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        #[cfg(feature = "tracing")]
        drop(self.span.take());
        #[cfg(feature = "tracing")]
        drop(self.dispatch.take());
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
