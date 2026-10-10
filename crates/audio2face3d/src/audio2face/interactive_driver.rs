#![cfg_attr(not(feature = "tensorrt"), allow(dead_code))]

use crate::Result;
use crate::audio2face::GeometryResults;
use crate::audio2x::{
    InteractiveExecutionReport, InteractiveExecutionStatus, InteractiveInterruptHandle,
};
use std::future::Future;
use std::ops::ControlFlow;
use std::pin::Pin;
use std::task::{Context, Poll};

pub(crate) trait InteractiveGeometryDriver {
    fn interrupt_handle(&self) -> InteractiveInterruptHandle;
    fn total_frames(&self) -> Result<usize>;
    fn prepare_all(&mut self) -> Result<()>;
    fn compute(
        &mut self,
        frame: usize,
        callback: &mut (dyn for<'r> FnMut(GeometryResults<'r>) -> ControlFlow<()> + Send),
        generation: u64,
        stateful: bool,
    ) -> Result<InteractiveExecutionReport>;
}

fn report(status: InteractiveExecutionStatus, emitted_frames: usize) -> InteractiveExecutionReport {
    InteractiveExecutionReport {
        status,
        emitted_frames,
    }
}

pub(crate) async fn compute_frame<E: InteractiveGeometryDriver>(
    executor: &mut E,
    frame: usize,
    callback: &mut (dyn for<'r> FnMut(GeometryResults<'r>) -> ControlFlow<()> + Send),
) -> Result<InteractiveExecutionReport> {
    let interrupt = executor.interrupt_handle();
    let generation = interrupt.generation();
    YieldOnce(false).await;
    if interrupt.is_interrupted_since(generation) {
        return Ok(report(InteractiveExecutionStatus::Interrupted, 0));
    }
    executor.compute(frame, callback, generation, false)
}

pub(crate) async fn compute_all_frames<E: InteractiveGeometryDriver>(
    executor: &mut E,
    callback: &mut (dyn for<'r> FnMut(GeometryResults<'r>) -> ControlFlow<()> + Send),
) -> Result<InteractiveExecutionReport> {
    let interrupt = executor.interrupt_handle();
    let generation = interrupt.generation();
    YieldOnce(false).await;
    // Preserve the existing order: validate the total before observing a stop,
    // and prepare state only after that stop check.
    let total = executor.total_frames()?;
    if interrupt.is_interrupted_since(generation) {
        return Ok(report(InteractiveExecutionStatus::Interrupted, 0));
    }
    executor.prepare_all()?;
    let mut emitted_frames = 0;
    for frame in 0..total {
        if interrupt.is_interrupted_since(generation) {
            return Ok(report(
                InteractiveExecutionStatus::Interrupted,
                emitted_frames,
            ));
        }
        let step = executor.compute(frame, callback, generation, true)?;
        emitted_frames += step.emitted_frames;
        if step.status == InteractiveExecutionStatus::Interrupted {
            return Ok(report(
                InteractiveExecutionStatus::Interrupted,
                emitted_frames,
            ));
        }
        YieldOnce(false).await;
    }
    Ok(report(InteractiveExecutionStatus::Complete, emitted_frames))
}

struct YieldOnce(bool);

impl Future for YieldOnce {
    type Output = ();
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        interrupt: InteractiveInterruptHandle,
        trace: Vec<(usize, bool)>,
        prepared: bool,
        stop: bool,
    }

    impl Default for Fake {
        fn default() -> Self {
            Self {
                interrupt: InteractiveInterruptHandle::new(),
                trace: Vec::new(),
                prepared: false,
                stop: false,
            }
        }
    }

    impl InteractiveGeometryDriver for Fake {
        fn interrupt_handle(&self) -> InteractiveInterruptHandle {
            self.interrupt.clone()
        }
        fn total_frames(&self) -> Result<usize> {
            Ok(3)
        }
        fn prepare_all(&mut self) -> Result<()> {
            self.prepared = true;
            Ok(())
        }
        fn compute(
            &mut self,
            frame: usize,
            _callback: &mut (dyn for<'r> FnMut(GeometryResults<'r>) -> ControlFlow<()> + Send),
            _generation: u64,
            stateful: bool,
        ) -> Result<InteractiveExecutionReport> {
            self.trace.push((frame, stateful));
            Ok(report(
                if self.stop {
                    InteractiveExecutionStatus::Interrupted
                } else {
                    InteractiveExecutionStatus::Complete
                },
                2,
            ))
        }
    }

    #[test]
    fn initial_yield_allows_interrupt_without_preparing_or_computing() {
        let mut executor = Fake::default();
        let interrupt = executor.interrupt.clone();
        let mut callback = |_: GeometryResults<'_>| ControlFlow::Continue(());
        let mut future = Box::pin(compute_all_frames(&mut executor, &mut callback));
        let mut cx = Context::from_waker(std::task::Waker::noop());
        assert!(future.as_mut().poll(&mut cx).is_pending());
        interrupt.interrupt();
        let Poll::Ready(Ok(result)) = future.as_mut().poll(&mut cx) else {
            panic!("expected interrupt")
        };
        assert_eq!(result.status, InteractiveExecutionStatus::Interrupted);
        assert_eq!(result.emitted_frames, 0);
        drop(future);
        assert!(!executor.prepared);
        assert!(executor.trace.is_empty());
    }

    #[test]
    fn all_frames_preserves_yields_stateful_order_and_report_counts() {
        for stop in [false, true] {
            let mut executor = Fake {
                stop,
                ..Fake::default()
            };
            let mut callback = |_: GeometryResults<'_>| ControlFlow::Continue(());
            let mut future = Box::pin(compute_all_frames(&mut executor, &mut callback));
            let mut cx = Context::from_waker(std::task::Waker::noop());
            let mut yields = 0;
            let result = loop {
                match future.as_mut().poll(&mut cx) {
                    Poll::Pending => yields += 1,
                    Poll::Ready(result) => break result.unwrap(),
                }
            };
            drop(future);
            assert!(executor.prepared);
            assert_eq!(
                executor.trace,
                if stop {
                    vec![(0, true)]
                } else {
                    vec![(0, true), (1, true), (2, true)]
                }
            );
            assert_eq!(result.emitted_frames, if stop { 2 } else { 6 });
            assert_eq!(yields, if stop { 1 } else { 4 });
            assert_eq!(
                result.status,
                if stop {
                    InteractiveExecutionStatus::Interrupted
                } else {
                    InteractiveExecutionStatus::Complete
                }
            );
        }
    }
}
