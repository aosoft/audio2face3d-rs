use std::{future::pending, sync::Arc, time::Duration};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;
use tonic::Status;

/// Tokio's execution semaphore orders waiters FIFO. The second semaphore only
/// bounds the waiting population; its permit is released before execution.
pub(crate) struct Admission {
    slots: Arc<Semaphore>,
    waiting: Arc<Semaphore>,
    timeout: Duration,
}

impl Admission {
    pub(crate) fn new(active: usize, queued: usize, timeout: Duration) -> Self {
        Self {
            slots: Arc::new(Semaphore::new(active)),
            waiting: Arc::new(Semaphore::new(queued)),
            timeout,
        }
    }

    pub(crate) async fn acquire(
        &self,
        shutdown: &CancellationToken,
    ) -> Result<OwnedSemaphorePermit, Status> {
        if shutdown.is_cancelled() {
            return Err(Status::unavailable("server shutting down"));
        }
        if let Ok(permit) = self.slots.clone().try_acquire_owned() {
            return Ok(permit);
        }
        let _waiting = self
            .waiting
            .clone()
            .try_acquire_owned()
            .map_err(|_| Status::resource_exhausted("request queue full"))?;
        tracing::debug!("waiting for execution slot");
        let deadline = async {
            if self.timeout.is_zero() {
                pending::<()>().await;
            } else {
                tokio::time::sleep(self.timeout).await;
            }
        };
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => Err(Status::unavailable("server shutting down")),
            _ = deadline => Err(Status::deadline_exceeded("request queue wait timed out")),
            permit = self.slots.clone().acquire_owned() => {
                permit.map_err(|_| Status::unavailable("execution queue closed"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        future::{Future, poll_fn},
        pin::Pin,
        task::Poll,
    };
    use tonic::Code;

    async fn assert_waiting<F: Future>(mut future: Pin<&mut F>) {
        poll_fn(|cx| {
            assert!(future.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
    }

    #[tokio::test]
    async fn fifo_uses_any_free_slot_for_one_or_multiple_engines() {
        for active in [1, 2, 4] {
            let queue = Admission::new(active, 3, Duration::ZERO);
            let shutdown = CancellationToken::new();
            let mut running = Vec::new();
            for _ in 0..active {
                running.push(queue.acquire(&shutdown).await.unwrap());
            }
            let first = queue.acquire(&shutdown);
            let second = queue.acquire(&shutdown);
            let third = queue.acquire(&shutdown);
            tokio::pin!(first, second, third);
            assert_waiting(first.as_mut()).await;
            assert_waiting(second.as_mut()).await;
            assert_waiting(third.as_mut()).await;
            assert_eq!(
                queue.acquire(&shutdown).await.unwrap_err().code(),
                Code::ResourceExhausted
            );
            drop(running.pop());
            let one = first.await.unwrap();
            assert_waiting(second.as_mut()).await;
            drop(one);
            let two = second.await.unwrap();
            assert_waiting(third.as_mut()).await;
            drop(two);
            drop(third.await.unwrap());
            drop(running);
            assert_eq!(queue.slots.available_permits(), active);
            assert_eq!(queue.waiting.available_permits(), 3);
        }
    }

    #[tokio::test]
    async fn cancelled_waiter_leaves_queue_and_preserves_fifo() {
        let queue = Admission::new(1, 2, Duration::ZERO);
        let shutdown = CancellationToken::new();
        let active = queue.acquire(&shutdown).await.unwrap();
        let mut cancelled = Box::pin(queue.acquire(&shutdown));
        assert_waiting(cancelled.as_mut()).await;
        let next = queue.acquire(&shutdown);
        tokio::pin!(next);
        assert_waiting(next.as_mut()).await;
        drop(cancelled);
        assert_eq!(queue.waiting.available_permits(), 1);
        let last = queue.acquire(&shutdown);
        tokio::pin!(last);
        assert_waiting(last.as_mut()).await;
        drop(active);
        let next = next.await.unwrap();
        assert_waiting(last.as_mut()).await;
        drop(next);
        drop(last.await.unwrap());
    }

    #[tokio::test]
    async fn waiting_timeout_and_shutdown_do_not_leak_capacity() {
        let queue = Admission::new(1, 1, Duration::from_millis(20));
        let shutdown = CancellationToken::new();
        let active = queue.acquire(&shutdown).await.unwrap();
        assert_eq!(
            queue.acquire(&shutdown).await.unwrap_err().code(),
            Code::DeadlineExceeded
        );
        assert_eq!(queue.waiting.available_permits(), 1);
        let waiting = queue.acquire(&shutdown);
        tokio::pin!(waiting);
        assert_waiting(waiting.as_mut()).await;
        shutdown.cancel();
        assert_eq!(waiting.await.unwrap_err().code(), Code::Unavailable);
        assert_eq!(queue.waiting.available_permits(), 1);
        drop(active);
        assert_eq!(queue.slots.available_permits(), 1);
    }
}
