use audio2face3d_inference::{
    Cancellation,
    admission::{Admission as Queue, Permit},
};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tonic::Status;

/// Only the transport cancellation bridge lives here; all queue policy is shared.
pub(crate) struct Admission(Queue);
impl Admission {
    pub(crate) fn new(active: usize, queued: usize, timeout: Duration) -> Result<Self, Status> {
        Queue::new(active, queued, timeout)
            .map(Self)
            .map_err(crate::backend::status)
    }
    pub(crate) async fn acquire(&self, shutdown: &CancellationToken) -> Result<Permit, Status> {
        if shutdown.is_cancelled() {
            return Err(Status::unavailable("server shutting down"));
        }
        let cancel = Cancellation::new();
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => Err(Status::unavailable("server shutting down")),
            result = self.0.acquire(&cancel) => result.map_err(crate::backend::status),
        }
    }
}
