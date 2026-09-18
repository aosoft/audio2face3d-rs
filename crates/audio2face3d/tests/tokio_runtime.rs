#![cfg(feature = "animation")]
use audio2face3d::audio2x::{
    Execution, ExecutionReport, ExecutorFuture, InteractiveExecutionReport,
    InteractiveExecutionStatus, Result,
};

fn spawn_execution(execution: Execution) -> tokio::task::JoinHandle<Result<ExecutionReport>> {
    tokio::spawn(execution)
}

#[tokio::test]
async fn executor_future_can_be_spawned_and_selected() {
    let future: ExecutorFuture<'static, InteractiveExecutionReport> = Box::pin(async {
        tokio::task::yield_now().await;
        Ok(InteractiveExecutionReport {
            status: InteractiveExecutionStatus::Complete,
            emitted_frames: 3,
        })
    });
    let task = tokio::spawn(future);

    let report = tokio::select! {
        result = task => result.expect("spawned executor future panicked").unwrap(),
        _ = std::future::pending::<()>() => unreachable!(),
    };

    assert_eq!(report.status, InteractiveExecutionStatus::Complete);
    assert_eq!(report.emitted_frames, 3);
    let _ = spawn_execution;
}
