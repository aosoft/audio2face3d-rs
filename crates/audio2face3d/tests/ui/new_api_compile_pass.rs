use audio2face3d::audio2x::{
    AudioAccumulator, EmotionAccumulator, Execution, ExecutionReport, Executor, ExecutorFuture,
    InteractiveExecutionReport, InteractiveExecutor,
};
#[cfg(any(feature = "animation", feature = "emotion"))]
use audio2face3d::audio2x::Result;
#[cfg(any(feature = "animation", feature = "emotion"))]
use std::ops::ControlFlow;
use std::sync::Arc;

fn assert_send<T: Send>() {}

fn consume_execution(execution: Execution) -> ExecutorFuture<'static, ExecutionReport> {
    execution.wait_all()
}

#[cfg(feature = "animation")]
fn face_pipeline(
    executor: &mut dyn audio2face3d::audio2face::GeometryExecutor,
) -> Result<Execution> {
    use audio2face3d::audio2face::{FaceEmotions, GeometryCallbacks, GeometryResults};

    let mut frame_count = 0;
    let mut geometry = |_: GeometryResults<'_>| {
        frame_count += 1;
        ControlFlow::Continue(())
    };
    let mut emotions = |_: FaceEmotions<'_>| {};
    let execution = executor.execute(GeometryCallbacks {
        results: &mut geometry,
        emotions: Some(&mut emotions),
    })?;
    let _frames_observed_after_execute = frame_count;
    Ok(execution)
}

#[cfg(feature = "emotion")]
fn emotion_pipeline(
    executor: &mut dyn audio2face3d::audio2emotion::EmotionExecutor,
) -> Result<Execution> {
    let mut frame_count = 0;
    let execution = executor.execute(&mut |_| {
        frame_count += 1;
        ControlFlow::Continue(())
    })?;
    let _frames_observed_after_execute = frame_count;
    Ok(execution)
}

#[cfg(feature = "animation")]
struct CustomRunner;

#[cfg(feature = "animation")]
impl audio2face3d::audio2face::JobRunner for CustomRunner {
    fn enqueue(&self, task: audio2face3d::audio2face::JobRunnerTask) -> Result<()> {
        task.run();
        Ok(())
    }
}

#[cfg(feature = "animation")]
fn inject_custom_runner<'a>(
    components: audio2face3d::audio2face::BlendshapeSolveExecutorCreationParameters<'a>,
) -> audio2face3d::audio2face::HostBlendshapeSolveExecutorCreationParameters<'a> {
    audio2face3d::audio2face::HostBlendshapeSolveExecutorCreationParameters {
        components,
        job_runner: Some(Arc::new(CustomRunner)),
    }
}

#[cfg(all(feature = "animation", feature = "tensorrt"))]
async fn run_regression_pipeline(
    parameters: audio2face3d::audio2face::regression::RegressionGeometryExecutorCreationParameters,
) -> Result<ExecutionReport> {
    use audio2face3d::audio2face::regression::RegressionGeometryExecutorFactory;
    use audio2face3d::audio2face::{GeometryCallbacks, GeometryExecutor};

    let mut executor = RegressionGeometryExecutorFactory::load(parameters).await?;
    let mut callback = |_: audio2face3d::audio2face::GeometryResults<'_>| {
        ControlFlow::Continue(())
    };
    let execution = executor.execute(GeometryCallbacks {
        results: &mut callback,
        emotions: None,
    })?;
    execution.await
}

#[cfg(all(feature = "emotion", feature = "tensorrt"))]
async fn run_classifier_pipeline(
    parameters: audio2face3d::audio2emotion::classifier::ClassifierEmotionExecutorCreationParameters,
) -> Result<ExecutionReport> {
    use audio2face3d::audio2emotion::EmotionExecutor;
    use audio2face3d::audio2emotion::classifier::ClassifierEmotionExecutorFactory;

    let mut executor = ClassifierEmotionExecutorFactory::load(parameters).await?;
    let execution = executor.execute(
        &mut |_: audio2face3d::audio2emotion::EmotionResults<'_>| ControlFlow::Continue(()),
    )?;
    execution.await
}

fn main() {
    assert_send::<Execution>();
    assert_send::<ExecutorFuture<'static, ExecutionReport>>();
    assert_send::<ExecutorFuture<'static, InteractiveExecutionReport>>();
    let _: Option<&dyn Executor> = None;
    let _: Option<&dyn InteractiveExecutor> = None;
    let _ = consume_execution;
    let _ = Arc::new(AudioAccumulator::new(16, 1).unwrap());
    let _ = Arc::new(EmotionAccumulator::new(2, 1).unwrap());

    #[cfg(feature = "animation")]
    {
        let _ = face_pipeline;
        let _ = inject_custom_runner;
        let _: Option<&dyn audio2face3d::audio2face::GeometryExecutor> = None;
        let _: Option<&dyn audio2face3d::audio2face::GeometryInteractiveExecutor> = None;
    }

    #[cfg(feature = "emotion")]
    {
        let _ = emotion_pipeline;
        let _: Option<&dyn audio2face3d::audio2emotion::EmotionExecutor> = None;
        let _: Option<&dyn audio2face3d::audio2emotion::EmotionInteractiveExecutor> = None;
    }

    #[cfg(all(feature = "animation", feature = "tensorrt"))]
    {
        use audio2face3d::audio2face::regression::{
            RegressionGeometryExecutor, RegressionGeometryExecutorFactory,
            RegressionGeometryInteractiveExecutor,
            RegressionGeometryInteractiveExecutorFactory,
        };
        assert_send::<RegressionGeometryExecutor>();
        assert_send::<RegressionGeometryInteractiveExecutor>();
        let _ = RegressionGeometryExecutorFactory::load;
        let _ = RegressionGeometryInteractiveExecutorFactory::load;
        let _ = run_regression_pipeline;
    }

    #[cfg(all(feature = "emotion", feature = "tensorrt"))]
    {
        use audio2face3d::audio2emotion::classifier::{
            ClassifierEmotionExecutor, ClassifierEmotionExecutorFactory,
            ClassifierEmotionInteractiveExecutor, ClassifierEmotionInteractiveExecutorFactory,
        };
        assert_send::<ClassifierEmotionExecutor>();
        assert_send::<ClassifierEmotionInteractiveExecutor>();
        let _ = ClassifierEmotionExecutorFactory::load;
        let _ = ClassifierEmotionInteractiveExecutorFactory::load;
        let _ = run_classifier_pipeline;
    }
}
