use crate::client::{Client, Limits, driver::*, executor, types::*};
use crate::inference::{self as engine, admission::Admission};
use crate::{Audio2Face3DContext, logging::integration::LogScope};
use std::{
    future::{Future, poll_fn},
    pin::pin,
    sync::{Arc, Mutex},
    task::Poll,
    time::Duration,
};

/// Direct inference configuration. `runtime` enables native Regression/A2E.
#[derive(Clone, Debug)]
pub struct DirectConfig {
    pub engine: engine::Config,
    pub limits: Limits,
    pub max_executions: usize,
    pub max_queued: usize,
    /// Zero means no admission timeout; RequestOptions.timeout still applies.
    pub queue_timeout: Duration,
}
impl Default for DirectConfig {
    fn default() -> Self {
        Self {
            engine: engine::Config::default(),
            limits: Limits::default(),
            max_executions: 1,
            max_queued: 64,
            queue_timeout: Duration::ZERO,
        }
    }
}
struct Direct {
    executor: Arc<executor::Executor>,
    factory: Arc<engine::Factory>,
    admission: Admission,
    gate: Mutex<bool>,
}
impl Client {
    async fn direct_inner(config: DirectConfig) -> Result<Self> {
        config.limits.validate()?;
        config.engine.validate()?;
        let admission = Admission::new(
            config.max_executions,
            config.max_queued,
            config.queue_timeout,
        )?;
        let executor = executor::Executor::new()?;
        let (tx, rx) = executor::channel();
        // Executor ownership travels with initialization; dropping the caller cannot leak it.
        let holder = Arc::new(executor);
        holder.spawn(async move {
            let result = engine::Factory::prepare(config.engine).await;
            tx.send(result);
        })?;
        let factory = match rx.await {
            Ok(f) => f,
            Err(e) => {
                holder.stop(async {});
                return Err(e);
            }
        };
        let executor = holder;
        let driver = Arc::new(Direct {
            executor,
            factory: Arc::new(factory),
            admission,
            gate: Mutex::new(false),
        });
        Self::with_driver(config.limits, driver)
    }
}
impl Driver for Direct {
    fn launch(&self, options: RequestOptions, session: WorkerSession) -> Result<()> {
        let closing = self.gate.lock().unwrap();
        if *closing {
            return Err(Error::new(ErrorKind::ShuttingDown, "direct client closed"));
        }
        let cancellation = engine::Cancellation::new();
        let admission = self.admission.acquire(&cancellation);
        let factory = self.factory.clone();
        self.executor.spawn(async move {
            let (mut reader, mut writer, guard) = session.split();
            let work = async {
                let permit = interrupt(admission, &guard, &cancellation).await?;
                // Loading cannot be abandoned: its owner must complete native cleanup.
                let mut backend = factory.start(options).await?;
                let result = run(
                    &mut reader,
                    &mut writer,
                    &guard,
                    &cancellation,
                    backend.as_mut(),
                )
                .await;
                if let Err(e) = &result {
                    guard.fail(e.clone());
                }
                let cleanup = backend.close().await;
                drop(backend);
                drop(permit);
                result.and(cleanup)
            }
            .await;
            guard.finish(work);
        })
    }
    fn shutdown(&self, completion: DriverShutdown) {
        *self.gate.lock().unwrap() = true;
        self.admission.close();
        let factory = self.factory.clone();
        self.executor.stop(async move {
            completion.finish(factory.release_prepared().await);
        });
    }
}
async fn interrupt<T>(
    future: impl Future<Output = Result<T>>,
    guard: &WorkerGuard,
    cancel: &engine::Cancellation,
) -> Result<T> {
    let mut future = pin!(future);
    let mut stopped = pin!(guard.cancelled());
    poll_fn(|cx| {
        if let Poll::Ready(e) = stopped.as_mut().poll(cx) {
            cancel.cancel();
            return Poll::Ready(Err(e));
        }
        future.as_mut().poll(cx)
    })
    .await
}
async fn run(
    reader: &mut Reader,
    writer: &mut Writer,
    guard: &WorkerGuard,
    cancel: &engine::Cancellation,
    backend: &mut dyn engine::Backend,
) -> Result<()> {
    interrupt(
        writer.emit(OutputEvent::StreamInfo(StreamInfo::new(
            Some(AudioFormat::MONO_16KHZ),
            Some(engine::animation::layout()),
        ))),
        guard,
        cancel,
    )
    .await?;
    loop {
        let input = interrupt(reader.recv(), guard, cancel).await?;
        let finished = input.is_none();
        if let Some(input) = input {
            let (chunk, lease) = input.into_parts();
            backend.push(chunk).await?;
            drop(lease);
        } else {
            backend.finish().await?;
        }
        loop {
            let frame = interrupt(backend.next_frame(cancel), guard, cancel).await?;
            let Some(frame) = frame else { break };
            emit_batch(writer, frame, guard, cancel).await?;
        }
        if finished {
            interrupt(writer.emit(OutputEvent::ProcessingFinished), guard, cancel).await?;
            return Ok(());
        }
    }
}
async fn emit_batch(
    writer: &mut Writer,
    batch: OutputBatch,
    guard: &WorkerGuard,
    cancel: &engine::Cancellation,
) -> Result<()> {
    if let Some(audio) = batch.audio {
        interrupt(writer.emit(OutputEvent::Audio(audio)), guard, cancel).await?;
    }
    for frame in batch.curves {
        interrupt(writer.emit(OutputEvent::Curves(frame)), guard, cancel).await?;
    }
    if let Some(emotion) = batch.emotion {
        interrupt(writer.emit(OutputEvent::Emotion(emotion)), guard, cancel).await?;
    }
    for diagnostic in batch.diagnostics {
        interrupt(
            writer.emit(OutputEvent::Diagnostic(diagnostic)),
            guard,
            cancel,
        )
        .await?;
    }
    Ok(())
}

impl Client {
    pub async fn direct(config: DirectConfig) -> Result<Self> {
        let scope = LogScope::capture();
        scope.wrap_future(Self::direct_inner(config)).await
    }
    pub async fn direct_with_context(
        config: DirectConfig,
        context: Audio2Face3DContext,
    ) -> Result<Self> {
        let scope = LogScope::new(context);
        scope.wrap_future(Self::direct_inner(config)).await
    }
}
