use super::{Event, Mode, Request};
use crate::core::{Error, Result};
use audio2face3d::{
    Audio2Face3DContext,
    client::{Client, Control},
    logging::{LogLevel, LogRecord, Logger},
    types::*,
};
use std::{
    future::Future,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{SyncSender, TrySendError},
    },
    task::{Context, Poll, Wake, Waker},
    time::{Duration, Instant},
};
struct Signal(std::thread::Thread);
impl Wake for Signal {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
}
fn wait<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(Signal(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        if let Poll::Ready(result) = future.as_mut().poll(&mut cx) {
            return result;
        }
        std::thread::park();
    }
}
fn send(sender: &SyncSender<Event>, mut event: Event, cancelled: &AtomicBool) -> Result<()> {
    loop {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error("cancelled".into()));
        }
        match sender.try_send(event) {
            Ok(()) => return Ok(()),
            Err(TrySendError::Full(value)) => {
                event = value;
                std::thread::sleep(Duration::from_millis(2));
            }
            Err(TrySendError::Disconnected(_)) => return Err(Error("GUI receiver closed".into())),
        }
    }
}
fn error(e: impl std::fmt::Display) -> Error {
    Error(e.to_string())
}

pub fn run(
    request: Request,
    logger: Arc<dyn Logger>,
    sender: SyncSender<Event>,
    cancelled: Arc<AtomicBool>,
    control_slot: Arc<Mutex<Option<Control>>>,
) -> Result<()> {
    logger.write_log(
        LogLevel::Info,
        LogRecord::new("Loading WAV").field("source", "gui.inference"),
    );
    let bytes = crate::wav::load(&request.wav)?;
    if cancelled.load(Ordering::Acquire) {
        return Err(Error("cancelled".into()));
    }
    let context = Audio2Face3DContext::builder()
        .logger(logger.clone())
        .native_runtime(request.native_runtime()?)
        .build();
    #[cfg(feature = "grpc")]
    let runtime = if request.mode == Mode::Grpc {
        Some(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(error)?,
        )
    } else {
        None
    };
    let client = match request.mode {
        #[cfg(feature = "local")]
        Mode::Local => {
            let engine = audio2face3d::client::InferenceConfig::builder(
                audio2face3d::client::BackendKind::Regression,
            )
            .model(&request.model)
            .device(request.device)
            .max_audio_seconds(600)
            .build()
            .map_err(error)?;
            wait(Client::direct_with_context(
                audio2face3d::client::DirectConfig::builder(engine)
                    .build()
                    .map_err(error)?,
                context,
            ))
            .map_err(error)?
        }
        #[cfg(feature = "mock")]
        Mode::Mock => {
            let engine = audio2face3d::client::InferenceConfig::builder(
                audio2face3d::client::BackendKind::Mock,
            )
            .build()
            .map_err(error)?;
            wait(Client::direct_with_context(
                audio2face3d::client::DirectConfig::builder(engine)
                    .build()
                    .map_err(error)?,
                context,
            ))
            .map_err(error)?
        }
        #[cfg(feature = "grpc")]
        Mode::Grpc => {
            let rt = runtime.as_ref().expect("gRPC runtime initialized above");
            let config = audio2face3d::client::ServerConfig::builder(&request.endpoint)
                .runtime(rt.handle().clone())
                .optional_api_key((!request.api_key.is_empty()).then_some(request.api_key))
                .max_message_bytes(1024 * 1024)
                .build()
                .map_err(error)?;
            wait(Client::server_with_context(config, context)).map_err(error)?
        }
        #[allow(unreachable_patterns)]
        _ => return Err(Error("inference mode not enabled in this build".into())),
    };
    let result = (|| -> Result<()> {
        if cancelled.load(Ordering::Acquire) {
            return Err(Error("cancelled".into()));
        }
        let options = RequestOptions::builder(AudioFormat::MONO_16KHZ)
            .timeout(Duration::from_secs(1200))
            .build()
            .map_err(error)?;
        let (mut input, mut output, control) = client.start(options).map_err(error)?.split();
        *control_slot.lock().unwrap() = Some(control.clone());
        if cancelled.load(Ordering::Acquire) {
            control.cancel();
        }
        std::thread::scope(|scope| {
            let input_control = control.clone();
            let token = cancelled.clone();
            let events = sender.clone();
            let input_worker = scope.spawn(move || -> Result<()> {
                let result = (|| {
                    let start = Instant::now();
                    for (index, chunk) in bytes.chunks(3200).enumerate() {
                        if token.load(Ordering::Acquire) {
                            return Err(Error("cancelled".into()));
                        }
                        if request.pace_input && index > 4 {
                            let due = Duration::from_millis((index as u64 - 4) * 100);
                            while start.elapsed() < due && !token.load(Ordering::Acquire) {
                                std::thread::sleep(Duration::from_millis(2));
                            }
                        }
                        wait(input.send(InputChunk::new(
                            PcmBuffer::from_vec(chunk.to_vec()).map_err(error)?,
                            vec![],
                        )))
                        .map_err(error)?;
                    }
                    wait(input.finish()).map_err(error)?;
                    send(&events, Event::InputFinished, &token)
                })();
                if result.is_err() {
                    input_control.cancel();
                }
                result
            });
            let received = (|| -> Result<()> {
                let mut completed = false;
                while let Some(event) = wait(output.recv()).map_err(error)? {
                    if let OutputEvent::Diagnostic(ref d) = event {
                        logger.write_log(
                            LogLevel::Info,
                            LogRecord::new(&d.message).field("source", "inference"),
                        );
                    }
                    if matches!(event, OutputEvent::Completed(_)) {
                        completed = true;
                    }
                    send(&sender, Event::Output(event), &cancelled)?;
                }
                if !completed {
                    return Err(Error("response ended without successful completion".into()));
                }
                Ok(())
            })();
            if received.is_err() {
                control.cancel();
            }
            let sent = input_worker
                .join()
                .unwrap_or_else(|_| Err(Error("input worker panicked".into())));
            let closed = wait(control.closed()).map_err(error);
            received.and(sent).and(closed)
        })
    })();
    let shutdown = wait(client.shutdown()).map_err(error);
    *control_slot.lock().unwrap() = None;
    #[cfg(feature = "grpc")]
    drop(runtime);
    logger.write_log(
        LogLevel::Info,
        LogRecord::new("Inference resources released").field("source", "gui.inference"),
    );
    result.and(shutdown)
}
