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
    cache: &mut Cache,
    request: Request,
    logger: Arc<dyn Logger>,
    sender: SyncSender<Event>,
    cancelled: Arc<AtomicBool>,
    control_slot: Arc<Mutex<Option<Control>>>,
) -> Result<()> {
    let result = run_inner(
        cache,
        request,
        logger,
        sender,
        cancelled.clone(),
        control_slot.clone(),
    );
    *control_slot.lock().unwrap() = None;
    if result.is_err() && !cancelled.load(Ordering::Acquire) {
        cache.backend.take();
    }
    result
}
fn run_inner(
    cache: &mut Cache,
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
    let key = Key::new(&request)?;
    if cache
        .backend
        .as_ref()
        .is_some_and(|backend| backend.key != key)
    {
        cache.backend.take();
    }
    if cache.backend.is_none() {
        logger.write_log(
            LogLevel::Info,
            LogRecord::new("Initializing inference backend").field("source", "gui.inference"),
        );
        cache.backend = Some(create(&request, logger.clone(), key)?);
    } else {
        logger.write_log(
            LogLevel::Info,
            LogRecord::new("Reusing inference backend").field("source", "gui.inference"),
        );
    }
    let client = cache.backend.as_ref().unwrap().client.as_ref().unwrap();
    if cancelled.load(Ordering::Acquire) {
        return Err(Error("cancelled".into()));
    }
    let options = request.options()?;
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
        let mut collected = (!request.pace_input).then(crate::core::Clip::running);
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
                #[cfg(feature = "emotion")]
                if request.mode == Mode::Local
                    && request.emotion.model.is_none()
                    && matches!(event, OutputEvent::Emotion(_))
                {
                    continue;
                }
                if let Some(clip) = &mut collected {
                    super::apply_event(clip, event)?;
                } else {
                    send(&sender, Event::Output(event), &cancelled)?;
                }
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
        received.and(sent).and(closed)?;
        if let Some(clip) = collected {
            send(&sender, Event::Ready(Box::new(clip)), &cancelled)?;
        }
        Ok(())
    })
}
fn create(request: &Request, logger: Arc<dyn Logger>, key: Key) -> Result<CachedBackend> {
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
            .max_audio_seconds(600);
            #[cfg(feature = "emotion")]
            let engine = engine.optional_emotion_model(request.emotion.model.clone());
            let engine = engine.build().map_err(error)?;
            wait(Client::direct_with_context(
                audio2face3d::client::DirectConfig::builder(engine)
                    .reuse_model(true)
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
                    .reuse_model(true)
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
                .optional_api_key((!request.api_key.is_empty()).then_some(request.api_key.clone()))
                .max_message_bytes(1024 * 1024)
                .build()
                .map_err(error)?;
            wait(Client::server_with_context(config, context)).map_err(error)?
        }
        #[allow(unreachable_patterns)]
        _ => return Err(Error("inference mode not enabled in this build".into())),
    };
    Ok(CachedBackend {
        key,
        client: Some(client),
        #[cfg(feature = "grpc")]
        runtime,
    })
}

#[derive(Default)]
pub(super) struct Cache {
    backend: Option<CachedBackend>,
}
// Keys exclude WAV, playback pacing and settings irrelevant to the selected mode.
#[derive(PartialEq)]
enum Key {
    Local {
        model: std::path::PathBuf,
        #[cfg(feature = "emotion")]
        emotion_model: Option<std::path::PathBuf>,
        device: usize,
        runtime: audio2face3d::runtime::NativeRuntimeConfig,
    },
    Grpc {
        endpoint: String,
        api_key: String,
    },
    Mock,
}
impl Key {
    fn new(request: &Request) -> Result<Self> {
        Ok(match request.mode {
            Mode::Local => Self::Local {
                model: request.model.clone(),
                #[cfg(feature = "emotion")]
                emotion_model: request.emotion.model.clone(),
                device: request.device,
                runtime: request.native_runtime()?,
            },
            Mode::Grpc => Self::Grpc {
                endpoint: request.endpoint.clone(),
                api_key: request.api_key.clone(),
            },
            Mode::Mock => Self::Mock,
            Mode::Disabled => return Err(Error("inference disabled".into())),
        })
    }
}
struct CachedBackend {
    key: Key,
    client: Option<Client>,
    #[cfg(feature = "grpc")]
    runtime: Option<tokio::runtime::Runtime>,
}
impl Drop for CachedBackend {
    fn drop(&mut self) {
        let client = self.client.take();
        #[cfg(feature = "grpc")]
        let runtime = self.runtime.take();
        // Library hosts may drop the cache inside Tokio. Release outside that runtime.
        let _ = std::thread::spawn(move || {
            if let Some(client) = client {
                let _ = wait(client.shutdown());
            }
            #[cfg(feature = "grpc")]
            drop(runtime);
        })
        .join();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reuse_key_tracks_backend_settings_only() {
        let mut request = Request {
            mode: Mode::Local,
            model: "model.json".into(),
            ..Default::default()
        };
        let original = Key::new(&request).unwrap();
        request.wav = "other.wav".into();
        request.pace_input = true;
        request.stream_buffer_ms = 500;
        request.endpoint = "unused".into();
        assert!(original == Key::new(&request).unwrap());
        request.model = "another.json".into();
        assert!(original != Key::new(&request).unwrap());
        request.model = "model.json".into();
        request.device = 1;
        assert!(original != Key::new(&request).unwrap());
        request.device = 0;
        request.cuda_root = std::env::temp_dir().join("another-cuda");
        assert!(original != Key::new(&request).unwrap());
        request.mode = Mode::Grpc;
        let grpc = Key::new(&request).unwrap();
        assert!(original != grpc);
        request.api_key = "changed".into();
        assert!(grpc != Key::new(&request).unwrap());
        request.api_key.clear();
        request.endpoint = "another-server".into();
        assert!(grpc != Key::new(&request).unwrap());
    }
}
