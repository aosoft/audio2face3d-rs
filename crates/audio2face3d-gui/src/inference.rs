//! Bounded session worker. Input and output progress independently of the UI.
use crate::core::{Clip, Error, Result, SessionState};
use audio2face3d::{
    client::Control,
    logging::Logger,
    types::{AudioFormat, OutputEvent},
};
use std::{
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::Receiver,
    },
    thread::JoinHandle,
};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    Local,
    Grpc,
    Mock,
}
#[derive(Clone)]
pub struct Request {
    pub mode: Mode,
    pub wav: PathBuf,
    pub model: PathBuf,
    pub endpoint: String,
    pub api_key: String,
    pub cuda_root: PathBuf,
    pub tensorrt_root: PathBuf,
    /// Host-resolved base configuration. GUI root fields override the corresponding SDK group.
    pub runtime: audio2face3d::runtime::NativeRuntimeConfig,
    pub device: usize,
    pub pace_input: bool,
}
impl Default for Request {
    fn default() -> Self {
        Self {
            mode: if cfg!(feature = "grpc") {
                Mode::Grpc
            } else {
                Mode::Local
            },
            wav: PathBuf::new(),
            model: PathBuf::new(),
            endpoint: "http://127.0.0.1:52000".into(),
            api_key: String::new(),
            cuda_root: PathBuf::new(),
            tensorrt_root: PathBuf::new(),
            runtime: Default::default(),
            device: 0,
            pace_input: false,
        }
    }
}
impl Request {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.wav.as_os_str().is_empty() {
            return Err(Error(
                "WAV is required. Select an audio file with Browse WAV.".into(),
            ));
        }
        if self.mode == Mode::Local && self.model.as_os_str().is_empty() {
            return Err(Error(
                "Model JSON is required for local inference. Select model.json with Browse model."
                    .into(),
            ));
        }
        Ok(())
    }
    /// Preserve configured directory lists/search policy when applying GUI root overrides.
    pub fn native_runtime(&self) -> Result<audio2face3d::runtime::NativeRuntimeConfig> {
        let mut builder = audio2face3d::runtime::NativeRuntimeConfig::builder()
            .search_policy(self.runtime.search_policy());
        for (cuda, override_root, root, dirs) in [
            (
                true,
                &self.cuda_root,
                self.runtime.cuda_root(),
                self.runtime.cuda_library_dirs(),
            ),
            (
                false,
                &self.tensorrt_root,
                self.runtime.tensorrt_root(),
                self.runtime.tensorrt_library_dirs(),
            ),
        ] {
            let root = if override_root.as_os_str().is_empty() {
                root
            } else {
                Some(override_root.as_path())
            };
            if let Some(root) = root {
                builder = if cuda {
                    builder.cuda_root(root)
                } else {
                    builder.tensorrt_root(root)
                };
            } else if !dirs.is_empty() {
                builder = if cuda {
                    builder.cuda_library_dirs(dirs.to_vec())
                } else {
                    builder.tensorrt_library_dirs(dirs.to_vec())
                };
            }
        }
        builder.build().map_err(|e| Error(e.to_string()))
    }
}
pub enum Event {
    InputFinished,
    Output(OutputEvent),
}
pub struct Job {
    pub id: u64,
    pub events: Receiver<Event>,
    cancelled: Arc<AtomicBool>,
    control: Arc<Mutex<Option<Control>>>,
    worker: Option<JoinHandle<Result<()>>>,
}
impl Job {
    pub fn start(id: u64, request: Request, logger: Arc<dyn Logger>) -> Result<Self> {
        request.validate()?;
        #[cfg(any(feature = "local", feature = "grpc", feature = "mock"))]
        {
            let (sender, events) = std::sync::mpsc::sync_channel(16);
            let cancelled = Arc::new(AtomicBool::new(false));
            let control = Arc::new(Mutex::new(None));
            let token = cancelled.clone();
            let slot = control.clone();
            let worker = std::thread::Builder::new()
                .name(format!("a2f-preview-{id}"))
                .spawn(move || worker::run(request, logger, sender, token, slot))
                .map_err(|e| Error(e.to_string()))?;
            Ok(Self {
                id,
                events,
                cancelled,
                control,
                worker: Some(worker),
            })
        }
        #[cfg(not(any(feature = "local", feature = "grpc", feature = "mock")))]
        {
            let _ = (id, request, logger);
            Err(Error("enable local or grpc feature to infer".into()))
        }
    }
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        if let Some(control) = self.control.lock().unwrap().as_ref() {
            control.cancel();
        }
    }
    pub fn try_finish(&mut self) -> Option<Result<()>> {
        if !self.worker.as_ref()?.is_finished() {
            return None;
        }
        Some(
            self.worker
                .take()
                .unwrap()
                .join()
                .unwrap_or_else(|_| Err(Error("inference worker panicked".into()))),
        )
    }
}
impl Drop for Job {
    fn drop(&mut self) {
        self.cancel();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

pub fn apply_event(clip: &mut Clip, event: OutputEvent) -> Result<()> {
    match event {
        OutputEvent::StreamInfo(info) => {
            if info.audio_format != Some(AudioFormat::MONO_16KHZ) {
                return Err(Error(
                    "preview requires returned mono PCM16/16kHz audio".into(),
                ));
            }
            clip.set_names(
                info.curves
                    .ok_or_else(|| Error("missing curve layout".into()))?
                    .names()
                    .to_vec(),
            )?;
        }
        OutputEvent::Audio(audio) => {
            if audio.format() != AudioFormat::MONO_16KHZ {
                return Err(Error("audio format changed".into()));
            }
            let values: Vec<_> = audio
                .pcm()
                .as_bytes()
                .chunks_exact(2)
                .map(|v| i16::from_le_bytes([v[0], v[1]]) as f32 / 32768.)
                .collect();
            clip.push_audio(audio.position().0, &values)?;
        }
        OutputEvent::Curves(frame) => {
            if clip.names != frame.layout().names() {
                return Err(Error("curve layout mismatch".into()));
            }
            let (_, time, values) = frame.into_parts();
            clip.push_frame(time.as_seconds(), values)?;
        }
        OutputEvent::Completed(_) => {
            clip.session = SessionState::Completed;
            if clip.ready_until() + 1e-6 < clip.duration() {
                clip.session = SessionState::Failed("incomplete curve coverage".into());
                return Err(Error("incomplete curve coverage".into()));
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(any(feature = "local", feature = "grpc", feature = "mock"))]
mod worker;
