//! CPAL streams are owned by a worker; media commands never wait for device setup.
use crate::{
    core::{Error, Result},
    playback::{Command, PlaybackState, Player},
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    Arc, Condvar, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

#[cfg(test)]
mod tests;

pub type SharedPlayer = Arc<Mutex<Player>>;
#[derive(Default, Clone, Copy)]
struct DeviceRequest {
    generation: u64,
    running: bool,
    shutdown: bool,
}
type Requests = Arc<(Mutex<DeviceRequest>, Condvar)>;

pub struct AudioOutput {
    pub player: SharedPlayer,
    generation: Arc<AtomicU64>,
    requests: Requests,
    worker: Option<std::thread::JoinHandle<()>>,
    pub callback_contention: Arc<AtomicU64>,
    pub errors: Arc<Mutex<Option<String>>>,
}
impl AudioOutput {
    pub fn new(player: SharedPlayer) -> Self {
        Self::with_stream_factory(player, Device::start)
    }
    fn with_stream_factory<S: 'static>(
        player: SharedPlayer,
        start: impl FnMut(&Device, u64) -> Result<S> + Send + 'static,
    ) -> Self {
        let generation = Arc::new(AtomicU64::new(0));
        let requests = Arc::new((Mutex::new(DeviceRequest::default()), Condvar::new()));
        let callback_contention = Arc::new(AtomicU64::new(0));
        let errors = Arc::new(Mutex::new(None));
        let device = Device {
            player: player.clone(),
            generation: generation.clone(),
            contention: callback_contention.clone(),
            errors: errors.clone(),
        };
        let worker_requests = requests.clone();
        let worker = std::thread::spawn(move || device.run(worker_requests, start));
        Self {
            player,
            generation,
            requests,
            worker: Some(worker),
            callback_contention,
            errors,
        }
    }
    /// Updates the media position synchronously; device errors arrive through `errors`.
    pub fn command(&mut self, command: Command) -> Result<()> {
        let mut player = self.player.lock().unwrap();
        if matches!(command, Command::SetLoop(_)) {
            return player.command(command, Instant::now());
        }
        // Restarting Play retains the audible position, not the queued-audio end.
        if matches!(command, Command::Play)
            && matches!(
                player.state,
                PlaybackState::Playing | PlaybackState::Buffering
            )
        {
            player.command(Command::Pause, Instant::now())?;
        }
        player.command(command, Instant::now())?;
        // Invalidate callbacks under the same lock that protects the new position.
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let running = matches!(
            player.state,
            PlaybackState::Playing | PlaybackState::Buffering
        );
        if matches!(command, Command::Play) {
            *self.errors.lock().unwrap() = None;
        }
        drop(player);
        *self.requests.0.lock().unwrap() = DeviceRequest {
            generation,
            running,
            shutdown: false,
        };
        self.requests.1.notify_one();
        Ok(())
    }
    pub fn stop(&mut self) {
        let _ = self.command(Command::Pause);
    }
}
impl Drop for AudioOutput {
    fn drop(&mut self) {
        self.stop();
        self.requests.0.lock().unwrap().shutdown = true;
        self.requests.1.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Clone)]
struct Device {
    player: SharedPlayer,
    generation: Arc<AtomicU64>,
    contention: Arc<AtomicU64>,
    errors: Arc<Mutex<Option<String>>>,
}
impl Device {
    fn run<S>(&self, requests: Requests, mut start: impl FnMut(&Self, u64) -> Result<S>) {
        let mut stream = None;
        let mut seen = 0;
        loop {
            let request = {
                let mut pending = requests.0.lock().unwrap();
                while pending.generation == seen && !pending.shutdown {
                    pending = requests.1.wait(pending).unwrap();
                }
                *pending
            };
            // Both stream destruction and creation can block in OS audio APIs.
            drop(stream.take());
            if request.shutdown {
                break;
            }
            seen = request.generation;
            if !request.running || self.generation.load(Ordering::SeqCst) != seen {
                continue;
            }
            match start(self, seen) {
                Ok(created) if self.generation.load(Ordering::SeqCst) == seen => {
                    stream = Some(created)
                }
                Ok(_) => {} // Superseded while opening: discard without touching the player.
                Err(error) => {
                    let mut player = self.player.lock().unwrap();
                    if self.generation.load(Ordering::SeqCst) == seen {
                        let _ = player.command(Command::Pause, Instant::now());
                        *self.errors.lock().unwrap() = Some(error.to_string());
                    }
                }
            }
        }
    }
    fn start(&self, generation: u64) -> Result<cpal::Stream> {
        let device = cpal::default_host()
            .default_output_device()
            .ok_or_else(|| Error("no audio output device".into()))?;
        let config = device
            .default_output_config()
            .map_err(|e| Error(e.to_string()))?;
        if config.sample_rate().0 < 16_000 {
            return Err(Error("output device must support at least 16 kHz".into()));
        }
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => self.build::<f32>(&device, &config.into(), generation),
            cpal::SampleFormat::I16 => self.build::<i16>(&device, &config.into(), generation),
            cpal::SampleFormat::U16 => self.build::<u16>(&device, &config.into(), generation),
            other => return Err(Error(format!("unsupported output format: {other:?}"))),
        }?;
        if self.generation.load(Ordering::SeqCst) == generation {
            stream.play().map_err(|e| Error(e.to_string()))?;
        }
        Ok(stream)
    }
    fn build<T: cpal::SizedSample + cpal::FromSample<f32>>(
        &self,
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        generation: u64,
    ) -> Result<cpal::Stream> {
        let output = self.clone();
        let error_output = self.clone();
        let channels = config.channels as usize;
        let rate = config.sample_rate.0;
        device
            .build_output_stream(
                config,
                move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
                    let time = info.timestamp();
                    let latency = time
                        .playback
                        .duration_since(&time.callback)
                        .unwrap_or_default();
                    output.render(generation, data, channels, rate, Instant::now() + latency);
                },
                move |error| {
                    if let Ok(_player) = error_output.player.try_lock()
                        && error_output.generation.load(Ordering::SeqCst) == generation
                        && let Ok(mut slot) = error_output.errors.try_lock()
                    {
                        *slot = Some(error.to_string());
                    }
                },
                None,
            )
            .map_err(|e| Error(e.to_string()))
    }
    fn render<T: cpal::SizedSample + cpal::FromSample<f32>>(
        &self,
        generation: u64,
        data: &mut [T],
        channels: usize,
        rate: u32,
        audible_at: Instant,
    ) {
        if let Ok(mut player) = self.player.try_lock() {
            if self.generation.load(Ordering::SeqCst) == generation {
                player.render_audio(data.len() / channels, rate, audible_at, |i, sample| {
                    for c in 0..channels {
                        data[i * channels + c] = T::from_sample(sample);
                    }
                });
                return;
            }
        } else {
            self.contention.fetch_add(1, Ordering::Relaxed);
        }
        data.fill(T::from_sample(0.));
    }
}
