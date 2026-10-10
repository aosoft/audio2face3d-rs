//! CPAL streams and device failure transitions are owned by a worker.
use crate::{
    core::{Error, Result},
    playback::{Command, PlaybackState, Player},
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU8, AtomicU64, Ordering},
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
type Requests = Arc<Mutex<DeviceRequest>>;

const HEALTHY: u8 = 0;
const WRITING_FAULT: u8 = 1;
const FAULT_READY: u8 = 2;
const FAULT_HANDLED: u8 = 3;

struct StreamEpoch {
    generation: u64,
    state: AtomicU8,
    first_fault: OnceLock<cpal::StreamError>,
    repeated_faults: AtomicU64,
    worker_thread: std::thread::Thread,
}
impl StreamEpoch {
    fn new(generation: u64) -> Self {
        Self {
            generation,
            state: AtomicU8::new(HEALTHY),
            first_fault: OnceLock::new(),
            repeated_faults: AtomicU64::new(0),
            worker_thread: std::thread::current(),
        }
    }
    fn report_fault(&self, error: cpal::StreamError) {
        self.publish_fault(error, || {});
    }
    fn publish_fault(&self, error: cpal::StreamError, before_ready: impl FnOnce()) {
        if self
            .state
            .compare_exchange(HEALTHY, WRITING_FAULT, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            // Only the writer owning WRITING_FAULT touches this OnceLock.
            let stored = self.first_fault.set(error);
            debug_assert!(stored.is_ok());
            before_ready();
            self.state.store(FAULT_READY, Ordering::Release);
        } else {
            self.repeated_faults.fetch_add(1, Ordering::Relaxed);
        }
        self.worker_thread.unpark();
    }
    fn faulted(&self) -> bool {
        self.state.load(Ordering::Acquire) != HEALTHY
    }
    fn take_fault(&self) -> Option<String> {
        if self
            .state
            .compare_exchange(
                FAULT_READY,
                FAULT_HANDLED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return None;
        }
        // String conversion happens only on the worker, after READY publication.
        Some(
            self.first_fault
                .get()
                .expect("fault published before ready")
                .to_string(),
        )
    }
}
struct DeviceFailure {
    generation: u64,
    message: String,
}

pub struct AudioOutput {
    pub player: SharedPlayer,
    generation: Arc<AtomicU64>,
    requests: Requests,
    worker: Option<std::thread::JoinHandle<()>>,
    pub callback_contention: Arc<AtomicU64>,
    /// Compatibility view; use take_error to exclude failures from older commands.
    pub errors: Arc<Mutex<Option<String>>>,
    failure: Arc<Mutex<Option<DeviceFailure>>>,
}
impl AudioOutput {
    pub fn new(player: SharedPlayer) -> Self {
        Self::with_stream_factory(player, Device::start)
    }
    fn with_stream_factory<S: 'static>(
        player: SharedPlayer,
        start: impl FnMut(&Device, u64) -> Result<S> + Send + 'static,
    ) -> Self {
        Self::with_stream_factory_and_before_wait(player, start, || {})
    }
    fn with_stream_factory_and_before_wait<S: 'static>(
        player: SharedPlayer,
        start: impl FnMut(&Device, u64) -> Result<S> + Send + 'static,
        before_wait: impl FnMut() + Send + 'static,
    ) -> Self {
        let generation = Arc::new(AtomicU64::new(0));
        let requests = Arc::new(Mutex::new(DeviceRequest::default()));
        let callback_contention = Arc::new(AtomicU64::new(0));
        let errors = Arc::new(Mutex::new(None));
        let failure = Arc::new(Mutex::new(None));
        let state = DeviceState {
            player: player.clone(),
            generation: generation.clone(),
            contention: callback_contention.clone(),
            errors: errors.clone(),
            failure: failure.clone(),
        };
        let worker_requests = requests.clone();
        let worker = std::thread::spawn(move || state.run(worker_requests, start, before_wait));
        Self {
            player,
            generation,
            requests,
            worker: Some(worker),
            callback_contention,
            errors,
            failure,
        }
    }
    /// Updates media position synchronously; the worker handles device failures.
    pub fn command(&mut self, command: Command) -> Result<()> {
        let mut player = self.player.lock().unwrap();
        if matches!(command, Command::SetLoop(_)) {
            return player.command(command, Instant::now());
        }
        if matches!(command, Command::Play)
            && matches!(
                player.state(),
                PlaybackState::Playing | PlaybackState::Buffering
            )
        {
            player.command(Command::Pause, Instant::now())?;
        }
        player.command(command, Instant::now())?;
        // Generation and position share the same exclusion boundary as failure handling.
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        let running = matches!(
            player.state(),
            PlaybackState::Playing | PlaybackState::Buffering
        );
        *self.failure.lock().unwrap() = None;
        *self.errors.lock().unwrap() = None;
        drop(player);
        *self.requests.lock().unwrap() = DeviceRequest {
            generation,
            running,
            shutdown: false,
        };
        self.wake_worker();
        Ok(())
    }
    fn wake_worker(&self) {
        if let Some(worker) = &self.worker {
            worker.thread().unpark();
        }
    }
    pub fn stop(&mut self) {
        let _ = self.command(Command::Pause);
    }
    /// Consume only the current generation's failure, ordered with media commands.
    pub fn take_error(&self) -> Option<String> {
        let _player = self.player.lock().unwrap();
        let generation = self.generation.load(Ordering::SeqCst);
        let failure = self.failure.lock().unwrap().take();
        self.errors.lock().unwrap().take();
        failure
            .filter(|failure| failure.generation == generation)
            .map(|failure| failure.message)
    }
}
impl Drop for AudioOutput {
    fn drop(&mut self) {
        self.stop();
        self.requests.lock().unwrap().shutdown = true;
        self.wake_worker();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[derive(Clone)]
struct DeviceState {
    player: SharedPlayer,
    generation: Arc<AtomicU64>,
    contention: Arc<AtomicU64>,
    errors: Arc<Mutex<Option<String>>>,
    failure: Arc<Mutex<Option<DeviceFailure>>>,
}
struct ActiveStream<S> {
    _stream: S,
    epoch: Arc<StreamEpoch>,
}
impl DeviceState {
    fn handle_failure<S>(
        &self,
        generation: u64,
        message: String,
        active: &mut Option<ActiveStream<S>>,
    ) -> bool {
        {
            let mut player = self.player.lock().unwrap();
            // Recheck after obtaining the same lock used by command, not merely
            // when READY was first observed. Obsolete failures cannot drop a new stream.
            if self.generation.load(Ordering::SeqCst) != generation {
                return false;
            }
            let _ = player.command(Command::Pause, Instant::now());
            *self.errors.lock().unwrap() = Some(message.clone());
            *self.failure.lock().unwrap() = Some(DeviceFailure {
                generation,
                message,
            });
        }
        // OS stream destruction is on this worker, outside the player lock.
        drop(active.take());
        true
    }
    fn run<S>(
        &self,
        requests: Requests,
        mut start: impl FnMut(&Device, u64) -> Result<S>,
        mut before_wait: impl FnMut(),
    ) {
        let mut active = None;
        let mut seen = 0;
        loop {
            let request = *requests.lock().unwrap();
            if request.shutdown {
                drop(active.take());
                break;
            }
            if request.generation != seen {
                drop(active.take());
                seen = request.generation;
                if request.running && self.generation.load(Ordering::SeqCst) == seen {
                    let epoch = Arc::new(StreamEpoch::new(seen));
                    let device = Device {
                        state: self.clone(),
                        epoch: epoch.clone(),
                    };
                    match start(&device, seen) {
                        Ok(created) if self.generation.load(Ordering::SeqCst) == seen => {
                            active = Some(ActiveStream {
                                _stream: created,
                                epoch,
                            });
                        }
                        Ok(_) => {} // Superseded while opening; drop only the obsolete candidate.
                        Err(error) => {
                            let message = epoch.take_fault().unwrap_or_else(|| error.to_string());
                            self.handle_failure(seen, message, &mut active);
                        }
                    }
                }
            }
            // Also inspect faults immediately after start returns, before parking.
            let fault = active.as_ref().and_then(|stream| {
                stream
                    .epoch
                    .take_fault()
                    .map(|message| (stream.epoch.generation, message))
            });
            if let Some((generation, message)) = fault {
                self.handle_failure(generation, message, &mut active);
            }
            before_wait();
            // Device setup/destruction or a blocking library may have consumed
            // a park token. Recheck predicates after all such work and test gates.
            let pending = *requests.lock().unwrap();
            let ready_fault = active
                .as_ref()
                .is_some_and(|stream| stream.epoch.state.load(Ordering::Acquire) == FAULT_READY);
            if pending.shutdown || pending.generation != seen || ready_fault {
                continue;
            }
            // No intervening blocking work: a later unpark retains its token.
            std::thread::park();
        }
    }
}

#[derive(Clone)]
struct Device {
    state: DeviceState,
    epoch: Arc<StreamEpoch>,
}
impl Device {
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
        if self.state.generation.load(Ordering::SeqCst) == generation && !self.epoch.faulted() {
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
        let errors = self.epoch.clone();
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
                    errors.report_fault(error);
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
        if self.epoch.faulted()
            || self.epoch.generation != generation
            || self.state.generation.load(Ordering::SeqCst) != generation
        {
            data.fill(T::from_sample(0.));
            return;
        }
        if let Ok(mut player) = self.state.player.try_lock() {
            if self.state.generation.load(Ordering::SeqCst) == generation && !self.epoch.faulted() {
                player.render_audio(data.len() / channels, rate, audible_at, |i, sample| {
                    for c in 0..channels {
                        data[i * channels + c] = T::from_sample(sample);
                    }
                });
                return;
            }
        } else {
            self.state.contention.fetch_add(1, Ordering::Relaxed);
        }
        data.fill(T::from_sample(0.));
    }
}
