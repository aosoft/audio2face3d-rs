//! Standard CPAL host. Device recreation clears queued audio after pause/seek.
use crate::{
    core::{Error, Result},
    playback::{Command, Player},
};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

pub type SharedPlayer = Arc<Mutex<Player>>;
pub struct AudioOutput {
    pub player: SharedPlayer,
    stream: Option<cpal::Stream>,
    pub callback_contention: Arc<AtomicU64>,
    pub errors: Arc<Mutex<Option<String>>>,
}
impl AudioOutput {
    pub fn new(player: SharedPlayer) -> Self {
        Self {
            player,
            stream: None,
            callback_contention: Arc::new(AtomicU64::new(0)),
            errors: Arc::new(Mutex::new(None)),
        }
    }
    pub fn command(&mut self, command: Command) -> Result<()> {
        if matches!(command, Command::SetLoop(_)) {
            return self.player.lock().unwrap().command(command, Instant::now());
        }
        if let Command::Seek(time) = command
            && (!time.is_finite()
                || time < 0.
                || time > self.player.lock().unwrap().clip.ready_until())
        {
            return Err(Error("seek outside received media".into()));
        }
        // Drop the device stream before resetting the media generation/position.
        self.stream.take();
        if matches!(command, Command::Play)
            && matches!(
                self.player.lock().unwrap().state,
                crate::playback::PlaybackState::Playing | crate::playback::PlaybackState::Buffering
            )
        {
            self.player
                .lock()
                .unwrap()
                .command(Command::Pause, Instant::now())?;
        }
        self.player
            .lock()
            .unwrap()
            .command(command, Instant::now())?;
        if !matches!(command, Command::Pause) {
            self.start()?;
        }
        Ok(())
    }
    pub fn stop(&mut self) {
        self.stream.take();
        let _ = self
            .player
            .lock()
            .unwrap()
            .command(Command::Pause, Instant::now());
    }
    fn start(&mut self) -> Result<()> {
        let result = (|| {
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
                cpal::SampleFormat::F32 => self.build::<f32>(&device, &config.into()),
                cpal::SampleFormat::I16 => self.build::<i16>(&device, &config.into()),
                cpal::SampleFormat::U16 => self.build::<u16>(&device, &config.into()),
                other => return Err(Error(format!("unsupported output format: {other:?}"))),
            }?;
            stream.play().map_err(|e| Error(e.to_string()))?;
            Ok(stream)
        })();
        match result {
            Ok(stream) => {
                self.stream = Some(stream);
                Ok(())
            }
            Err(e) => {
                let _ = self
                    .player
                    .lock()
                    .unwrap()
                    .command(Command::Pause, Instant::now());
                Err(e)
            }
        }
    }
    fn build<T: cpal::SizedSample + cpal::FromSample<f32>>(
        &self,
        device: &cpal::Device,
        config: &cpal::StreamConfig,
    ) -> Result<cpal::Stream> {
        let player = self.player.clone();
        let contention = self.callback_contention.clone();
        let errors = self.errors.clone();
        let channels = config.channels as usize;
        let rate = config.sample_rate.0;
        device
            .build_output_stream(
                config,
                move |data: &mut [T], info: &cpal::OutputCallbackInfo| {
                    let now = Instant::now();
                    let time = info.timestamp();
                    let latency = time
                        .playback
                        .duration_since(&time.callback)
                        .unwrap_or_default();
                    if let Ok(mut player) = player.try_lock() {
                        player.render_audio(
                            data.len() / channels,
                            rate,
                            now + latency,
                            |i, sample| {
                                for c in 0..channels {
                                    data[i * channels + c] = T::from_sample(sample);
                                }
                            },
                        );
                    } else {
                        contention.fetch_add(1, Ordering::Relaxed);
                        data.fill(T::from_sample(0.));
                    }
                },
                move |error| {
                    if let Ok(mut slot) = errors.try_lock() {
                        *slot = Some(error.to_string());
                    }
                },
                None,
            )
            .map_err(|e| Error(e.to_string()))
    }
}
