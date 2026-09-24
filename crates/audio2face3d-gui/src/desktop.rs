use crate::render::{Camera, HeadRenderer, RenderTarget};
use audio2face3d::logging::{LogLevel, LogRecord, Logger};
use std::{collections::BTreeMap, path::Path};

pub struct DesktopApp {
    gpu: egui_wgpu::RenderState,
    renderer: Option<HeadRenderer>,
    target: Option<(RenderTarget, egui::TextureId)>,
    weights: BTreeMap<String, f32>,
    filter: String,
    camera: Camera,
    message: String,
    audio: crate::audio::AudioOutput,
    manual: bool,
    selected: std::collections::BTreeSet<usize>,
    timeline: crate::ui::Timeline,
    logger: std::sync::Arc<crate::logging::GuiLogger>,
    logs: crate::logging::LogBuffer,
    log_view: crate::logging::LogView,
    last_message: String,
    request: crate::inference::Request,
    job: Option<crate::inference::Job>,
    next_job: u64,
    input_finished: bool,
    user_cancelled: bool,
    stream_started: bool,
    #[cfg(feature = "capture")]
    frames: u32,
}
impl DesktopApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        Self::with_options(cc, Default::default())
    }
    pub fn with_options(
        cc: &eframe::CreationContext<'_>,
        options: crate::startup::Options,
    ) -> Self {
        install_host_fonts(&cc.egui_ctx);
        let (logger, logs) = crate::logging::channel(2048, 10_000, LogLevel::Debug);
        let mut app = Self {
            gpu: cc.wgpu_render_state.clone().expect("wgpu host required"),
            renderer: None,
            target: None,
            weights: BTreeMap::new(),
            filter: String::new(),
            camera: Camera::default(),
            message: "Choose a generated head GLB".into(),
            audio: crate::audio::AudioOutput::new(std::sync::Arc::new(std::sync::Mutex::new(
                crate::playback::Player::default(),
            ))),
            manual: true,
            selected: [0, 7, 17].into_iter().collect(),
            timeline: Default::default(),
            logger,
            logs,
            log_view: Default::default(),
            last_message: String::new(),
            request: options.request,
            job: None,
            next_job: 1,
            input_finished: false,
            user_cancelled: false,
            stream_started: false,
            #[cfg(feature = "capture")]
            frames: 0,
        };
        let head = options.head.map(Ok).unwrap_or_else(|| {
            std::env::current_exe().map(|p| p.with_file_name("assets").join("default-head.glb"))
        });
        match head {
            Ok(path) => app.load(&path),
            Err(error) => {
                app.message = format!("Cannot locate standard head: {error}. Use Open head.")
            }
        }
        #[cfg(feature = "capture")]
        if std::env::var_os("A2F_GUI_DEMO").is_some() {
            app.audio
                .player
                .lock()
                .unwrap()
                .replace(crate::core::demo_clip());
            app.manual = false;
            let _ = app.audio.command(crate::playback::Command::Seek(0.5));
        }
        #[cfg(feature = "capture")]
        if let Some(wav) = std::env::var_os("A2F_GUI_WAV") {
            app.request.wav = wav.into();
            app.request.model = std::env::var_os("A2F_GUI_MODEL").unwrap_or_default().into();
            app.request.cuda_root = std::env::var_os("CUDA_PATH").unwrap_or_default().into();
            app.request.tensorrt_root = std::env::var_os("TENSORRT_ROOT_DIR")
                .unwrap_or_default()
                .into();
            app.request.mode = match std::env::var("A2F_GUI_MODE").as_deref() {
                Ok("local") => crate::inference::Mode::Local,
                Ok("mock") => crate::inference::Mode::Mock,
                _ => crate::inference::Mode::Grpc,
            };
            app.request.pace_input = std::env::var_os("A2F_GUI_STREAM").is_some();
            app.start_inference();
        }
        if options.infer && app.job.is_none() {
            app.start_inference();
        }
        app
    }
    fn load(&mut self, path: &Path) {
        let load = (|| -> Result<_, Box<dyn std::error::Error>> {
            if std::fs::metadata(path)?.len()
                > audio2face3d_gui_core::validation::MAX_GLB_BYTES as u64
            {
                return Err("GLB exceeds 64 MiB".into());
            }
            let model = audio2face3d_gui_core::gltf::from_glb(&std::fs::read(path)?)?;
            let renderer = HeadRenderer::new(&self.gpu.device, &model)?;
            Ok((renderer, model))
        })();
        match load {
            Ok((renderer, model)) => {
                self.renderer = Some(renderer);
                self.weights = model.channel_names().into_iter().map(|n| (n, 0.)).collect();
                self.message = format!(
                    "{} | {} | {} channels",
                    path.display(),
                    model.metadata.rig_profile,
                    self.weights.len()
                );
            }
            Err(error) => {
                self.message = format!(
                    "Cannot load head {}: {error}. Use Open head to select a valid GLB.",
                    path.display()
                )
            }
        }
    }
    fn start_inference(&mut self) {
        if self.job.is_some() {
            return;
        }
        match crate::inference::Job::start(self.next_job, self.request.clone(), self.logger.clone())
        {
            Ok(job) => {
                self.audio.stop();
                self.audio
                    .player
                    .lock()
                    .unwrap()
                    .replace(crate::core::Clip::running());
                self.audio.player.lock().unwrap().streaming = self.request.pace_input;
                self.stream_started = false;
                self.job = Some(job);
                self.next_job += 1;
                self.input_finished = false;
                self.user_cancelled = false;
                self.manual = false;
                self.timeline = Default::default();
                self.selected = [0, 7, 17].into_iter().collect();
                self.message = "Inference running".into();
            }
            Err(e) => self.message = e.to_string(),
        }
    }
    fn poll_inference(&mut self) {
        let Some(job) = &mut self.job else {
            return;
        };
        let mut events: Vec<_> = job.events.try_iter().take(256).collect();
        let finished = job.try_finish();
        if finished.is_some() {
            events.extend(job.events.try_iter());
        }
        {
            let mut player = self.audio.player.lock().unwrap();
            for event in events {
                match event {
                    crate::inference::Event::InputFinished => self.input_finished = true,
                    crate::inference::Event::Output(event) => {
                        if matches!(player.clip.session, crate::core::SessionState::Failed(_)) {
                            continue;
                        }
                        let layout =
                            matches!(&event, audio2face3d::types::OutputEvent::StreamInfo(_));
                        if let Err(error) = crate::inference::apply_event(&mut player.clip, event) {
                            player.clip.session =
                                crate::core::SessionState::Failed(error.to_string());
                            self.message = error.to_string();
                            job.cancel();
                        }
                        if layout {
                            self.selected = ["JawOpen", "EyeBlinkLeft", "EyeBlinkRight"]
                                .iter()
                                .filter_map(|name| player.clip.names.iter().position(|n| n == name))
                                .collect();
                        }
                    }
                }
            }
        }
        let should_start = {
            let player = self.audio.player.lock().unwrap();
            player.streaming
                && !self.stream_started
                && !self.user_cancelled
                && (player.clip.ready_until() >= 0.1
                    || (player.clip.session == crate::core::SessionState::Completed
                        && player.clip.ready_until() > 0.))
                && !matches!(
                    player.clip.session,
                    crate::core::SessionState::Failed(_) | crate::core::SessionState::Cancelled
                )
        };
        if should_start {
            self.stream_started = true;
            if let Err(e) = self.audio.command(crate::playback::Command::Play) {
                self.message = e.to_string();
            } else {
                self.logger.log(LogLevel::Info, || {
                    LogRecord::new("Streaming playback started").field("source", "gui.playback")
                });
            }
        }
        if let Some(result) = finished {
            self.job = None;
            let mut player = self.audio.player.lock().unwrap();
            if self.user_cancelled {
                player.clip.session = crate::core::SessionState::Cancelled;
                self.message = "Cancelled; resources released".into();
            } else if let Err(e) = result {
                player.clip.session = crate::core::SessionState::Failed(e.to_string());
                self.message = e.to_string();
            } else if player.clip.session == crate::core::SessionState::Completed {
                self.message = "Results complete; resources released; ready to play".into();
            } else if !matches!(player.clip.session, crate::core::SessionState::Failed(_)) {
                player.clip.session =
                    crate::core::SessionState::Failed("missing completion".into());
            }
            self.logger.write_log(
                if matches!(player.clip.session, crate::core::SessionState::Failed(_)) {
                    LogLevel::Error
                } else {
                    LogLevel::Info
                },
                LogRecord::new(&self.message).field("source", "gui.inference"),
            );
            let failed = matches!(
                player.clip.session,
                crate::core::SessionState::Failed(_) | crate::core::SessionState::Cancelled
            );
            drop(player);
            if failed {
                self.audio.stop();
            }
        }
    }
}
impl eframe::App for DesktopApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll_inference();
        if self.job.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        if self.message != self.last_message {
            self.logger.log(LogLevel::Info, || {
                LogRecord::new(self.message.clone()).field("source", "gui")
            });
            self.last_message = self.message.clone();
        }
        self.logs.drain();
        #[cfg(feature = "capture")]
        if let Some(path) = std::env::var_os("A2F_GUI_CAPTURE") {
            let streaming_capture = std::env::var_os("A2F_GUI_STREAM").is_some()
                && self
                    .audio
                    .player
                    .lock()
                    .unwrap()
                    .audible_position(std::time::Instant::now())
                    > 1.;
            if self.job.is_none() || streaming_capture {
                self.frames += 1;
            }
            if self.frames == 1 && std::env::var_os("A2F_GUI_WAV").is_some() && !streaming_capture {
                let _ = self.audio.command(crate::playback::Command::Seek(0.5));
            }
            if self.frames == 5 {
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            }
            for event in ctx.input(|i| i.events.clone()) {
                if let egui::Event::Screenshot { image, .. } = event {
                    let bytes: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
                    if let Err(e) = image::save_buffer(
                        Path::new(&path),
                        &bytes,
                        image.width() as u32,
                        image.height() as u32,
                        image::ColorType::Rgba8,
                    ) {
                        eprintln!("capture failed: {e}");
                    }
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
            ctx.request_repaint();
        }
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Audio2Face-3D");
                if ui
                    .add_enabled(self.job.is_none(), egui::Button::new("Sync demo"))
                    .clicked()
                {
                    self.audio.stop();
                    self.audio
                        .player
                        .lock()
                        .unwrap()
                        .replace(crate::core::demo_clip());
                    self.manual = false;
                    self.timeline = Default::default();
                    if let Err(e) = self.audio.command(crate::playback::Command::Play) {
                        self.message = e.to_string();
                    }
                }
                if ui.checkbox(&mut self.manual, "Manual").changed() && self.manual {
                    self.stream_started = true;
                    self.audio.stop();
                }
                if ui.button("Open head").clicked()
                    && let Some(path) = rfd::FileDialog::new()
                        .add_filter("Head", &["glb"])
                        .pick_file()
                {
                    self.load(&path);
                }
                if ui.button("Front").clicked() {
                    self.camera = Camera::default();
                }
                if ui.button("Side").clicked() {
                    self.camera.yaw = std::f32::consts::FRAC_PI_2;
                    self.camera.pitch = 0.;
                }
                if ui.button("Three-quarter").clicked() {
                    self.camera.yaw = 0.65;
                    self.camera.pitch = 0.1;
                }
            });
            ui.label(&self.message);
            egui::CollapsingHeader::new("Inference")
                .default_open(true)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("WAV");
                        path_edit(ui, &mut self.request.wav);
                        if ui.button("Browse WAV").clicked()
                            && let Some(path) = rfd::FileDialog::new()
                                .add_filter("WAV", &["wav"])
                                .pick_file()
                        {
                            self.request.wav = path;
                        }
                        ui.add_enabled_ui(self.job.is_none(), |ui| {
                            egui::ComboBox::from_id_salt("inference-mode")
                                .selected_text(format!("{:?}", self.request.mode))
                                .show_ui(ui, |ui| {
                                    #[cfg(feature = "local")]
                                    ui.selectable_value(
                                        &mut self.request.mode,
                                        crate::inference::Mode::Local,
                                        "Local",
                                    );
                                    #[cfg(feature = "grpc")]
                                    ui.selectable_value(
                                        &mut self.request.mode,
                                        crate::inference::Mode::Grpc,
                                        "gRPC",
                                    );
                                    #[cfg(feature = "mock")]
                                    ui.selectable_value(
                                        &mut self.request.mode,
                                        crate::inference::Mode::Mock,
                                        "Mock (development)",
                                    );
                                    let _ = ui;
                                });
                            if ui.button("Infer WAV").clicked() {
                                self.start_inference();
                            }
                        });
                        if ui
                            .add_enabled(self.job.is_some(), egui::Button::new("Cancel"))
                            .clicked()
                            && let Some(job) = &self.job
                        {
                            job.cancel();
                            self.user_cancelled = true;
                            self.audio.stop();
                        }
                    });
                    match self.request.mode {
                        crate::inference::Mode::Grpc => {
                            ui.horizontal(|ui| {
                                ui.label("Endpoint");
                                ui.text_edit_singleline(&mut self.request.endpoint);
                                ui.label("API key");
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.request.api_key)
                                        .password(true),
                                );
                            });
                        }
                        crate::inference::Mode::Local => {
                            ui.horizontal(|ui| {
                                ui.label("Model JSON (required)");
                                path_edit(ui, &mut self.request.model);
                                if ui.button("Browse model").clicked()
                                    && let Some(path) = rfd::FileDialog::new()
                                        .add_filter("Model", &["json"])
                                        .pick_file()
                                {
                                    self.request.model = path;
                                }
                                ui.label("GPU");
                                ui.add(
                                    egui::DragValue::new(&mut self.request.device).range(0..=15),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label("CUDA root override");
                                path_edit(ui, &mut self.request.cuda_root);
                                ui.label("TensorRT root override");
                                path_edit(ui, &mut self.request.tensorrt_root);
                            });
                            let runtime = &self.request.runtime;
                            let describe = |root: Option<&Path>, dirs: &[std::path::PathBuf]| {
                                root.map(|p| p.display().to_string()).unwrap_or_else(|| {
                                    if dirs.is_empty() { "automatic discovery".into() }
                                    else { dirs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ") }
                                })
                            };
                            ui.label(format!("Startup runtime: CUDA {} | TensorRT {} | {:?}. Blank overrides keep these settings.",
                                describe(runtime.cuda_root(), runtime.cuda_library_dirs()),
                                describe(runtime.tensorrt_root(), runtime.tensorrt_library_dirs()), runtime.search_policy()));
                        }
                        _ => {}
                    }
                    let player = self.audio.player.lock().unwrap();
                    ui.label(format!(
                        "Input sent: {} | Result: {:?} | Resources released: {}",
                        self.input_finished,
                        player.clip.session,
                        self.job.is_none()
                    ));
                    drop(player);
                    ui.add_enabled_ui(self.job.is_none(), |ui| {
                        ui.checkbox(
                            &mut self.request.pace_input,
                            "Play while inferring (paced input, 100 ms buffer)",
                        );
                    });
                });
        });
        egui::TopBottomPanel::bottom("logs").show(ctx, |ui| {
            egui::CollapsingHeader::new(format!("Logs ({})", self.logs.entries.len()))
                .show(ui, |ui| self.log_view.show(ui, &self.logs));
        });
        let shared = self.audio.player.clone();
        let names = shared.lock().unwrap().clip.names.clone();
        let mut snapshot = shared.lock().unwrap().snapshot(std::time::Instant::now());
        self.selected.retain(|i| *i < snapshot.values.len());
        let mut command = None;
        if !self.manual {
            egui::TopBottomPanel::bottom("playback")
                .resizable(true)
                .default_height(370.)
                .min_height(270.)
                .show(ctx, |ui| {
                    if let Some(time) = crate::ui::playback_seek_bar(ui, &snapshot) {
                        command = Some(crate::playback::Command::Seek(time));
                        // Let the viewport follow this target in the same frame.
                        self.timeline.reveal_position(time, snapshot.duration);
                        snapshot.time = time;
                    }
                    if let Some(time) = self.timeline.show_player_with_controls(
                        ui,
                        &shared,
                        &snapshot,
                        &self.selected,
                        |ui| {
                            let playing = matches!(
                                snapshot.state,
                                crate::playback::PlaybackState::Playing
                                    | crate::playback::PlaybackState::Buffering
                            );
                            if ui
                                .add_sized(
                                    [60., ui.spacing().interact_size.y],
                                    egui::Button::new(if playing { "Pause" } else { "Play" }),
                                )
                                .clicked()
                            {
                                command = Some(if playing {
                                    crate::playback::Command::Pause
                                } else {
                                    crate::playback::Command::Play
                                });
                            }
                            let mut looping = snapshot.looping;
                            if ui.checkbox(&mut looping, "Loop").changed() {
                                command = Some(crate::playback::Command::SetLoop(looping));
                            }
                            ui.label(format!("{:.3} / {:.3} s", snapshot.time, snapshot.duration));
                            ui.label(format!(
                            "{:?} | received {:.2}s | buffer {:.2}s | underruns {} | audio busy {}",
                            snapshot.state,
                            snapshot.duration,
                            (snapshot.ready_until - snapshot.time).max(0.),
                            snapshot.underruns,
                            self.audio
                                .callback_contention
                                .load(std::sync::atomic::Ordering::Relaxed)
                        ));
                        },
                    ) {
                        command = Some(crate::playback::Command::Seek(time));
                    }
                });
        }
        if let Some(command) = command {
            if matches!(
                command,
                crate::playback::Command::Pause | crate::playback::Command::Play
            ) {
                self.stream_started = true;
            }
            if let Err(e) = self.audio.command(command) {
                self.message = e.to_string();
            }
            snapshot = shared.lock().unwrap().snapshot(std::time::Instant::now());
            ctx.request_repaint();
        }
        if !self.manual {
            self.weights.values_mut().for_each(|v| *v = 0.);
            for (name, &value) in names.iter().zip(&snapshot.values) {
                if let Some(weight) = self.weights.get_mut(name) {
                    *weight = value;
                }
            }
        }
        if matches!(
            snapshot.state,
            crate::playback::PlaybackState::Playing | crate::playback::PlaybackState::Buffering
        ) {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
        if let Some(error) = self.audio.errors.lock().unwrap().take() {
            self.message = error;
        }
        egui::SidePanel::left("head-preview")
            .default_width(260.)
            .width_range(180.0..=360.0)
            .show(ctx, |ui| {
                let available = ui.available_size().max(egui::vec2(1., 1.));
                // Keep a portrait viewport even when no timeline is visible.
                let size = egui::vec2(available.x, available.y.min(available.x * 1.35));
                let pixels = [
                    (size.x * ctx.pixels_per_point()) as u32,
                    (size.y * ctx.pixels_per_point()) as u32,
                ]
                .map(|v| v.clamp(1, 4096));
                if self.target.as_ref().is_none_or(|(t, _)| t.size != pixels) {
                    if let Some((_, id)) = self.target.take() {
                        self.gpu.renderer.write().free_texture(&id);
                    }
                    let target = RenderTarget::new(&self.gpu.device, pixels);
                    let id = self.gpu.renderer.write().register_native_texture(
                        &self.gpu.device,
                        &target.view,
                        wgpu::FilterMode::Linear,
                    );
                    self.target = Some((target, id));
                }
                let (target, id) = self.target.as_ref().unwrap();
                if let Some(renderer) = &self.renderer {
                    let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
                    match renderer.render(
                        &self.gpu.queue,
                        &mut encoder,
                        target,
                        self.camera,
                        &self.weights,
                    ) {
                        Ok(()) => {
                            self.gpu.queue.submit([encoder.finish()]);
                        }
                        Err(e) => self.message = e.to_string(),
                    }
                    let response = ui.add(egui::Image::new((*id, size)).sense(egui::Sense::drag()));
                    if response.dragged() {
                        let d = ctx.input(|i| i.pointer.delta());
                        self.camera.yaw -= d.x * 0.008;
                        self.camera.pitch = (self.camera.pitch + d.y * 0.008).clamp(-1.3, 1.3);
                    }
                    if response.hovered() {
                        self.camera.distance = (self.camera.distance
                            * ctx.input(|i| (-i.smooth_scroll_delta.y * 0.002).exp()))
                        .clamp(0.18, 2.);
                    }
                } else {
                    ui.centered_and_justified(|ui| {
                        ui.label("Open a generated GLB to inspect its morph channels");
                    });
                }
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.manual {
                crate::ui::manual_channels(ui, &mut self.weights, &mut self.filter);
            } else {
                crate::ui::channel_values_for_names(
                    ui,
                    &names,
                    &snapshot,
                    &mut self.selected,
                    &mut self.filter,
                );
            }
        });
    }
}

fn path_edit(ui: &mut egui::Ui, path: &mut std::path::PathBuf) {
    let mut text = path.display().to_string();
    if ui.text_edit_singleline(&mut text).changed() {
        *path = text.into();
    }
}
fn install_host_fonts(ctx: &egui::Context) {
    // Installed OS fonts cover Japanese paths/errors; binaries are not redistributed.
    #[cfg(target_os = "windows")]
    if let Some(windows) = std::env::var_os("WINDIR") {
        for file in ["meiryo.ttc", "YuGothM.ttc"] {
            let path = std::path::PathBuf::from(&windows).join("Fonts").join(file);
            if !std::fs::metadata(&path).is_ok_and(|m| m.len() <= 32 * 1024 * 1024) {
                continue;
            }
            if let Ok(data) = std::fs::read(path) {
                let mut fonts = egui::FontDefinitions::default();
                fonts.font_data.insert(
                    "host-japanese".into(),
                    egui::FontData::from_owned(data).into(),
                );
                for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
                    fonts
                        .families
                        .entry(family)
                        .or_default()
                        .push("host-japanese".into());
                }
                ctx.set_fonts(fonts);
                break;
            }
        }
    }
    let _ = ctx;
}

impl Drop for DesktopApp {
    fn drop(&mut self) {
        self.audio.stop();
        self.job.take();
        self.logs.drain();
    }
}

/// Start the standard host. Embedding callers use library modules instead.
pub fn run() -> eframe::Result {
    run_with_options(Default::default())
}
pub fn run_with_options(options: crate::startup::Options) -> eframe::Result {
    eframe::run_native(
        "Audio2Face-3D",
        eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            viewport: egui::ViewportBuilder::default().with_inner_size([1440., 900.]),
            ..Default::default()
        },
        Box::new(move |cc| Ok(Box::new(DesktopApp::with_options(cc, options)))),
    )
}
