use crate::render::{Camera, HeadRenderer, RenderTarget};
use audio2face3d::logging::{LogLevel, LogRecord, Logger};
use std::{collections::BTreeMap, path::Path};
mod transport;
use transport::{Action, Transport};

pub struct DesktopApp {
    gpu: egui_wgpu::RenderState,
    renderer: Option<HeadRenderer>,
    target: Option<(RenderTarget, egui::TextureId)>,
    weights: BTreeMap<String, f32>,
    filter: String,
    camera: Camera,
    message: String,
    head_info: String,
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
    transport: Transport,
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
            weights: inference_channel_weights(),
            filter: String::new(),
            camera: Camera::default(),
            message: "Choose a head GLB".into(),
            head_info: String::new(),
            audio: crate::audio::AudioOutput::new(std::sync::Arc::new(std::sync::Mutex::new(
                crate::playback::Player::default(),
            ))),
            manual: false,
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
            transport: Transport::default(),
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
            app.transport.finished(true, false);
            let _ = app.audio.command(crate::playback::Command::Seek(0.5));
        }
        #[cfg(feature = "capture")]
        if let Some(wav) = std::env::var_os("A2F_GUI_WAV") {
            app.request.wav = wav.into();
            app.request.model = std::env::var_os("A2F_GUI_MODEL").unwrap_or_default().into();
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
                self.head_info = head_summary(&model);
                self.renderer = Some(renderer);
                // Channel controls belong to inference, not to the loaded head.
                // The renderer maps supported targets by name and ignores the rest.
                self.message = format!(
                    "{} | {} | {} channels",
                    path.display(),
                    model.metadata.rig_profile,
                    model.channel_names().len()
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
                self.transport.invalidate();
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
                self.message = "Results complete; resources released".into();
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
            if self.transport.finished(!failed, self.request.pace_input) {
                self.stream_started = true;
                if let Err(e) = self.audio.command(crate::playback::Command::Play) {
                    self.message = e.to_string();
                }
            }
        }
    }

    fn stop_inference(&mut self) {
        if let Some(job) = &self.job {
            job.cancel();
        }
        self.user_cancelled = true;
        self.stream_started = true;
        self.transport.invalidate();
        self.audio.stop();
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
        let playback_state = self
            .audio
            .player
            .lock()
            .unwrap()
            .snapshot(std::time::Instant::now())
            .state;
        let settings_editable = self
            .transport
            .action(self.request.pace_input, self.job.is_some(), playback_state)
            .editable();
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Audio2Face-3D");
                if ui
                    .add_enabled(settings_editable, egui::Button::new("Sync demo"))
                    .clicked()
                {
                    self.audio.stop();
                    self.audio
                        .player
                        .lock()
                        .unwrap()
                        .replace(crate::core::demo_clip());
                    self.manual = false;
                    self.request.pace_input = false;
                    self.transport.finished(true, false);
                    self.timeline = Default::default();
                    if let Err(e) = self.audio.command(crate::playback::Command::Play) {
                        self.message = e.to_string();
                    }
                }
                if ui
                    .add_enabled(
                        settings_editable,
                        egui::Checkbox::new(&mut self.manual, "Manual"),
                    )
                    .changed()
                    && self.manual
                {
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
            if !self.head_info.is_empty() {
                ui.label(&self.head_info);
            }
        });
        egui::TopBottomPanel::bottom("logs").show(ctx, |ui| {
            egui::CollapsingHeader::new(format!("Logs ({})", self.logs.entries.len()))
                .show(ui, |ui| self.log_view.show(ui, &self.logs));
        });
        let shared = self.audio.player.clone();
        let (mut names, mut snapshot) = playback_view(&shared);
        self.selected
            .retain(|i| *i < audio2face3d::types::CURVE_NAMES.len());
        let mut command = None;
        let action =
            self.transport
                .action(self.request.pace_input, self.job.is_some(), snapshot.state);
        let can_seek = self
            .transport
            .can_seek(self.request.pace_input, self.job.is_some());
        let mut activate = false;
        self.timeline.seek_disabled = !can_seek;
        if !self.manual {
            egui::TopBottomPanel::bottom("playback")
                .resizable(true)
                .default_height(370.)
                .min_height(270.)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .add_sized(
                                [130., ui.spacing().interact_size.y],
                                egui::Button::new(action.label()),
                            )
                            .on_hover_text(match action {
                                Action::Loading => "Inference in progress. Click to cancel.",
                                _ => action.label(),
                            })
                            .clicked()
                        {
                            activate = true;
                        }
                        let mut looping = snapshot.looping;
                        if ui
                            .add_enabled(can_seek, egui::Checkbox::new(&mut looping, "Loop"))
                            .changed()
                        {
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
                    });
                    if let Some(time) = ui
                        .add_enabled_ui(can_seek, |ui| crate::ui::playback_seek_bar(ui, &snapshot))
                        .inner
                    {
                        command = Some(crate::playback::Command::Seek(time));
                        // Let the viewport follow this target in the same frame.
                        self.timeline.reveal_position(time, snapshot.duration);
                        snapshot.time = time;
                    }
                    if let Some(time) = self.timeline.show_player(
                        ui,
                        &shared,
                        &snapshot,
                        &timeline_selection(&names, &self.selected),
                    ) {
                        command = Some(crate::playback::Command::Seek(time));
                    }
                });
        }
        if activate {
            match action {
                Action::Initialize => {
                    self.start_inference();
                    command = None;
                }
                Action::Loading | Action::Stop => {
                    self.stop_inference();
                    command = None;
                }
                Action::Play => command = Some(crate::playback::Command::Play),
                Action::Pause => command = Some(crate::playback::Command::Pause),
            }
            (names, snapshot) = playback_view(&shared);
            ctx.request_repaint();
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
            (names, snapshot) = playback_view(&shared);
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
        let settings_editable = self
            .transport
            .action(self.request.pace_input, self.job.is_some(), snapshot.state)
            .editable();
        egui::TopBottomPanel::bottom("inference").show(ctx, |ui| {
            let previous = self.request.clone();
            egui::CollapsingHeader::new("Inference")
                .default_open(true)
                .show(ui, |ui| {
                    ui.add_enabled_ui(settings_editable, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("WAV");
                            path_edit(ui, &mut self.request.wav);
                            if ui.button("Browse").clicked()
                                && let Some(path) = rfd::FileDialog::new()
                                    .add_filter("WAV", &["wav"])
                                    .pick_file()
                            {
                                self.request.wav = path;
                            }
                        });
                        ui.checkbox(
                            &mut self.request.pace_input,
                            "Play while inferring (paced input, 100 ms buffer)",
                        );
                        ui.add_space(ui.spacing().interact_size.y);
                        ui.horizontal(|ui| {
                            ui.label("Mode");
                            egui::ComboBox::from_id_salt("inference-mode")
                                .selected_text(match self.request.mode {
                                    crate::inference::Mode::Local => "Local",
                                    crate::inference::Mode::Grpc => "gRPC",
                                    crate::inference::Mode::Mock => "Mock",
                                })
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
                        });
                        ui.group(|ui| match self.request.mode {
                            crate::inference::Mode::Local => {
                                ui.horizontal(|ui| {
                                    ui.label("Local");
                                    ui.label("Model JSON");
                                    path_edit(ui, &mut self.request.model);
                                    if ui.button("Browse").clicked()
                                        && let Some(path) = rfd::FileDialog::new()
                                            .add_filter("Model", &["json"])
                                            .pick_file()
                                    {
                                        self.request.model = path;
                                    }
                                });
                            }
                            crate::inference::Mode::Grpc => {
                                ui.horizontal(|ui| {
                                    ui.label("gRPC");
                                    ui.label("End Point");
                                    ui.text_edit_singleline(&mut self.request.endpoint);
                                    ui.label("API Key");
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.request.api_key)
                                            .password(true),
                                    );
                                });
                            }
                            crate::inference::Mode::Mock => {
                                ui.label("Mock (development)");
                            }
                        });
                    });
                    let player = self.audio.player.lock().unwrap();
                    ui.label(format!(
                        "Input sent: {} | Result: {:?} | Resources released: {}",
                        self.input_finished,
                        player.clip.session,
                        self.job.is_none()
                    ));
                });
            if transport::inputs_changed(&previous, &self.request) {
                self.transport.invalidate();
                ctx.request_repaint();
            }
        });
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
                let (channel_names, channel_snapshot) = inference_channel_view(&names, &snapshot);
                crate::ui::channel_values_for_names(
                    ui,
                    &channel_names,
                    &channel_snapshot,
                    &mut self.selected,
                    &mut self.filter,
                );
                if names.is_empty() {
                    ui.label("Awaiting inference data");
                }
            }
        });
    }
}

/// Capture names and values from the same clip, including after a restart in this frame.
fn playback_view(
    player: &std::sync::Mutex<crate::playback::Player>,
) -> (Vec<String>, crate::playback::Snapshot) {
    let mut player = player.lock().unwrap();
    (
        player.clip.names.clone(),
        player.snapshot(std::time::Instant::now()),
    )
}

fn path_edit(ui: &mut egui::Ui, path: &mut std::path::PathBuf) {
    let mut text = path.display().to_string();
    let width = (ui.available_width() - 90.).clamp(80., 560.);
    if ui
        .add(egui::TextEdit::singleline(&mut text).desired_width(width))
        .changed()
    {
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

/// UI rows always use canonical inference order. Missing samples display zero;
/// this does not create frames or make an empty clip playable.
fn inference_channel_view(
    names: &[String],
    snapshot: &crate::playback::Snapshot,
) -> (Vec<String>, crate::playback::Snapshot) {
    let canonical = audio2face3d::types::CURVE_NAMES;
    let indices = canonical.map(|name| names.iter().position(|n| n == name));
    let values = |source: &[f32]| {
        indices
            .iter()
            .map(|i| i.and_then(|i| source.get(i)).copied().unwrap_or(0.))
            .collect()
    };
    (
        canonical.into_iter().map(str::to_owned).collect(),
        crate::playback::Snapshot {
            time: snapshot.time,
            duration: snapshot.duration,
            ready_until: snapshot.ready_until,
            state: snapshot.state,
            values: values(&snapshot.values),
            raw_values: values(&snapshot.raw_values),
            underruns: snapshot.underruns,
            looping: snapshot.looping,
        },
    )
}

/// Timeline tracks retain received-layout indices; selections belong to UI names.
fn timeline_selection(
    names: &[String],
    selected: &std::collections::BTreeSet<usize>,
) -> std::collections::BTreeSet<usize> {
    let canonical = audio2face3d::types::CURVE_NAMES;
    names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            canonical
                .iter()
                .position(|n| *n == name)
                .filter(|index| selected.contains(index))
                .map(|_| i)
        })
        .collect()
}

fn inference_channel_weights() -> BTreeMap<String, f32> {
    audio2face3d::types::CURVE_NAMES
        .into_iter()
        .map(|name| (name.to_owned(), 0.))
        .collect()
}

fn head_summary(model: &audio2face3d_gui_core::HeadModel) -> String {
    let unsupported = model.unsupported_channels();
    if unsupported.is_empty() {
        "Head model: all 52 channels supported".into()
    } else {
        format!(
            "Head model: {} channels supported; unsupported: {}",
            model.channel_names().len(),
            unsupported.join(", ")
        )
    }
}
#[cfg(test)]
mod head_tests {
    use super::*;
    #[test]
    fn channels_exist_before_initialization_and_received_order_does_not_change_selection() {
        let mut player = crate::playback::Player::default();
        let mut snapshot = player.snapshot(std::time::Instant::now());
        let (names, view) = inference_channel_view(&[], &snapshot);
        assert_eq!(names.len(), 52);
        assert_eq!(names[51], "TongueOut");
        assert_eq!(view.values, vec![0.; 52]);
        assert!(player.clip.names.is_empty());
        assert!(player.clip.frames.is_empty());
        snapshot.values = vec![0.7, 0.2, 0.9];
        snapshot.raw_values = vec![0.8, 0.3, 1.];
        let received = vec![
            "TongueOut".into(),
            "JawOpen".into(),
            "NotAnInferenceChannel".into(),
        ];
        let (_, view) = inference_channel_view(&received, &snapshot);
        assert_eq!(view.values[51], 0.7);
        assert_eq!(view.raw_values[17], 0.3);
        assert_eq!(view.values[0], 0.);
        let selected = [17, 51].into_iter().collect();
        assert_eq!(
            timeline_selection(&received, &selected),
            [0, 1].into_iter().collect()
        );
    }

    #[test]
    fn manual_channels_follow_inference_even_when_the_head_omits_tongue() {
        let mut model =
            audio2face3d_gui_core::gltf::from_glb(include_bytes!("../assets/default-head.glb"))
                .unwrap();
        for mesh in &mut model.meshes {
            mesh.targets.retain(|t| t.name != "TongueOut");
        }
        model.validate().unwrap();
        let mut weights = inference_channel_weights();
        assert_eq!(weights.len(), audio2face3d::types::CURVE_NAMES.len());
        assert!(weights.contains_key("TongueOut"));
        assert!(model.unsupported_channels().contains(&"TongueOut".into()));
        weights.insert("TongueOut".into(), 0.75);
        let before = weights.clone();
        let _ = head_summary(&model);
        assert_eq!(weights, before);
        assert_eq!(
            weights
                .keys()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>(),
            audio2face3d::types::CURVE_NAMES.into_iter().collect()
        );
    }

    #[test]
    fn subset_summary_does_not_mutate_the_head_or_input_values() {
        let mut model =
            audio2face3d_gui_core::gltf::from_glb(include_bytes!("../assets/default-head.glb"))
                .unwrap();
        assert!(head_summary(&model).contains("all 52"));
        for mesh in &mut model.meshes {
            mesh.targets.retain(|t| t.name != "TongueOut");
        }
        let before = model.clone();
        assert_eq!(
            head_summary(&model),
            "Head model: 51 channels supported; unsupported: TongueOut"
        );
        assert_eq!(model, before);
    }
}
