//! Model acquisition panel for the standard desktop host.
use audio2face3d::model_management::{
    DownloadOptions, EnginePrecision, MODEL_PRESETS, ModelEngineBuildRequest, ModelPreset,
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    thread::JoinHandle,
};

#[derive(Clone, Copy, Debug)]
enum Operation {
    Download,
    Engine,
    Prepare,
}

struct Completed {
    descriptor: Option<PathBuf>,
    emotion: bool,
    regression: bool,
    device: u32,
}
impl Completed {
    fn supported(&self) -> bool {
        self.descriptor.is_some()
            && cfg!(feature = "local")
            && (self.regression || (self.emotion && cfg!(feature = "emotion")))
    }
    fn apply(&self, request: &mut crate::inference::Request) {
        if !self.supported() {
            return;
        }
        let descriptor = self.descriptor.as_ref().unwrap();
        if self.emotion {
            #[cfg(feature = "emotion")]
            {
                request.emotion.model = Some(descriptor.clone());
            }
        } else {
            request.model = descriptor.clone();
        }
        request.device = self.device as usize;
        request.mode = crate::inference::Mode::Local;
    }
}
struct Job {
    worker: JoinHandle<Result<Completed, String>>,
    status: Arc<Mutex<String>>,
}

pub(super) struct Panel {
    pub open: bool,
    preset: usize,
    output: String,
    token_environment: String,
    precision: EnginePrecision,
    device: u32,
    max_batch: u32,
    force: bool,
    job: Option<Job>,
    status: String,
    completed: Option<Completed>,
}
impl Default for Panel {
    fn default() -> Self {
        Self {
            open: false,
            preset: 3,
            output: "models".into(),
            token_environment: "HF_TOKEN".into(),
            precision: EnginePrecision::Default,
            device: 0,
            max_batch: 0,
            force: false,
            job: None,
            status: String::new(),
            completed: None,
        }
    }
}
impl Panel {
    pub fn busy(&self) -> bool {
        self.job.is_some()
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        request: &mut crate::inference::Request,
        editable: bool,
        logger: Arc<crate::logging::GuiLogger>,
    ) {
        if let Some(job) = &self.job {
            self.status = job.status.lock().unwrap().clone();
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
            if job.worker.is_finished() {
                let job = self.job.take().unwrap();
                match job
                    .worker
                    .join()
                    .unwrap_or_else(|_| Err("Model worker failed".into()))
                {
                    Ok(completed) => {
                        self.status = completed.descriptor.as_ref().map_or_else(
                            || {
                                "Download verified. Generate an engine before local inference."
                                    .into()
                            },
                            |path| format!("Ready: {}", path.display()),
                        );
                        self.completed = Some(completed);
                    }
                    Err(error) => self.status = format!("Model operation failed: {error}"),
                }
            }
        }
        if self.busy() && ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.open = true;
        }
        let mut open = self.open;
        egui::Window::new("Audio2Face models").open(&mut open).resizable(true).show(ctx, |ui| {
            let enabled = editable && !self.busy();
            ui.label("Download models from Hugging Face and generate TensorRT engines on this computer.");
            ui.add_enabled_ui(enabled, |ui| {
                egui::ComboBox::from_label("Preset").selected_text(MODEL_PRESETS[self.preset].name)
                    .show_ui(ui, |ui| {
                        for (index, preset) in MODEL_PRESETS.iter().enumerate() {
                            ui.selectable_value(&mut self.preset, index, preset.name);
                        }
                    });
                let preset = MODEL_PRESETS[self.preset];
                ui.hyperlink_to(preset.repository, format!("https://huggingface.co/{}", preset.repository));
                ui.label("Accept the repository terms on Hugging Face and set your token environment variable before starting the app.");
                ui.horizontal(|ui| {
                    ui.label("Models directory"); ui.text_edit_singleline(&mut self.output);
                    if ui.button("Browse").clicked() && let Some(path) = rfd::FileDialog::new().pick_folder() {
                        self.output = path.to_string_lossy().into_owned();
                    }
                });
                ui.horizontal(|ui| { ui.label("Token environment variable"); ui.text_edit_singleline(&mut self.token_environment); });
                egui::ComboBox::from_label("Precision").selected_text(self.precision.to_string()).show_ui(ui, |ui| {
                    for precision in [EnginePrecision::Default, EnginePrecision::Fp16, EnginePrecision::Fp32] {
                        ui.selectable_value(&mut self.precision, precision, precision.to_string());
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("GPU device"); ui.add(egui::DragValue::new(&mut self.device));
                    ui.label("Max batch (0 = automatic)"); ui.add(egui::DragValue::new(&mut self.max_batch));
                });
                ui.checkbox(&mut self.force, "Replace existing model / engine artifacts (--force)");
                ui.label("Engine generation uses the configured CUDA / TensorRT paths and trtexec.");
                ui.horizontal(|ui| {
                    for (label, operation) in [("Download", Operation::Download), ("Generate engine", Operation::Engine), ("Download + generate", Operation::Prepare)] {
                        if ui.button(label).clicked() {
                            self.start(operation, preset, request, logger.clone());
                        }
                    }
                });
            });
            if self.busy() { ui.spinner(); ui.label("Wait for completion before closing the application."); }
            ui.label(&self.status);
            if let Some(completed) = &self.completed
                && ui.add_enabled(enabled && completed.supported(), egui::Button::new("Use for local inference")).clicked() {
                completed.apply(request);
            }
        });
        self.open = open;
    }

    fn start(
        &mut self,
        operation: Operation,
        preset: ModelPreset,
        request: &crate::inference::Request,
        logger: Arc<crate::logging::GuiLogger>,
    ) {
        if self.busy() {
            return;
        }
        if self.output.trim().is_empty() || self.token_environment.trim().is_empty() {
            self.status =
                "Models directory and token environment variable must not be empty".into();
            return;
        }
        let runtime = match request.native_runtime() {
            Ok(runtime) => runtime,
            Err(error) => {
                self.status = error.to_string();
                return;
            }
        };
        let context = audio2face3d::Audio2Face3DContext::builder()
            .logger(logger)
            .native_runtime(runtime)
            .build();
        let scope = audio2face3d::logging::integration::LogScope::new(context);
        let download = preset.request(&self.output, self.token_environment.clone());
        let build = ModelEngineBuildRequest {
            model_directory: download.output.clone(),
            precision: self.precision,
            device_id: self.device,
            max_batch_size: (self.max_batch != 0).then_some(self.max_batch as u64),
            replace: self.force,
            trtexec: std::env::var_os("TRTEXEC")
                .map(PathBuf::from)
                .unwrap_or_else(|| "trtexec".into()),
        };
        let force = self.force;
        let status = Arc::new(Mutex::new("Starting model operation".into()));
        let progress = status.clone();
        let worker = std::thread::Builder::new()
            .name("model-management".into())
            .spawn(move || {
                scope.in_scope(|| {
                    if matches!(operation, Operation::Download | Operation::Prepare) {
                        *progress.lock().unwrap() =
                            format!("Downloading / verifying {}", preset.name);
                        download
                            .execute_with_options(DownloadOptions { force }, None)
                            .map_err(|e| e.to_string())?;
                    }
                    let descriptor = if matches!(operation, Operation::Engine | Operation::Prepare)
                    {
                        *progress.lock().unwrap() =
                            format!("Generating / verifying {} TensorRT engine", preset.name);
                        Some(build.execute().map_err(|e| e.to_string())?.model_descriptor)
                    } else {
                        None
                    };
                    Ok(Completed {
                        descriptor,
                        emotion: preset.name == "emotion",
                        regression: matches!(preset.name, "mark" | "james" | "claire"),
                        device: build.device_id,
                    })
                })
            });
        match worker {
            Ok(worker) => {
                self.completed = None;
                self.job = Some(Job { worker, status });
            }
            Err(error) => self.status = format!("Cannot start model worker: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_supported_prepared_models_change_local_inputs() {
        let mut request = crate::inference::Request::default();
        let original = request.clone();
        let mut completed = Completed {
            descriptor: Some("models/mark/model_fp16.json".into()),
            emotion: false,
            regression: false,
            device: 2,
        };
        completed.apply(&mut request);
        assert_eq!(request.model, original.model); // Diffusion is acquisition-only.
        completed.regression = true;
        completed.apply(&mut request);
        if cfg!(feature = "local") {
            assert_eq!(request.model, completed.descriptor.clone().unwrap());
            assert_eq!(request.device, 2);
            assert_eq!(request.mode, crate::inference::Mode::Local);
        } else {
            assert_eq!(request.model, original.model);
            assert_eq!(request.mode, original.mode);
        }
        completed.descriptor = None;
        assert!(!completed.supported()); // A downloaded ONNX is not an engine.
    }
    #[cfg(all(feature = "local", feature = "emotion"))]
    #[test]
    fn emotion_selection_preserves_the_face_model_and_uses_the_build_device() {
        let mut request = crate::inference::Request {
            model: "models/mark/model.json".into(),
            ..Default::default()
        };
        Completed {
            descriptor: Some("models/emotion/model.json".into()),
            emotion: true,
            regression: false,
            device: 2,
        }
        .apply(&mut request);
        assert_eq!(request.model, PathBuf::from("models/mark/model.json"));
        assert_eq!(
            request.emotion.model,
            Some("models/emotion/model.json".into())
        );
        assert_eq!(request.device, 2);
        assert_eq!(request.mode, crate::inference::Mode::Local);
    }
}
