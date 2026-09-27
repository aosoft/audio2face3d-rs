//! Optional emotion inputs shared by the GUI, configuration file and embedded hosts.
use crate::core::{Error, Result};
use audio2face3d::types::{EmotionParameters, EmotionPostProcessing, RequestOptionsBuilder};
use serde::Deserialize;
use std::{collections::BTreeMap, path::PathBuf};

pub const NAMES: [&str; 10] = [
    "amazement",
    "anger",
    "cheekiness",
    "disgust",
    "fear",
    "grief",
    "joy",
    "outofbreath",
    "pain",
    "sadness",
];

/// Unspecified values preserve backend defaults. The model path only applies to Local.
#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Settings {
    pub model: Option<PathBuf>,
    /// Send emotion settings to a remote server; no local model is needed.
    pub send_to_server: bool,
    pub beginning: BTreeMap<String, f32>,
    pub transition_time: Option<f32>,
    pub contrast: Option<f32>,
    pub smoothing: Option<f32>,
    pub use_preferred: Option<bool>,
    pub preferred_strength: Option<f32>,
    pub strength: Option<f32>,
    pub max_emotions: Option<u32>,
}
impl Settings {
    pub fn active(&self, mode: crate::inference::Mode) -> bool {
        match mode {
            crate::inference::Mode::Local => self.model.is_some(),
            crate::inference::Mode::Grpc => self.send_to_server,
            _ => false,
        }
    }
    fn parameters(&self) -> EmotionParameters {
        let mut p = EmotionParameters::default();
        p.beginning = self.beginning.clone();
        p.transition_time = self.transition_time;
        p
    }
    fn post(&self) -> EmotionPostProcessing {
        let mut p = EmotionPostProcessing::default();
        p.contrast = self.contrast;
        p.smoothing = self.smoothing;
        p.use_preferred = self.use_preferred;
        p.preferred_strength = self.preferred_strength;
        p.strength = self.strength;
        p.max_emotions = self.max_emotions;
        p
    }
    pub fn validate(&self) -> Result<()> {
        if self
            .model
            .as_ref()
            .is_some_and(|p| p.as_os_str().is_empty())
        {
            return Err(Error("Emotion model path must not be empty".into()));
        }
        if self
            .beginning
            .keys()
            .any(|name| !NAMES.contains(&name.as_str()))
        {
            return Err(Error(
                "Unknown emotion name; use the ten Audio2Face emotion names".into(),
            ));
        }
        self.parameters()
            .validate()
            .map_err(|e| Error(e.to_string()))?;
        self.post().validate().map_err(|e| Error(e.to_string()))
    }
    pub fn apply(&self, builder: RequestOptionsBuilder) -> Result<RequestOptionsBuilder> {
        self.validate()?;
        let p = self.parameters();
        let post = self.post();
        Ok(builder
            .optional_emotion((p != EmotionParameters::default()).then_some(p))
            .optional_emotion_post_processing(
                (post != EmotionPostProcessing::default()).then_some(post),
            ))
    }
}

#[cfg(feature = "standalone-app")]
pub(crate) fn controls(ui: &mut egui::Ui, settings: &mut Settings, mode: crate::inference::Mode) {
    egui::CollapsingHeader::new("Emotion").show(ui, |ui| {
        if mode == crate::inference::Mode::Local {
            ui.horizontal(|ui| {
                let mut enabled = settings.model.is_some();
                if ui.checkbox(&mut enabled, "Audio2Emotion model").changed() {
                    settings.model = enabled.then(PathBuf::new);
                }
                if let Some(path) = &mut settings.model {
                    let mut text = path.to_string_lossy().into_owned();
                    if ui
                        .add(egui::TextEdit::singleline(&mut text).desired_width(500.))
                        .changed()
                    {
                        *path = text.into();
                    }
                    if ui.button("Browse").clicked()
                        && let Some(file) = rfd::FileDialog::new()
                            .add_filter("Model", &["json"])
                            .pick_file()
                    {
                        *path = file;
                    }
                }
            });
        } else if mode == crate::inference::Mode::Grpc {
            ui.label("Audio2Emotion model is configured on the server.");
            ui.checkbox(&mut settings.send_to_server, "Send emotion settings");
        }
        ui.add_enabled_ui(settings.active(mode), |ui| {
            ui.label("Unchecked settings use backend defaults.");
            ui.label("Beginning values initialize the emotion state.");
            let mut enabled = !settings.beginning.is_empty();
            if ui
                .checkbox(&mut enabled, "Override beginning emotions")
                .changed()
            {
                settings.beginning = if enabled {
                    NAMES.into_iter().map(|n| (n.into(), 0.)).collect()
                } else {
                    Default::default()
                };
            }
            if enabled {
                egui::Grid::new("emotion-values")
                    .num_columns(4)
                    .show(ui, |ui| {
                        for (i, name) in NAMES.iter().enumerate() {
                            ui.label(*name);
                            ui.add(egui::Slider::new(
                                settings.beginning.entry((*name).into()).or_default(),
                                0.0..=1.0,
                            ));
                            if i % 2 == 1 {
                                ui.end_row();
                            }
                        }
                    });
            }
            optional_slider(
                ui,
                "Transition time (s)",
                &mut settings.transition_time,
                0.01..=10.0,
                0.5,
            );
            ui.collapsing("Audio2Emotion post-processing", |ui| {
                ui.label(
                    "Mixing controls require an Audio2Emotion model on the inference backend.",
                );
                optional_slider(ui, "Contrast", &mut settings.contrast, 0.3..=3.0, 1.0);
                optional_slider(ui, "Smoothing", &mut settings.smoothing, 0.0..=1.0, 0.7);
                optional_slider(ui, "Strength", &mut settings.strength, 0.0..=1.0, 1.0);
                optional_slider(
                    ui,
                    "Preferred strength",
                    &mut settings.preferred_strength,
                    0.0..=1.0,
                    1.0,
                );
                ui.horizontal(|ui| {
                    let mut enabled = settings.use_preferred.is_some();
                    if ui
                        .checkbox(&mut enabled, "Override preferred emotion usage")
                        .changed()
                    {
                        settings.use_preferred = enabled.then_some(false);
                    }
                    if let Some(value) = &mut settings.use_preferred {
                        ui.checkbox(value, "Use preferred");
                    }
                });
                ui.horizontal(|ui| {
                    let mut enabled = settings.max_emotions.is_some();
                    if ui.checkbox(&mut enabled, "Override max emotions").changed() {
                        settings.max_emotions = enabled.then_some(3);
                    }
                    if let Some(value) = &mut settings.max_emotions {
                        ui.add(egui::Slider::new(value, 1..=6));
                    }
                });
            });
        });
    });
}
#[cfg(feature = "standalone-app")]
fn optional_slider(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut Option<f32>,
    range: std::ops::RangeInclusive<f32>,
    default: f32,
) {
    ui.horizontal(|ui| {
        let mut enabled = value.is_some();
        if ui.checkbox(&mut enabled, label).changed() {
            *value = enabled.then_some(default);
        }
        if let Some(value) = value {
            ui.add(egui::Slider::new(value, range));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::{Mode, Request};
    #[test]
    fn model_optional_and_remote_settings_are_explicit() {
        let mut request = Request {
            mode: Mode::Local,
            ..Default::default()
        };
        request.emotion.beginning.insert("joy".into(), 0.8);
        assert!(request.options().unwrap().emotion().is_none());
        request.emotion.model = Some("emotion/model.json".into());
        request.emotion.smoothing = Some(0.6);
        let options = request.options().unwrap();
        assert_eq!(options.emotion().as_ref().unwrap().beginning["joy"], 0.8);
        assert_eq!(
            options
                .emotion_post_processing()
                .as_ref()
                .unwrap()
                .smoothing,
            Some(0.6)
        );
        request.mode = Mode::Grpc;
        assert!(request.options().unwrap().emotion().is_none());
        request.emotion.model = None;
        request.emotion.send_to_server = true;
        assert!(request.options().unwrap().emotion().is_some());
    }
    #[cfg(feature = "grpc")]
    #[test]
    fn remote_options_encode_as_ace_emotion_parameters() {
        let mut request = Request {
            mode: Mode::Grpc,
            ..Default::default()
        };
        request.emotion.send_to_server = true;
        request.emotion.beginning.insert("joy".into(), 0.8);
        request.emotion.transition_time = Some(0.4);
        request.emotion.smoothing = Some(0.6);
        let header = audio2face3d::protocol::convert::encode_request(request.options().unwrap())
            .unwrap()
            .header;
        let emotion = header.emotion_params.unwrap();
        assert_eq!(emotion.beginning_emotion["joy"], 0.8);
        assert_eq!(emotion.live_transition_time, Some(0.4));
        assert_eq!(
            header
                .emotion_post_processing_params
                .unwrap()
                .live_blend_coef,
            Some(0.6)
        );
        request.emotion.send_to_server = false;
        let header = audio2face3d::protocol::convert::encode_request(request.options().unwrap())
            .unwrap()
            .header;
        assert!(header.emotion_params.is_none());
        assert!(header.emotion_post_processing_params.is_none());
    }
    #[test]
    fn defaults_preserve_backend_and_invalid_settings_fail() {
        let settings = Settings {
            send_to_server: true,
            ..Default::default()
        };
        let mut request = Request {
            mode: Mode::Grpc,
            emotion: settings,
            ..Default::default()
        };
        assert!(request.options().unwrap().emotion().is_none());
        assert!(
            request
                .options()
                .unwrap()
                .emotion_post_processing()
                .is_none()
        );
        request.emotion.beginning.insert("unknown".into(), 0.5);
        assert!(request.options().is_err());
        request.emotion.beginning.clear();
        request.emotion.beginning.insert("joy".into(), f32::NAN);
        assert!(request.options().is_err());
        request.emotion.beginning.clear();
        request.emotion.max_emotions = Some(7);
        assert!(request.options().is_err());
    }
    #[test]
    fn output_tracks_seek_position_and_rejects_backwards_time() {
        use audio2face3d::types::{EmotionKeyframe, EmotionTrace, MediaTime, OutputEvent};
        let frame = |t, value| {
            EmotionKeyframe::new(
                MediaTime::from_seconds(t).unwrap(),
                [("joy".into(), value)].into(),
            )
            .unwrap()
        };
        let mut clip = crate::core::Clip::default();
        crate::inference::apply_event(
            &mut clip,
            OutputEvent::Emotion(EmotionTrace {
                smoothed: vec![frame(1., 0.2), frame(2., 0.8)],
                ..Default::default()
            }),
        )
        .unwrap();
        assert!(clip.emotion_at(0.).is_none());
        assert_eq!(clip.emotion_at(1.5).unwrap()["joy"], 0.2);
        assert_eq!(clip.emotion_at(2.).unwrap()["joy"], 0.8);
        assert!(clip.push_emotions(vec![frame(0., 0.)]).is_err());
    }
}
