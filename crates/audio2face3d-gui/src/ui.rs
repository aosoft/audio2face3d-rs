//! Reusable UI widgets; no native window or event loop ownership.
use std::collections::BTreeMap;

fn channel_color(channel: usize) -> egui::Color32 {
    egui::Color32::from_rgb(
        90 + ((channel * 47) % 140) as u8,
        100 + ((channel * 71) % 130) as u8,
        150 + ((channel * 23) % 100) as u8,
    )
}

pub fn manual_channels(
    ui: &mut egui::Ui,
    weights: &mut BTreeMap<String, f32>,
    filter: &mut String,
) {
    ui.heading("Channels");
    ui.text_edit_singleline(filter);
    if ui.button("Reset all").clicked() {
        weights.values_mut().for_each(|v| *v = 0.);
    }
    let query = filter.to_lowercase();
    egui::ScrollArea::vertical().show(ui, |ui| {
        for (name, value) in weights {
            if !name.to_lowercase().contains(&query) {
                continue;
            }
            ui.label(name);
            ui.add(egui::Slider::new(value, 0.0..=1.0).fixed_decimals(3));
        }
    });
}

pub fn channel_values(
    ui: &mut egui::Ui,
    clip: &crate::core::Clip,
    snapshot: &crate::playback::Snapshot,
    selected: &mut std::collections::BTreeSet<usize>,
    filter: &mut String,
) {
    ui.heading("Channels");
    ui.text_edit_singleline(filter);
    let query = filter.to_lowercase();
    egui::ScrollArea::vertical().show(ui, |ui| {
        for (i, name) in clip.names.iter().enumerate() {
            if !name.to_lowercase().contains(&query) {
                continue;
            }
            let mut show = selected.contains(&i);
            if ui
                .checkbox(&mut show, egui::RichText::new(name).color(channel_color(i)))
                .changed()
            {
                if show {
                    selected.insert(i);
                } else {
                    selected.remove(&i);
                }
            }
            let value = snapshot.values[i];
            ui.add(
                egui::ProgressBar::new(value.clamp(0., 1.))
                    .text(format!("{value:.3}  (raw {:.3})", snapshot.raw_values[i])),
            );
        }
    });
}

#[derive(Default)]
pub struct Timeline {
    revision: u64,
    width: usize,
    cache: BTreeMap<usize, Vec<Option<[f32; 2]>>>,
    pub overlay: bool,
    zoom: f64,
    offset: f64,
    range: [f64; 2],
}
impl Timeline {
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        clip: &crate::core::Clip,
        snapshot: &crate::playback::Snapshot,
        selected: &std::collections::BTreeSet<usize>,
    ) -> Option<f64> {
        ui.horizontal(|ui| {
            ui.heading("Timeline");
            ui.checkbox(&mut self.overlay, "Overlay selected");
            ui.label("Click/drag to seek • min/max peaks");
        });
        let duration = snapshot.duration.max(1e-6);
        self.zoom = self.zoom.clamp(1., 1000.);
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut self.zoom, 1.0..=1000.0)
                    .logarithmic(true)
                    .fixed_decimals(1)
                    .text("Zoom"),
            );
            if ui.button("Fit").clicked() {
                self.zoom = 1.;
                self.offset = 0.;
            }
            let span = duration / self.zoom;
            self.offset = self.offset.clamp(0., duration - span);
            ui.add(
                egui::Slider::new(&mut self.offset, 0.0..=duration - span)
                    .fixed_decimals(3)
                    .text("Start (s)"),
            );
            ui.label(format!("{:.3}–{:.3} s", self.offset, self.offset + span));
        });
        let range = [self.offset, self.offset + duration / self.zoom];
        let width = ui.available_width().clamp(1., 4096.) as usize;
        if self.revision != clip.revision || self.width != width || self.range != range {
            self.cache.clear();
            self.revision = clip.revision;
            self.width = width;
            self.range = range;
        }
        for &channel in selected {
            self.cache
                .entry(channel)
                .or_insert_with(|| clip.envelope_range(channel, width, range[0], range[1]));
        }
        let mut seek = None;
        egui::ScrollArea::vertical()
            .max_height(200.)
            .show(ui, |ui| {
                let groups: Vec<Vec<usize>> = if self.overlay {
                    vec![selected.iter().copied().collect()]
                } else {
                    selected.iter().map(|&i| vec![i]).collect()
                };
                for channels in groups {
                    if channels.is_empty() {
                        continue;
                    }
                    if !self.overlay {
                        ui.colored_label(channel_color(channels[0]), &clip.names[channels[0]]);
                    }
                    let (rect, response) = ui.allocate_exact_size(
                        egui::vec2(width as f32, if self.overlay { 130. } else { 50. }),
                        egui::Sense::click_and_drag(),
                    );
                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 2., egui::Color32::from_rgb(24, 30, 38));
                    let mut min = 0f32;
                    let mut max = 1f32;
                    for channel in &channels {
                        for range in self.cache[channel].iter().flatten() {
                            min = min.min(range[0]);
                            max = max.max(range[1]);
                        }
                    }
                    for channel in channels {
                        let color = channel_color(channel);
                        let mut previous = None;
                        for (i, range) in self.cache[&channel].iter().enumerate() {
                            if let Some([lo, hi]) = range {
                                let x = rect.left() + i as f32;
                                let a = egui::pos2(
                                    x,
                                    rect.bottom() - (lo - min) / (max - min) * rect.height(),
                                );
                                let b = egui::pos2(
                                    x,
                                    rect.bottom() - (hi - min) / (max - min) * rect.height(),
                                );
                                painter.line_segment([a, b], egui::Stroke::new(1.0_f32, color));
                                if let Some(p) = previous {
                                    painter.line_segment([p, a], egui::Stroke::new(1.0_f32, color));
                                }
                                previous = Some(b);
                            }
                        }
                    }
                    let x = rect.left()
                        + ((snapshot.time - range[0]) / (range[1] - range[0])) as f32
                            * rect.width();
                    painter.line_segment(
                        [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                        egui::Stroke::new(1.5_f32, egui::Color32::WHITE),
                    );
                    painter.text(
                        rect.left_top(),
                        egui::Align2::LEFT_TOP,
                        format!("{min:.2}..{max:.2}"),
                        egui::FontId::monospace(10.),
                        egui::Color32::GRAY,
                    );
                    if (response.clicked() || response.dragged())
                        && let Some(p) = response.interact_pointer_pos()
                    {
                        seek = Some(
                            range[0]
                                + ((p.x - rect.left()) / rect.width()).clamp(0., 1.) as f64
                                    * (range[1] - range[0]),
                        );
                    }
                }
            });
        seek
    }
}
