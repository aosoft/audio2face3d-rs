//! Reusable UI widgets; no native window or event loop ownership.
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

fn channel_name_width<'a>(ui: &egui::Ui, names: impl Iterator<Item = &'a str>) -> f32 {
    let font = egui::TextStyle::Body.resolve(ui.style());
    names
        .map(|name| {
            ui.painter()
                .layout_no_wrap(name.to_owned(), font.clone(), egui::Color32::WHITE)
                .size()
                .x
        })
        .fold(0., f32::max)
        .ceil()
        + 2.
}

// Keep columns compact: surplus window width must not stretch the name/value gap.
fn channel_grid(
    ui: &mut egui::Ui,
    count: usize,
    preferred_width: f32,
    mut cell: impl FnMut(&mut egui::Ui, usize),
) {
    const COLUMN_GAP: f32 = 24.;
    egui::ScrollArea::vertical()
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let width = preferred_width.min(ui.available_width()).max(1.);
            let columns = (((ui.available_width() + COLUMN_GAP) / (width + COLUMN_GAP)).floor()
                as usize)
                .max(1)
                .min(count.max(1));
            let rows = count.div_ceil(columns);
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = 0.;
                for column in 0..columns {
                    let response = ui
                        .allocate_ui_with_layout(
                            egui::vec2(width, 0.),
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                ui.set_width(width);
                                ui.spacing_mut().item_spacing = egui::vec2(4., 2.);
                                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
                                for index in column * rows..((column + 1) * rows).min(count) {
                                    ui.push_id(index, |ui| {
                                        ui.horizontal(|ui| cell(ui, index));
                                    });
                                }
                            },
                        )
                        .response;
                    if column + 1 < columns {
                        let x = response.rect.right() + COLUMN_GAP * 0.5;
                        ui.painter().vline(
                            x,
                            response.rect.y_range(),
                            ui.visuals().widgets.noninteractive.bg_stroke,
                        );
                        ui.add_space(COLUMN_GAP);
                    }
                }
            });
        });
}

fn channel_color(channel: usize) -> egui::Color32 {
    egui::Color32::from_rgb(
        90 + ((channel * 47) % 140) as u8,
        100 + ((channel * 71) % 130) as u8,
        150 + ((channel * 23) % 100) as u8,
    )
}

// Only Channels text is dimmed; timeline colors and controls remain unchanged.
fn channel_text_color(value: f32, active: egui::Color32) -> egui::Color32 {
    if value == 0.0 {
        egui::Color32::from_gray(80)
    } else {
        active
    }
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
    let name_width = channel_name_width(ui, weights.keys().map(String::as_str));
    let mut channels: Vec<_> = weights
        .iter_mut()
        .filter(|(name, _)| name.to_lowercase().contains(&query))
        .collect();
    channel_grid(ui, channels.len(), name_width + 106., |ui, index| {
        let (name, value) = &mut channels[index];
        let text_color = channel_text_color(**value, ui.visuals().text_color());
        let name_width = (ui.available_width() - 106.).max(40.);
        ui.allocate_ui_with_layout(
            egui::vec2(name_width, 20.),
            egui::Layout::left_to_right(egui::Align::Center),
            |ui| {
                // The allocation can shrink to its contents unless a minimum is set.
                ui.set_min_width(name_width);
                ui.label(egui::RichText::new(name.as_str()).color(text_color))
            },
        )
        .inner
        .on_hover_text(name.as_str());
        ui.scope(|ui| {
            ui.visuals_mut().override_text_color = Some(text_color);
            ui.add_sized(
                [42., 20.],
                egui::DragValue::new(*value)
                    .range(0.0..=1.0)
                    .speed(0.01)
                    .fixed_decimals(3),
            );
        });
        ui.spacing_mut().slider_width = 56.;
        ui.add(egui::Slider::new(*value, 0.0..=1.0).show_value(false));
    });
}

pub fn channel_values(
    ui: &mut egui::Ui,
    clip: &crate::core::Clip,
    snapshot: &crate::playback::Snapshot,
    selected: &mut std::collections::BTreeSet<usize>,
    filter: &mut String,
) {
    channel_values_for_names(ui, &clip.names, snapshot, selected, filter);
}

pub fn channel_values_for_names(
    ui: &mut egui::Ui,
    names: &[String],
    snapshot: &crate::playback::Snapshot,
    selected: &mut std::collections::BTreeSet<usize>,
    filter: &mut String,
) {
    ui.heading("Channels");
    ui.text_edit_singleline(filter);
    let query = filter.to_lowercase();
    let name_width = channel_name_width(ui, names.iter().map(String::as_str))
        + ui.spacing().icon_width
        + ui.spacing().icon_spacing;
    let channels: Vec<_> = names
        .iter()
        // Hosts may replace the result between taking names and taking a snapshot.
        // Only render rows whose interpolated and raw values are both available.
        .take(snapshot.values.len().min(snapshot.raw_values.len()))
        .enumerate()
        .filter(|(_, name)| name.to_lowercase().contains(&query))
        .collect();
    channel_grid(ui, channels.len(), name_width + 82., |ui, index| {
        let (i, name) = channels[index];
        let value = snapshot.values[i];
        let name_color = channel_text_color(value, channel_color(i));
        let value_color = channel_text_color(value, ui.visuals().text_color());
        let mut show = selected.contains(&i);
        let name_width = (ui.available_width() - 82.).max(40.);
        if ui
            .allocate_ui_with_layout(
                egui::vec2(name_width, 20.),
                egui::Layout::left_to_right(egui::Align::Center),
                |ui| {
                    ui.set_min_width(name_width);
                    ui.checkbox(&mut show, egui::RichText::new(name).color(name_color))
                },
            )
            .inner
            .on_hover_text(format!("{name}\nShow in Timeline"))
            .changed()
        {
            if show {
                selected.insert(i);
            } else {
                selected.remove(&i);
            }
        }
        ui.add_sized(
            [42., 20.],
            egui::Label::new(
                egui::RichText::new(format!("{value:.3}"))
                    .monospace()
                    .color(value_color),
            ),
        )
        .on_hover_text(format!(
            "Interpolated: {value:.6}\nRaw: {:.6}",
            snapshot.raw_values[i]
        ));
        ui.add(
            egui::ProgressBar::new(value.clamp(0., 1.))
                .desired_width(32.)
                .desired_height(8.)
                .fill(channel_color(i)),
        );
    });
}

#[derive(Default)]
pub struct Timeline {
    revision: u64,
    width: usize,
    cache: BTreeMap<usize, Vec<Option<[f32; 2]>>>,
    pub overlay: bool,
    /// Disable ruler and track seeking while retaining viewport navigation.
    pub seek_disabled: bool,
    zoom: f64,
    offset: f64,
    range: [f64; 2],
    last_time: Option<f64>,
    view_drag_offset: Option<f64>,
}
impl Timeline {
    /// Recenter only when the requested playhead lies outside the visible range.
    pub fn reveal_position(&mut self, time: f64, duration: f64) {
        let duration = duration.max(1e-6);
        self.zoom = self.zoom.clamp(1., 10.);
        let span = duration / self.zoom;
        self.offset = self.offset.clamp(0., duration - span);
        let time = time.clamp(0., duration);
        if time < self.offset || time > self.offset + span {
            self.offset = (time - span * 0.5).clamp(0., duration - span);
        }
    }

    fn view_bar(
        &mut self,
        ui: &mut egui::Ui,
        width: f32,
        duration: f64,
        playing_time: Option<f64>,
    ) {
        let span = duration / self.zoom;
        let (rect, response) =
            ui.allocate_exact_size(egui::vec2(width, 18.), egui::Sense::click_and_drag());
        let pointer = ui.input(|i| i.pointer.interact_pos());
        if response.is_pointer_button_down_on()
            && ui.input(|i| i.pointer.primary_pressed())
            && let Some(pos) = pointer
        {
            let time = ((pos.x - rect.left()) / rect.width()).clamp(0., 1.) as f64 * duration;
            self.view_drag_offset = Some(if (self.offset..=self.offset + span).contains(&time) {
                time - self.offset
            } else {
                span * 0.5
            });
        }
        if ui.input(|i| i.pointer.primary_down()) {
            if let (Some(grab), Some(pos)) = (self.view_drag_offset, pointer) {
                let time = (pos.x - rect.left()) as f64 / rect.width() as f64 * duration;
                self.offset = (time - grab).clamp(0., duration - span);
            }
        } else {
            self.view_drag_offset = None;
        }
        if let Some(time) = playing_time {
            self.reveal_position(time, duration);
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3., egui::Color32::from_gray(35));
        let thumb = egui::Rect::from_min_max(
            egui::pos2(
                rect.left() + (self.offset / duration) as f32 * rect.width(),
                rect.top() + 2.,
            ),
            egui::pos2(
                rect.left() + ((self.offset + span) / duration) as f32 * rect.width(),
                rect.bottom() - 2.,
            ),
        );
        painter.rect_filled(
            thumb,
            2.,
            egui::Color32::from_gray(if response.hovered() || self.view_drag_offset.is_some() {
                130
            } else {
                100
            }),
        );
        response.on_hover_text("Drag to pan; playback position stays unchanged");
    }
    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        clip: &crate::core::Clip,
        snapshot: &crate::playback::Snapshot,
        selected: &std::collections::BTreeSet<usize>,
    ) -> Option<f64> {
        self.show_prepared(
            ui,
            snapshot,
            selected,
            |_| {},
            |timeline, width, range| timeline.prepare(clip, selected, width, range),
        )
    }

    /// Copy graph data while locked, then release the player before drawing widgets.
    pub fn show_player(
        &mut self,
        ui: &mut egui::Ui,
        player: &std::sync::Mutex<crate::playback::Player>,
        snapshot: &crate::playback::Snapshot,
        selected: &std::collections::BTreeSet<usize>,
    ) -> Option<f64> {
        self.show_player_with_controls(ui, player, snapshot, selected, |_| {})
    }

    /// Append host playback controls after Fit, without holding the player lock.
    pub fn show_player_with_controls(
        &mut self,
        ui: &mut egui::Ui,
        player: &std::sync::Mutex<crate::playback::Player>,
        snapshot: &crate::playback::Snapshot,
        selected: &std::collections::BTreeSet<usize>,
        controls: impl FnOnce(&mut egui::Ui),
    ) -> Option<f64> {
        self.show_prepared(
            ui,
            snapshot,
            selected,
            controls,
            |timeline, width, range| {
                let player = player.lock().unwrap();
                timeline.prepare(&player.clip, selected, width, range)
            },
        )
    }

    fn prepare(
        &mut self,
        clip: &crate::core::Clip,
        selected: &std::collections::BTreeSet<usize>,
        width: usize,
        range: [f64; 2],
    ) -> Vec<String> {
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
        clip.names.clone()
    }

    fn show_prepared(
        &mut self,
        ui: &mut egui::Ui,
        snapshot: &crate::playback::Snapshot,
        selected: &std::collections::BTreeSet<usize>,
        controls: impl FnOnce(&mut egui::Ui),
        prepare: impl FnOnce(&mut Self, usize, [f64; 2]) -> Vec<String>,
    ) -> Option<f64> {
        let duration = snapshot.duration.max(1e-6);
        self.zoom = self.zoom.clamp(1., 10.);
        let previous_zoom = self.zoom;
        ui.horizontal(|ui| {
            ui.heading("Timeline");
            ui.add(
                egui::Slider::new(&mut self.zoom, 1.0..=10.0)
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
            ui.add_space(16.);
            controls(ui);
            ui.checkbox(&mut self.overlay, "Overlay selected");
            ui.label(if self.seek_disabled {
                "Seeking disabled • min/max peaks"
            } else {
                "Click/drag to seek • min/max peaks"
            });
        });
        let playing = matches!(
            snapshot.state,
            crate::playback::PlaybackState::Playing | crate::playback::PlaybackState::Buffering
        );
        let moving = self.last_time != Some(snapshot.time);
        if self.view_drag_offset.is_none() && (moving || playing || self.zoom != previous_zoom) {
            self.reveal_position(snapshot.time, duration);
        }
        self.last_time = Some(snapshot.time);
        // Reserve the scrollbar gutter in both the ruler and the tracks.
        let scroll = ui.spacing().scroll;
        let gutter = scroll.bar_width + scroll.bar_inner_margin + scroll.bar_outer_margin;
        let width = (ui.available_width() - gutter).clamp(1., 4096.) as usize;
        self.view_bar(ui, width as f32, duration, playing.then_some(snapshot.time));
        let range = [self.offset, self.offset + duration / self.zoom];
        let names = prepare(self, width, range);
        let seek_sense = if self.seek_disabled {
            egui::Sense::hover()
        } else {
            egui::Sense::click_and_drag()
        };
        let (ruler_rect, ruler_response) =
            ui.allocate_exact_size(egui::vec2(width as f32, 28.), seek_sense);
        draw_time_ruler(ui, ruler_rect, range);
        let mut seek = if self.seek_disabled {
            None
        } else {
            timeline_seek(ui, &ruler_response, ruler_rect, range, snapshot.duration)
        };
        let tracks = egui::ScrollArea::vertical()
            .id_salt("timeline-tracks")
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
            .max_height(ui.available_height().max(1.))
            .show(ui, |ui| {
                let groups: Vec<Vec<usize>> = if self.overlay {
                    vec![selected.iter().copied().collect()]
                } else {
                    selected.iter().map(|&i| vec![i]).collect()
                };
                let row_height = if self.overlay { 130. } else { 72. };
                let height = (groups.len() as f32 * row_height).max(1.);
                let (canvas, response) =
                    ui.allocate_exact_size(egui::vec2(width as f32, height), seek_sense);
                if !self.seek_disabled
                    && let Some(time) =
                        timeline_seek(ui, &response, canvas, range, snapshot.duration)
                {
                    seek = Some(time);
                }
                for (row, channels) in groups.into_iter().enumerate() {
                    if channels.is_empty() {
                        continue;
                    }
                    let top = canvas.top() + row as f32 * row_height;
                    if !self.overlay {
                        ui.painter().text(
                            egui::pos2(canvas.left(), top),
                            egui::Align2::LEFT_TOP,
                            &names[channels[0]],
                            egui::TextStyle::Body.resolve(ui.style()),
                            channel_color(channels[0]),
                        );
                    }
                    let rect = egui::Rect::from_min_size(
                        egui::pos2(canvas.left(), top + if self.overlay { 0. } else { 20. }),
                        egui::vec2(width as f32, if self.overlay { 130. } else { 50. }),
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
                    painter.text(
                        rect.left_top(),
                        egui::Align2::LEFT_TOP,
                        format!("{min:.2}..{max:.2}"),
                        egui::FontId::monospace(10.),
                        egui::Color32::GRAY,
                    );
                }
            });
        // Draw once, above all tracks and labels, without the individual plot clips.
        let cursor_time = seek.unwrap_or(snapshot.time);
        if (range[0]..=range[1]).contains(&cursor_time) {
            let x = ruler_rect.left()
                + ((cursor_time - range[0]) / (range[1] - range[0])) as f32 * ruler_rect.width();
            let rect = egui::Rect::from_min_max(
                ruler_rect.left_top(),
                egui::pos2(ruler_rect.right(), tracks.inner_rect.bottom()),
            );
            ui.painter_at(rect).vline(
                x,
                rect.y_range(),
                egui::Stroke::new(1.5_f32, egui::Color32::YELLOW),
            );
        }
        if seek.is_some() {
            ui.ctx().request_repaint();
        }
        seek
    }
}

/// Full-clip seek control, independent of the timeline viewport.
pub fn playback_seek_bar(ui: &mut egui::Ui, snapshot: &crate::playback::Snapshot) -> Option<f64> {
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width().max(16.), 22.),
        egui::Sense::click_and_drag(),
    );
    let track = rect.shrink2(egui::vec2(7., 5.));
    let seek = if snapshot.duration > 0. {
        timeline_seek(
            ui,
            &response,
            track,
            [0., snapshot.duration],
            snapshot.ready_until.min(snapshot.duration),
        )
    } else {
        None
    };
    let time = seek.unwrap_or(snapshot.time);
    let x =
        track.left() + (time / snapshot.duration.max(1e-6)).clamp(0., 1.) as f32 * track.width();
    let painter = ui.painter_at(rect);
    painter.rect_filled(track, 3., egui::Color32::from_gray(25));
    painter.rect_filled(
        egui::Rect::from_min_max(track.left_top(), egui::pos2(x, track.bottom())),
        3.,
        egui::Color32::from_rgb(45, 140, 170),
    );
    painter.circle_filled(
        egui::pos2(x, rect.center().y),
        6.,
        egui::Color32::LIGHT_GRAY,
    );
    response.on_hover_text(format!("Seek: {time:.3} / {:.3} s", snapshot.duration));
    seek
}

fn timeline_seek(
    ui: &egui::Ui,
    response: &egui::Response,
    rect: egui::Rect,
    range: [f64; 2],
    duration: f64,
) -> Option<f64> {
    // Seek on press, then only on motion while held, not on release or idle frames.
    let moved_or_pressed = ui.input(|i| {
        i.pointer.primary_down()
            && (i.pointer.primary_pressed() || i.pointer.delta() != egui::Vec2::ZERO)
    });
    if response.is_pointer_button_down_on() && moved_or_pressed {
        response.interact_pointer_pos().map(|p| {
            (range[0]
                + ((p.x - rect.left()) / rect.width()).clamp(0., 1.) as f64 * (range[1] - range[0]))
                .clamp(0., duration)
        })
    } else {
        None
    }
}

fn ruler_step(span: f64, width: f32) -> f64 {
    let desired = span / (width as f64 / 90.).max(1.);
    let magnitude = 10f64.powf(desired.log10().floor());
    let scaled = desired / magnitude;
    let nice = if scaled <= 1. {
        1.
    } else if scaled <= 2. {
        2.
    } else if scaled <= 5. {
        5.
    } else {
        10.
    };
    nice * magnitude
}

fn draw_time_ruler(ui: &egui::Ui, rect: egui::Rect, range: [f64; 2]) {
    let painter = ui.painter_at(rect);
    let span = range[1] - range[0];
    let step = ruler_step(span, rect.width());
    let minor = step / 5.;
    let decimals = (-step.log10().floor()).clamp(0., 9.) as usize;
    let stroke = egui::Stroke::new(1.0_f32, egui::Color32::GRAY);
    painter.hline(rect.x_range(), rect.bottom(), stroke);
    let first = (range[0] / minor).ceil() as i64;
    let last = (range[1] / minor).floor() as i64;
    for tick in first..=last {
        let time = tick as f64 * minor;
        let x = rect.left() + ((time - range[0]) / span) as f32 * rect.width();
        let major = tick % 5 == 0;
        painter.vline(
            x,
            (rect.bottom() - if major { 9. } else { 4. })..=rect.bottom(),
            stroke,
        );
        if major {
            let galley = painter.layout_no_wrap(
                format!("{time:.decimals$} s"),
                egui::FontId::monospace(10.),
                ui.visuals().text_color(),
            );
            let left = (x - galley.size().x * 0.5).clamp(
                rect.left(),
                (rect.right() - galley.size().x).max(rect.left()),
            );
            painter.galley(
                egui::pos2(left, rect.top()),
                galley,
                ui.visuals().text_color(),
            );
        }
    }
}
