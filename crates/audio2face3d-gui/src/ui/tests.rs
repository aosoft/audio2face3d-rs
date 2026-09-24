use super::*;
use egui::{Event, PointerButton, Pos2, Rect, Shape};

fn frame(
    ctx: &egui::Context,
    timeline: &mut Timeline,
    events: Vec<Event>,
) -> (Option<f64>, Vec<egui::epaint::ClippedShape>) {
    frame_at(ctx, timeline, events, 0., false)
}

fn frame_at(
    ctx: &egui::Context,
    timeline: &mut Timeline,
    events: Vec<Event>,
    time: f64,
    playing: bool,
) -> (Option<f64>, Vec<egui::epaint::ClippedShape>) {
    let mut player = crate::playback::Player::default();
    player.replace(crate::core::demo_clip());
    let mut snapshot = player.snapshot(std::time::Instant::now());
    snapshot.time = time;
    if playing {
        snapshot.state = crate::playback::PlaybackState::Playing;
    }
    let player = std::sync::Mutex::new(player);
    let mut seek = None;
    let output = ctx.run(
        egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1200., 700.))),
            events,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                seek =
                    timeline.show_player(ui, &player, &snapshot, &[0, 7, 17].into_iter().collect());
            });
        },
    );
    (seek, output.shapes)
}

fn plots(shapes: &[egui::epaint::ClippedShape]) -> Vec<Rect> {
    shapes
        .iter()
        .filter_map(|s| match &s.shape {
            Shape::Rect(r) if r.fill == egui::Color32::from_rgb(24, 30, 38) => Some(r.rect),
            _ => None,
        })
        .collect()
}

fn cursors(shapes: &[egui::epaint::ClippedShape]) -> Vec<[Pos2; 2]> {
    shapes
        .iter()
        .filter_map(|s| match &s.shape {
            Shape::LineSegment { points, stroke } if stroke.color == egui::Color32::YELLOW => {
                Some(*points)
            }
            _ => None,
        })
        .collect()
}

fn button(pos: Pos2, pressed: bool) -> Event {
    Event::PointerButton {
        pos,
        button: PointerButton::Primary,
        pressed,
        modifiers: egui::Modifiers::NONE,
    }
}

#[test]
fn mouse_down_seeks_immediately_and_drag_crosses_rows_without_releasing() {
    let ctx = egui::Context::default();
    let mut timeline = Timeline::default();
    frame(&ctx, &mut timeline, vec![]);
    let (_, shapes) = frame(&ctx, &mut timeline, vec![]);
    let plots = plots(&shapes);
    assert_eq!(plots.len(), 3);
    let pos = egui::pos2(
        plots[0].left() + plots[0].width() * 0.6,
        plots[0].center().y,
    );
    let (seek, shapes) = frame(
        &ctx,
        &mut timeline,
        vec![Event::PointerMoved(pos), button(pos, true)],
    );
    assert!((seek.expect("seek before mouse-up") - 3.6).abs() < 1e-5);
    let lines = cursors(&shapes);
    assert_eq!(lines.len(), 1, "one playhead for all channels");
    assert!((lines[0][0].x - pos.x).abs() < 1.);
    assert!(lines[0][0].y < plots[0].top());
    assert!(lines[0][1].y >= plots[2].bottom());
    assert_eq!(
        frame(&ctx, &mut timeline, vec![]).0,
        None,
        "stationary hold must not repeatedly reset audio"
    );
    let moved = egui::pos2(
        plots[2].left() + plots[2].width() * 0.8,
        plots[2].center().y,
    );
    assert!(
        (frame(&ctx, &mut timeline, vec![Event::PointerMoved(moved)])
            .0
            .unwrap()
            - 4.8)
            .abs()
            < 1e-5
    );
    assert_eq!(
        frame(&ctx, &mut timeline, vec![button(moved, false)]).0,
        None
    );
}

#[test]
fn ruler_seeks_in_zoomed_range_and_drag_clamps_to_visible_bounds() {
    let ctx = egui::Context::default();
    let mut timeline = Timeline {
        zoom: 4.,
        offset: 2.,
        last_time: Some(0.),
        ..Default::default()
    };
    frame(&ctx, &mut timeline, vec![]);
    let (_, shapes) = frame(&ctx, &mut timeline, vec![]);
    let plot = plots(&shapes)[0];
    // The fixed ruler is immediately above the first channel label (20px).
    let pos = egui::pos2(plot.left() + plot.width() * 0.4, plot.top() - 30.);
    assert!(
        (frame(
            &ctx,
            &mut timeline,
            vec![Event::PointerMoved(pos), button(pos, true)]
        )
        .0
        .unwrap()
            - 2.6)
            .abs()
            < 1e-5
    );
    let outside = egui::pos2(plot.right() + 100., pos.y);
    assert_eq!(
        frame(&ctx, &mut timeline, vec![Event::PointerMoved(outside)]).0,
        Some(3.5)
    );
    frame(&ctx, &mut timeline, vec![button(outside, false)]);
    timeline.overlay = true;
    let (_, shapes) = frame(&ctx, &mut timeline, vec![]);
    assert_eq!(plots(&shapes).len(), 1);
    assert!(
        cursors(&shapes).is_empty(),
        "playhead outside visible range stays hidden"
    );
}

#[test]
fn ruler_spacing_adapts_to_long_and_subsecond_ranges() {
    assert_eq!(ruler_step(600., 900.), 100.);
    assert_eq!(ruler_step(6., 900.), 1.);
    assert!((ruler_step(0.6, 900.) - 0.1).abs() < 1e-12);
    assert!((ruler_step(0.006, 900.) - 0.001).abs() < 1e-12);
}

#[test]
fn viewport_stays_still_until_the_playhead_leaves_then_recenters() {
    let ctx = egui::Context::default();
    let mut timeline = Timeline {
        zoom: 4.,
        ..Default::default()
    };
    for time in [0., 0.5, 1.49, 1.5] {
        let (_, shapes) = frame_at(&ctx, &mut timeline, vec![], time, true);
        assert_eq!(timeline.offset, 0.);
        assert_eq!(cursors(&shapes).len(), 1);
    }
    let (_, shapes) = frame_at(&ctx, &mut timeline, vec![], 1.51, true);
    assert!((timeline.offset - 0.76).abs() < 1e-9);
    let plot = plots(&shapes)[0];
    assert!((cursors(&shapes)[0][0].x - plot.center().x).abs() < 1.);
    frame_at(&ctx, &mut timeline, vec![], 2., true);
    assert!((timeline.offset - 0.76).abs() < 1e-9);
    frame_at(&ctx, &mut timeline, vec![], 5.9, true);
    assert_eq!(timeline.offset, 4.5);
    frame_at(&ctx, &mut timeline, vec![], 0.1, true);
    assert_eq!(
        timeline.offset, 0.,
        "loop/backward seek must reveal the start"
    );
}

#[test]
fn view_thumb_is_proportional_and_panning_does_not_seek() {
    let ctx = egui::Context::default();
    let mut timeline = Timeline {
        zoom: 4.,
        ..Default::default()
    };
    frame(&ctx, &mut timeline, vec![]);
    let (_, shapes) = frame(&ctx, &mut timeline, vec![]);
    let find_rect = |gray| {
        shapes
            .iter()
            .find_map(|s| match &s.shape {
                Shape::Rect(r) if r.fill == egui::Color32::from_gray(gray) => Some(r.rect),
                _ => None,
            })
            .unwrap()
    };
    let track = find_rect(35);
    let thumb = find_rect(100);
    assert!((thumb.width() / track.width() - 0.25).abs() < 1e-5);
    let pos = egui::pos2(track.left() + track.width() * 0.75, track.center().y);
    assert_eq!(
        frame(
            &ctx,
            &mut timeline,
            vec![Event::PointerMoved(pos), button(pos, true)]
        )
        .0,
        None
    );
    assert!((timeline.offset - 3.75).abs() < 1e-5);
    frame(&ctx, &mut timeline, vec![button(pos, false)]);
    frame(&ctx, &mut timeline, vec![]);
    assert!(
        (timeline.offset - 3.75).abs() < 1e-5,
        "paused manual panning must persist"
    );
    frame_at(&ctx, &mut timeline, vec![], 4.5, false);
    assert!(
        (timeline.offset - 3.75).abs() < 1e-5,
        "in-range seek must not pan"
    );
    let (_, shapes) = frame_at(&ctx, &mut timeline, vec![], 2., false);
    assert!((timeline.offset - 1.25).abs() < 1e-5);
    assert_eq!(cursors(&shapes).len(), 1);
}

#[test]
fn whole_clip_seek_bar_moves_on_mouse_down_and_reveals_target_in_same_frame() {
    let ctx = egui::Context::default();
    let mut timeline = Timeline {
        zoom: 4.,
        ..Default::default()
    };
    let mut player = crate::playback::Player::default();
    player.replace(crate::core::demo_clip());
    let mut snapshot = player.snapshot(std::time::Instant::now());
    let mut draw = |events: Vec<Event>| {
        let mut seek = None;
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1200., 700.))),
                events,
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    seek = playback_seek_bar(ui, &snapshot);
                    if let Some(time) = seek {
                        timeline.reveal_position(time, snapshot.duration);
                        snapshot.time = time;
                    }
                    timeline.show(ui, &player.clip, &snapshot, &[0].into_iter().collect());
                });
            },
        );
        (seek, output.shapes)
    };
    draw(vec![]);
    let (_, shapes) = draw(vec![]);
    let bar = shapes
        .iter()
        .find_map(|s| match &s.shape {
            Shape::Rect(r) if r.fill == egui::Color32::from_gray(25) => Some(r.rect),
            _ => None,
        })
        .unwrap();
    let pos = egui::pos2(bar.left() + bar.width() * 0.6, bar.center().y);
    let (seek, shapes) = draw(vec![Event::PointerMoved(pos), button(pos, true)]);
    assert!((seek.unwrap() - 3.6).abs() < 1e-5);
    assert!((cursors(&shapes)[0][0].x - plots(&shapes)[0].center().x).abs() < 1.);
}
