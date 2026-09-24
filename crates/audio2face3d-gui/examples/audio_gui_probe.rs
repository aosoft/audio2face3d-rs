//! Exercise real CPAL playback while drawing the standard channel/timeline widgets.
use audio2face3d_gui::{
    audio::AudioOutput,
    core::demo_clip,
    playback::{Command, Player},
    ui::{Timeline, channel_values_for_names},
};
use std::{
    sync::{Arc, Mutex, atomic::Ordering},
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut player = Player::default();
    player.replace(demo_clip());
    let shared = Arc::new(Mutex::new(player));
    let mut audio = AudioOutput::new(shared.clone());
    let ctx = egui::Context::default();
    let mut timeline = Timeline::default();
    let mut selected = (0..52).collect();
    let mut filter = String::new();
    let names = shared.lock().unwrap().clip.names.clone();
    audio.command(Command::Play)?;
    for _ in 0..240 {
        let frame_start = Instant::now();
        let snapshot = shared.lock().unwrap().snapshot(frame_start);
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1800., 1000.),
                )),
                ..Default::default()
            },
            |ctx| {
                egui::TopBottomPanel::bottom("timeline")
                    .exact_height(500.)
                    .show(ctx, |ui| {
                        timeline.show_player(ui, &shared, &snapshot, &selected);
                    });
                egui::CentralPanel::default().show(ctx, |ui| {
                    channel_values_for_names(ui, &names, &snapshot, &mut selected, &mut filter);
                });
            },
        );
        std::thread::sleep(Duration::from_millis(16).saturating_sub(frame_start.elapsed()));
    }
    let snapshot = shared.lock().unwrap().snapshot(Instant::now());
    println!(
        "GUI load: position {:.3}s, source underruns {}, audio busy {}",
        snapshot.time,
        snapshot.underruns,
        audio.callback_contention.load(Ordering::Relaxed)
    );
    audio.stop();
    if let Some(error) = audio.errors.lock().unwrap().take() {
        return Err(error.into());
    }
    Ok(())
}
