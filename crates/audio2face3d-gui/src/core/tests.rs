use super::*;

fn receiving() -> Clip {
    let mut clip = Clip::running();
    clip.set_names(vec!["JawOpen".into()]).unwrap();
    clip.push_audio(0, &[0.25; 1600]).unwrap();
    clip.push_frame(0., vec![0.1]).unwrap();
    clip.push_frame(1. / 30., vec![0.2]).unwrap();
    clip.push_frame(2. / 30., vec![0.3]).unwrap();
    clip
}
fn fingerprint(
    clip: &Clip,
) -> (
    u64,
    ClipRevision,
    usize,
    usize,
    usize,
    Vec<String>,
    f64,
    bool,
) {
    (
        clip.revision(),
        clip.revisions(),
        clip.retained,
        clip.audio().len(),
        clip.frames().len(),
        clip.names().to_vec(),
        clip.curve_end,
        clip.curve_gap,
    )
}

#[test]
fn rejected_media_layout_and_budget_changes_are_atomic() {
    let mut clip = receiving();
    let before = fingerprint(&clip);
    assert!(clip.push_frame(f64::NAN, vec![0.]).is_err());
    assert!(clip.push_frame(0.1, vec![]).is_err());
    assert!(clip.push_frame(0.1, vec![f32::INFINITY]).is_err());
    assert!(clip.push_frame(0., vec![0.]).is_err());
    assert!(clip.push_audio(1, &[0.]).is_err());
    assert!(clip.push_audio(1600, &[f32::NAN]).is_err());
    assert!(clip.set_names(vec!["Other".into()]).is_err());
    assert!(clip.set_names(vec![]).is_err());
    assert_eq!(fingerprint(&clip), before);

    // Inject a full budget without allocating hundreds of megabytes.
    clip.retained = MAX_BYTES;
    let before = fingerprint(&clip);
    assert!(clip.push_audio(1600, &[0.]).is_err());
    assert!(clip.push_frame(0.1, vec![0.]).is_err());
    assert_eq!(fingerprint(&clip), before);
    let mut layout = Clip::running();
    layout.retained = MAX_BYTES;
    let before = fingerprint(&layout);
    assert!(layout.set_names(vec!["JawOpen".into()]).is_err());
    assert_eq!(fingerprint(&layout), before);
}

#[test]
fn completion_checks_coverage_and_rejects_later_media_or_reopening() {
    let mut incomplete = Clip::running();
    incomplete.set_names(vec!["JawOpen".into()]).unwrap();
    incomplete.push_audio(0, &[0.; 1600]).unwrap();
    let before = fingerprint(&incomplete);
    assert!(incomplete.finish_collection().is_err());
    assert_eq!(fingerprint(&incomplete), before);
    let mut clip = receiving();
    clip.finish_collection().unwrap();
    let before = fingerprint(&clip);
    assert!(clip.push_audio(1600, &[0.]).is_err());
    assert!(clip.push_frame(0.1, vec![0.]).is_err());
    assert!(clip.set_names(vec!["JawOpen".into()]).is_err());
    assert!(clip.set_session(SessionState::Running).is_err());
    assert!(
        clip.set_session(SessionState::Failed("late job failure".into()))
            .is_err()
    );
    assert_eq!(fingerprint(&clip), before);
    assert_eq!(clip.session(), &SessionState::Completed);
}

#[test]
fn audio_changes_preserve_curve_revision_and_exhaustion_never_wraps() {
    let mut clip = receiving();
    let revisions = clip.revisions();
    clip.push_audio(1600, &[0.]).unwrap();
    assert_eq!(clip.revisions().layout, revisions.layout);
    assert_eq!(clip.revisions().curves, revisions.curves);
    assert_eq!(clip.revisions().audio, revisions.audio + 1);
    clip.revision = u64::MAX;
    let before = fingerprint(&clip);
    assert!(clip.push_audio(1601, &[0.]).is_err());
    assert!(clip.push_frame(0.1, vec![0.]).is_err());
    assert_eq!(fingerprint(&clip), before);
}
