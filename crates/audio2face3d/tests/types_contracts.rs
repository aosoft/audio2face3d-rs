use audio2face3d::types::*;
use std::{collections::BTreeMap, sync::Arc};

#[test]
fn pcm_ownership_and_channel_boundaries() {
    let data = vec![1, 2, 3, 4];
    let ptr = data.as_ptr();
    let pcm = PcmBuffer::from_vec(data).unwrap();
    assert_eq!(ptr, pcm.as_bytes().as_ptr());
    assert_eq!(pcm.sample_frames(AudioFormat::MONO_16KHZ).unwrap(), 2);
    assert_eq!(
        pcm.sample_frames(AudioFormat::pcm16(48000, 2).unwrap())
            .unwrap(),
        1
    );
    let returned = pcm.into_vec();
    assert_eq!(ptr, returned.as_ptr());
    assert!(PcmBuffer::from_vec(vec![0]).is_err());
    assert!(
        PcmBuffer::from_vec(vec![0; 2])
            .unwrap()
            .sample_frames(AudioFormat::pcm16(48000, 2).unwrap())
            .is_err()
    );
    assert!(AudioFormat::pcm16(0, 1).is_err());
    assert!(AudioFormat::pcm16(16000, 0).is_err());
}

#[test]
fn sample_time_does_not_accumulate_rounding_error() {
    for rate in [16000, 44100, 48000] {
        for samples in [0, 1, 533, 534, 16001, u64::from(rate) * 600 + 1] {
            let time = MediaTime::from_samples(SamplePosition(samples), rate).unwrap();
            assert_eq!(time.nearest_sample(rate).unwrap(), SamplePosition(samples));
            assert!(
                MediaTime::from_seconds(time.as_seconds())
                    .unwrap()
                    .as_nanos()
                    .abs_diff(time.as_nanos())
                    <= 1
            );
        }
    }
    assert_eq!(
        MediaTime::from_samples(SamplePosition(1), 16000)
            .unwrap()
            .as_nanos(),
        62500
    );
    assert!(MediaTime::from_samples(SamplePosition(u64::MAX), 1).is_err());
    assert!(SamplePosition(u64::MAX).checked_add(1).is_err());
    assert!(MediaTime::ZERO.nearest_sample(0).is_err());
}

#[test]
fn time_rejects_nonfinite_negative_and_overflow() {
    for v in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.1,
        f64::MAX,
        u64::MAX as f64 / 1e9,
    ] {
        assert!(MediaTime::from_seconds(v).is_err(), "{v}");
    }
    assert_eq!(MediaTime::from_seconds(0.0000000016).unwrap().as_nanos(), 2);
}

#[test]
fn dynamic_layout_is_shared_and_weights_are_moved() {
    let layout = Arc::new(CurveLayout::new(LayoutId(7), vec!["A".into(), "B".into()]).unwrap());
    let weights = vec![0.2, 1.1];
    let ptr = weights.as_ptr();
    let frame = CurveFrame::new(layout.clone(), MediaTime::ZERO, weights).unwrap();
    assert!(Arc::ptr_eq(&layout, frame.layout()));
    assert_eq!(ptr, frame.values().as_ptr());
    let (_, _, returned) = frame.into_parts();
    assert_eq!(returned.as_ptr(), ptr);
    assert!(CurveFrame::new(layout.clone(), MediaTime::ZERO, vec![0.0]).is_err());
    assert!(CurveFrame::new(layout, MediaTime::ZERO, vec![0.0, f32::NAN]).is_err());
    assert!(CurveLayout::new(LayoutId(0), vec!["A".into(), "A".into()]).is_err());
    assert!(CurveLayout::new(LayoutId(0), vec![String::new()]).is_err());
}

#[test]
fn options_preserve_absence_empty_zero_and_false() {
    let mut request = RequestOptions::default();
    assert!(request.face.is_none());
    request.face = Some(FaceParameters::default());
    assert_eq!(request.face.as_ref().unwrap().upper_face_strength, None);
    request.face.as_mut().unwrap().upper_face_strength = Some(0.0);
    let mut blend = BlendshapeParameters::default();
    blend.clamp = Some(false);
    blend.multipliers.insert("JawOpen".into(), 0.0);
    request.blendshapes = Some(blend);
    let mut post = EmotionPostProcessing::default();
    post.use_preferred = Some(false);
    post.strength = Some(0.0);
    request.emotion_post_processing = Some(post);
    request.validate().unwrap();
    assert_eq!(request.blendshapes.as_ref().unwrap().clamp, Some(false));
    request.face.as_mut().unwrap().upper_face_strength = Some(f32::NAN);
    assert!(request.validate().is_err());
}

#[test]
fn emotion_trace_can_represent_outputs_without_weakening_input_validation() {
    let frame =
        EmotionKeyframe::new(MediaTime::ZERO, BTreeMap::from([("joy".into(), 1.2)])).unwrap();
    assert!(frame.validate_input().is_err());
    assert!(
        EmotionKeyframe::new(MediaTime::ZERO, BTreeMap::from([("joy".into(), f32::NAN)])).is_err()
    );
    let frame =
        EmotionKeyframe::new(MediaTime::ZERO, BTreeMap::from([("joy".into(), 0.0)])).unwrap();
    frame.validate_input().unwrap();
    assert_eq!(frame.values().get("joy"), Some(&0.0));
    assert_eq!(frame.values().get("anger"), None);
}
