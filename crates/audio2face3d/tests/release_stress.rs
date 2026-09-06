use audio2face3d::common::AudioAccumulator;

#[test]
fn long_audio_accumulation_reset_and_reuse_stays_bounded() {
    const CHUNK_SAMPLES: usize = 1_600;
    const CHUNKS: usize = 600;
    let accumulator = AudioAccumulator::new(16_000, 0).unwrap();
    let chunk = vec![0.125; CHUNK_SAMPLES];
    for _ in 0..CHUNKS {
        accumulator.accumulate(&chunk).unwrap();
    }
    assert_eq!(accumulator.nb_accumulated_samples(), CHUNK_SAMPLES * CHUNKS);
    accumulator.close().unwrap();
    accumulator.reset().unwrap();
    assert_eq!(accumulator.nb_accumulated_samples(), 0);
    accumulator.accumulate(&chunk).unwrap();
    assert_eq!(accumulator.nb_accumulated_samples(), CHUNK_SAMPLES);
}

#[cfg(feature = "animation")]
#[test]
fn published_track_boundary_remains_fixed() {
    assert_eq!(audio2face3d::animation::MAX_REGRESSION_TRACKS, 32);
    assert_eq!(audio2face3d::animation::MAX_DIFFUSION_TRACKS, 32);
}
