use audio2face3d::common::AudioAccumulator;
use std::sync::Arc;

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

#[test]
fn shared_audio_producers_close_reset_and_reuse_across_threads() {
    const PRODUCERS: usize = 8;
    const CHUNKS_PER_PRODUCER: usize = 64;
    const CHUNK_SAMPLES: usize = 257;

    let accumulator = Arc::new(AudioAccumulator::new(16_000, 0).unwrap());
    let workers: Vec<_> = (0..PRODUCERS)
        .map(|producer| {
            let accumulator = Arc::clone(&accumulator);
            std::thread::spawn(move || {
                let samples = vec![producer as f32; CHUNK_SAMPLES];
                for _ in 0..CHUNKS_PER_PRODUCER {
                    accumulator.accumulate(&samples).unwrap();
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }

    assert_eq!(
        accumulator.nb_accumulated_samples(),
        PRODUCERS * CHUNKS_PER_PRODUCER * CHUNK_SAMPLES
    );
    accumulator.close().unwrap();

    let accumulator = std::thread::spawn(move || {
        accumulator.reset().unwrap();
        accumulator.accumulate(&[0.25; CHUNK_SAMPLES]).unwrap();
        accumulator.close().unwrap();
        accumulator
    })
    .join()
    .unwrap();
    assert_eq!(accumulator.nb_accumulated_samples(), CHUNK_SAMPLES);
}

#[cfg(feature = "animation")]
#[test]
fn published_track_boundary_remains_fixed() {
    assert_eq!(audio2face3d::animation::MAX_REGRESSION_TRACKS, 32);
    assert_eq!(audio2face3d::animation::MAX_DIFFUSION_TRACKS, 32);
}
