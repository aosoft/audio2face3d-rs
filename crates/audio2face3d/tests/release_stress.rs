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
fn asynchronous_blendshape_reset_replay_and_drop_stress() {
    // This is a low-level CPU solver/job-runner stress test, not a public
    // facade integration test; facade ownership and completion are covered by
    // the API contract tests.
    use audio2face3d::animation::{
        BlendshapeData, BlendshapeSolverParameters, CpuBlendshapeJobRunner, CpuBlendshapeSolver,
    };
    use std::sync::mpsc;

    let data = BlendshapeData {
        neutral_pose: vec![0.0, 0.0, 0.0],
        delta_poses: vec![1.0, 0.0, 0.0],
        pose_names: vec!["pose".into()],
        pose_mask: None,
    };
    let mut solver = CpuBlendshapeSolver::new(data).unwrap();
    solver
        .set_parameters(BlendshapeSolverParameters {
            l1_regularization: 0.0,
            l2_regularization: 0.0,
            symmetry_regularization: 0.0,
            temporal_regularization: 1.0,
            ..BlendshapeSolverParameters::default()
        })
        .unwrap();
    solver.prepare().unwrap();
    let runner = CpuBlendshapeJobRunner::new(solver);
    let (sender, receiver) = mpsc::channel();
    for iteration in 0..512 {
        if iteration % 16 == 0 {
            runner.reset().unwrap();
        }
        let sender = sender.clone();
        runner
            .solve_async(vec![0.75, 0.0, 0.0], move |result| {
                sender.send(result.unwrap()[0]).unwrap();
            })
            .unwrap();
    }
    drop(sender);
    runner.wait().unwrap();
    let values = receiver.into_iter().collect::<Vec<_>>();
    assert_eq!(values.len(), 512);
    assert!(values.iter().all(|value| value.is_finite()));
}

#[cfg(feature = "animation")]
#[test]
fn published_track_boundary_remains_fixed() {
    assert_eq!(audio2face3d::animation::MAX_REGRESSION_TRACKS, 32);
    assert_eq!(audio2face3d::animation::MAX_DIFFUSION_TRACKS, 32);
}
