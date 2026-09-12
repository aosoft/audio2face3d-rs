//! Regression coverage for the production CPU solver using local SDK captures.
use super::*;
use serde::Deserialize;
use std::{path::PathBuf, sync::Arc};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    config: PathBuf,
    data: PathBuf,
    geometry_capture: PathBuf,
    sdk_weights_capture: PathBuf,
    output: PathBuf,
}

fn values(bytes: &[u8], record: &serde_json::Value) -> Vec<f32> {
    assert_eq!(record["dtype"], "f32le");
    let start = record["offset_bytes"].as_u64().unwrap() as usize;
    let len = record["byte_length"].as_u64().unwrap() as usize;
    assert_eq!(len % 4, 0);
    bytes[start..start + len]
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires CUDA and AUDIO2FACE3D_BVLS_REPLAY_MANIFEST with licensed model/captures"]
fn production_cpu_solver_matches_sdk_capture() {
    let path = PathBuf::from(std::env::var_os("AUDIO2FACE3D_BVLS_REPLAY_MANIFEST").unwrap());
    let manifest: Manifest = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let resolve = |name: &std::path::Path| path.parent().unwrap().join(name);
    let config = crate::common::load_blendshape_config(resolve(&manifest.config))
        .unwrap()
        .blendshape_params;
    let data = BlendshapeData::load_npz(resolve(&manifest.data)).unwrap();
    let mut solver = CpuBlendshapeSolver::from_config(data, &config).unwrap();
    solver.prepare().unwrap();
    let device = crate::cuda::GpuDevice::new(0).unwrap();
    let stream = Arc::new(device.create_stream().unwrap());
    let mut generator = rhs::GpuRhs::new(
        solver.prepared.as_ref().unwrap(),
        solver.data.neutral_pose.len(),
        &device,
        Arc::clone(&stream),
    )
    .unwrap();
    let mut target_buffer = device.allocate(solver.data.neutral_pose.len()).unwrap();
    let geometry = resolve(&manifest.geometry_capture);
    let expected = resolve(&manifest.sdk_weights_capture);
    let read = |directory: &std::path::Path| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(directory.join("artifact.json")).unwrap()).unwrap()
    };
    let geometry_manifest = read(&geometry);
    let expected_manifest = read(&expected);
    let geometry_bytes = std::fs::read(geometry.join("values.f32le")).unwrap();
    let expected_bytes = std::fs::read(expected.join("values.f32le")).unwrap();
    let mut outputs = Vec::new();
    let mut maximum = 0.0_f32;
    let mut elapsed_ns = 0_u128;
    let mut svd_calls = 0;
    let mut iterations = 0;
    for record in geometry_manifest["records"].as_array().unwrap() {
        if record["component"] != "skin" || record["track"] != 0 {
            continue;
        }
        let frame = record["frame"].as_u64().unwrap() as usize;
        assert_eq!(frame, outputs.len());
        target_buffer
            .copy_from(&values(&geometry_bytes, record), &stream)
            .unwrap();
        let mut atb = vec![0.0; generator.active_count()];
        generator.compute(target_buffer.view(), &mut atb).unwrap();
        // Count the very same BVLS core in a separate, untimed call. Production
        // disables instrumentation and does not perform this extra solve.
        let prepared = solver.prepared.as_ref().unwrap();
        let a = prepared
            .cpu_matrix
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>();
        let temporal = solver.parameters.temporal_regularization * prepared.scale_factor as f32;
        let b = atb
            .iter()
            .zip(&prepared.previous_weights)
            .map(|(value, previous)| f64::from(*value + temporal * previous))
            .collect::<Vec<_>>();
        let mut upper = vec![1.0; atb.len()];
        let (mut traced, first) = bvls::solve(&a, &b, &upper, solver.parameters.tolerance).unwrap();
        svd_calls += first.svd_calls;
        iterations += first.iterations;
        if !prepared.cancel_pairs.is_empty() {
            for &(i, j) in &prepared.cancel_pairs {
                upper[if traced[i] >= traced[j] { j } else { i }] = 1e-10;
            }
            let (next, second) = bvls::solve(&a, &b, &upper, solver.parameters.tolerance).unwrap();
            traced = next;
            svd_calls += second.svd_calls;
            iterations += second.iterations;
        }
        let started = std::time::Instant::now();
        let actual = solver.solve_from_atb(&atb).unwrap();
        elapsed_ns += started.elapsed().as_nanos();
        assert_eq!(
            solver.prepared.as_ref().unwrap().previous_weights,
            traced.iter().map(|value| *value as f32).collect::<Vec<_>>()
        );
        let matches = expected_manifest["records"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|candidate| {
                candidate["component"] == "weights"
                    && candidate["track"] == 0
                    && candidate["frame"] == record["frame"]
            })
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["timestamp"], record["timestamp"]);
        let expected = values(&expected_bytes, matches[0]);
        assert!(expected.len() >= actual.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(actual.is_finite() && expected.is_finite());
            let error = (actual - expected).abs();
            maximum = maximum.max(error);
            assert!(
                error <= 0.0005 + 0.0005 * actual.abs().max(expected.abs()),
                "frame {frame}, weight {index}: {actual} != {expected}"
            );
        }
        outputs.push(serde_json::json!({"frame":frame,"values":actual}));
    }
    assert!(!outputs.is_empty());
    let report = serde_json::json!({"frames":outputs.len(), "outputs":outputs,
        "max_weight_error":maximum,"cpu_solve_elapsed_ns":elapsed_ns,
        "svd_calls":svd_calls,"iterations":iterations,
        "scope":"production CPU preparation, cuBLAS RHS, own temporal history; CPU solve time excludes GPU, preparation and instrumentation"});
    std::fs::write(
        resolve(&manifest.output),
        serde_json::to_vec(&report).unwrap(),
    )
    .unwrap();
    eprintln!(
        "frames={}, max error={maximum}, SVD={svd_calls}, iterations={iterations}, CPU solve ns={elapsed_ns}",
        outputs.len()
    );
}
