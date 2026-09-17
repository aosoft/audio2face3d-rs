//! Repeatable payload-ownership measurements; timings are informational, not assertions.
use crate::inference::{audio::FrameBuffer, resample::Resampler};
#[test]
#[ignore = "manual buffer profiling; run with --nocapture"]
fn pcm_ownership_profile() {
    for aligned in [false, true] {
        let mut r = Resampler::new(16000, 600);
        let mut buffer = FrameBuffer::default();
        let mut received = 0usize;
        let mut copied = 0usize;
        let mut copied_out = 0usize;
        let mut copied_in = 0usize;
        let mut frames = 0;
        let start = std::time::Instant::now();
        for i in 0..1800 {
            let n = if aligned {
                ((i + 1) * 16000 / 30 - i * 16000 / 30) * 2
            } else {
                1120
            };
            let input = vec![7; n];
            let pointer = input.as_ptr();
            received += n;
            let output = r.push_owned(input).unwrap();
            if output.as_ptr() != pointer {
                copied += output.len();
            }
            let pointer = output.as_ptr();
            buffer.push_owned(output, 16000 * 600).unwrap();
            if buffer.first_pointer() != pointer {
                copied_in += n;
            }
            loop {
                let pointer = buffer.first_pointer();
                let Some((_, pcm)) = buffer.pop(false) else {
                    break;
                };
                if pcm.as_ptr() != pointer {
                    copied_out += pcm.len();
                }
                frames += 1;
            }
        }
        while let Some((_, pcm)) = buffer.pop(true) {
            copied_out += pcm.len();
            frames += 1;
        }
        println!(
            "PROFILE aligned={aligned} input_bytes={received} passthrough_copied_bytes={copied} queue_copied_bytes={copied_in} frame_repacked_bytes={copied_out} frames={frames} elapsed_us={}",
            start.elapsed().as_micros()
        );
    }
}

#[test]
fn passthrough_retains_allocation_and_limits_are_still_enforced() {
    let mut r = Resampler::new(16000, 1);
    let pcm = vec![0; 32000];
    let pointer = pcm.as_ptr();
    let output = r.push_owned(pcm).unwrap();
    assert_eq!(output.as_ptr(), pointer);
    assert_eq!(output.len(), 32000);
    assert!(r.push_owned(vec![0, 0]).is_err());
    assert!(r.finish().is_empty());
    let mut r = Resampler::new(16000, 1);
    assert!(r.push_owned(vec![0]).is_err());
    assert_eq!(r.push_owned(vec![1, 2]).unwrap(), [1, 2]);
}
#[test]
fn owned_frame_buffer_matches_borrowed_for_all_partitions() {
    for chunk in [2, 34, 1066, 1068, 1120, 32000] {
        let pcm: Vec<u8> = (0..32002).map(|i| (i % 251) as u8).collect();
        let mut owned = FrameBuffer::default();
        let mut borrowed = FrameBuffer::default();
        for bytes in pcm.chunks(chunk) {
            owned.push_owned(bytes.to_vec(), 32000).unwrap();
            borrowed.push(bytes, 32000).unwrap();
            loop {
                let a = owned.pop(false);
                let b = borrowed.pop(false);
                assert_eq!(a, b);
                if a.is_none() {
                    break;
                }
            }
        }
        loop {
            let a = owned.pop(true);
            let b = borrowed.pop(true);
            assert_eq!(a, b);
            if a.is_none() {
                break;
            }
        }
    }
    let bytes = vec![0; 1066];
    let pointer = bytes.as_ptr();
    let mut buffer = FrameBuffer::default();
    buffer.push_owned(bytes, 16000).unwrap();
    assert_eq!(buffer.pop(false).unwrap().1.as_ptr(), pointer);
}
#[test]
fn stack_normalization_matches_reference_including_wrapped_storage() {
    use std::collections::VecDeque;
    let mut bytes: VecDeque<u8> = (0..600)
        .flat_map(|i| (i as i16 * 31 - 9000).to_le_bytes())
        .collect();
    bytes.drain(..200);
    bytes.extend((0..100).flat_map(|i| (i as i16 * -19).to_le_bytes()));
    for offset in [0, 2, 200] {
        for count in [1, 17, 400] {
            let reference: Vec<f32> = bytes
                .iter()
                .skip(offset)
                .take(count * 2)
                .copied()
                .collect::<Vec<_>>()
                .chunks_exact(2)
                .map(|v| f32::from(i16::from_le_bytes([v[0], v[1]])) / 32768.0)
                .collect();
            let mut scratch = [0.0; 534];
            crate::inference::audio::normalized_samples(&bytes, offset, &mut scratch[..count]);
            assert_eq!(&scratch[..count], reference);
        }
    }
}
#[test]
fn identity_weights_move_and_permuted_weights_keep_clamping() {
    let weights = vec![-0.5, 0.75, 1.5];
    let pointer = weights.as_ptr();
    let weights = crate::inference::animation::ordered_weights(weights, &[0, 1, 2], true);
    assert_eq!(weights.as_ptr(), pointer);
    assert_eq!(weights, [0.0, 0.75, 1.0]);
    assert_eq!(
        crate::inference::animation::ordered_weights(vec![-0.5, 0.75, 1.5], &[2, 0, 1], false),
        [1.5, -0.5, 0.75]
    );
    assert_eq!(
        crate::inference::animation::ordered_weights(vec![-0.5, 0.75, 1.5], &[2, 0, 1], true),
        [1.0, 0.0, 0.75]
    );
}

#[test]
#[ignore = "manual native staging profile; no GPU required"]
fn native_staging_profile() {
    use std::{collections::VecDeque, hint::black_box, time::Instant};
    let pcm: VecDeque<u8> = (0..534i16).flat_map(|v| v.to_le_bytes()).collect();
    let iterations = 1800;
    let start = Instant::now();
    let mut old_sum = 0.0;
    for _ in 0..iterations {
        let bytes = black_box(&pcm).iter().copied().collect::<Vec<_>>();
        let samples = bytes
            .chunks_exact(2)
            .map(|v| f32::from(i16::from_le_bytes([v[0], v[1]])) / 32768.0)
            .collect::<Vec<_>>();
        old_sum += black_box(samples)[31];
    }
    let before = start.elapsed().as_micros();
    let start = Instant::now();
    let mut new_sum = 0.0;
    for _ in 0..iterations {
        let mut samples = [0.0; 534];
        crate::inference::audio::normalized_samples(black_box(&pcm), 0, &mut samples);
        new_sum += black_box(samples)[31];
    }
    assert_eq!(old_sum, new_sum);
    println!(
        "STAGING ticks={iterations} pcm_recopy_before={} pcm_recopy_after=0 heap_buffer_allocations_before={} heap_buffer_allocations_after=0 float_converted_bytes={} stack_bytes_per_tick={} before_us={before} after_us={}",
        iterations * 534 * 2,
        iterations * 2,
        iterations * 534 * 4,
        534 * 4,
        start.elapsed().as_micros()
    );
    let mut reused = 0;
    for _ in 0..iterations {
        let values = vec![0.5; 52];
        let ptr = values.as_ptr();
        let values = crate::inference::animation::ordered_weights(
            values,
            &(0..52).collect::<Vec<_>>(),
            true,
        );
        if values.as_ptr() == ptr {
            reused += 1;
        }
    }
    println!(
        "WEIGHTS frames={iterations} identity_reused={reused} avoided_payload_copy_bytes={} avoided_output_allocations={reused}; nonidentity still copies 208 bytes/frame",
        reused * 52 * 4
    );
}
