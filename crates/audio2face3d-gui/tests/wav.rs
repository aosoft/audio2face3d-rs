#![cfg(feature = "session")]
use audio2face3d_gui::wav;
#[test]
fn pcm16_native_rate_preserves_exact_samples() {
    let path = std::env::temp_dir().join(format!("a2f-pcm-{}.wav", std::process::id()));
    let mut writer = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )
    .unwrap();
    let samples = [i16::MIN, -17000, -1, 0, 1, 17000, i16::MAX];
    for sample in samples {
        writer.write_sample(sample).unwrap();
    }
    writer.finalize().unwrap();
    assert_eq!(
        wav::load(&path).unwrap(),
        samples
            .into_iter()
            .flat_map(i16::to_le_bytes)
            .collect::<Vec<_>>()
    );
    std::fs::remove_file(path).unwrap();
}
#[test]
fn stereo_downmix_resamples_and_filters_aliases() {
    let path = std::env::temp_dir().join(format!("a2f-wav-{}.wav", std::process::id()));
    for frequency in [1000., 12000.] {
        let mut writer = hound::WavWriter::create(
            &path,
            hound::WavSpec {
                channels: 2,
                sample_rate: 48000,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            },
        )
        .unwrap();
        for i in 0..4800 {
            let value =
                (12000. * (std::f64::consts::TAU * frequency * i as f64 / 48000.).sin()) as i16;
            writer.write_sample(value).unwrap();
            writer.write_sample(value).unwrap();
        }
        writer.finalize().unwrap();
        let data = wav::load(&path).unwrap();
        assert_eq!(data.len(), 3200);
        let rms = (data
            .chunks_exact(2)
            .skip(100)
            .take(1400)
            .map(|v| (i16::from_le_bytes([v[0], v[1]]) as f64).powi(2))
            .sum::<f64>()
            / 1400.)
            .sqrt();
        if frequency == 1000. {
            assert!(rms > 7000.);
        } else {
            assert!(rms < 100.);
        }
    }
    std::fs::remove_file(path).unwrap();
}
#[test]
fn rejects_nonfinite_float_wav() {
    let path = std::env::temp_dir().join(format!("a2f-nan-{}.wav", std::process::id()));
    let mut writer = hound::WavWriter::create(
        &path,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    writer.write_sample(f32::NAN).unwrap();
    writer.finalize().unwrap();
    assert!(wav::load(&path).is_err());
    std::fs::remove_file(path).unwrap();
}
