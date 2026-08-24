use crate::{
    EyesAnimator, JawParameters, JawTransform, RegressionGeometry, SkinAnimator, TongueAnimator,
};
use audio2x_core::{
    Audio2xError, Binding, BindingSchema, DiffusionAudioParameters, DiffusionParameters, Dimension,
    ElementType, IoMode, Result, Shape, WindowProgress, WindowProgressParameters,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiffusionResultLayout {
    pub skin: usize,
    pub tongue: usize,
    pub jaw: usize,
    pub eyes: usize,
}

impl DiffusionResultLayout {
    pub fn total(self) -> Result<usize> {
        self.skin
            .checked_add(self.tongue)
            .and_then(|value| value.checked_add(self.jaw))
            .and_then(|value| value.checked_add(self.eyes))
            .ok_or(Audio2xError::IntegerOverflow {
                field: "diffusion_result_size",
                value: self.eyes,
                target: "usize",
            })
    }

    pub fn split<'a>(self, values: &'a [f32]) -> Result<DiffusionResultSlices<'a>> {
        if values.len() != self.total()? {
            return Err(invalid("diffusion result length does not match layout"));
        }
        let (skin, rest) = values.split_at(self.skin);
        let (tongue, rest) = rest.split_at(self.tongue);
        let (jaw, eyes) = rest.split_at(self.jaw);
        Ok(DiffusionResultSlices {
            skin,
            tongue,
            jaw,
            eyes,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DiffusionResultSlices<'a> {
    pub skin: &'a [f32],
    pub tongue: &'a [f32],
    pub jaw: &'a [f32],
    pub eyes: &'a [f32],
}

#[derive(Debug, Clone)]
pub struct DiffusionContract {
    pub emotion_size: usize,
    pub identity_size: usize,
    pub audio_size: usize,
    pub diffusion_steps: usize,
    pub gru_layers: usize,
    pub gru_latent_size: usize,
    pub left_frames: usize,
    pub center_frames: usize,
    pub right_frames: usize,
    pub result_layout: DiffusionResultLayout,
    pub progress: WindowProgress,
    pub frame_progress: WindowProgress,
}

impl DiffusionContract {
    pub fn new(parameters: &DiffusionParameters, audio: &DiffusionAudioParameters) -> Result<Self> {
        let total_frames = parameters
            .num_frames_left_truncate
            .checked_add(parameters.num_frames_center)
            .and_then(|value| value.checked_add(parameters.num_frames_right_truncate))
            .ok_or(Audio2xError::IntegerOverflow {
                field: "diffusion_total_frames",
                value: parameters.num_frames_center,
                target: "usize",
            })?;
        if parameters.emotions.is_empty()
            || parameters.identities.is_empty()
            || parameters.num_frames_center == 0
            || parameters.num_diffusion_steps == 0
            || parameters.num_gru_layers == 0
            || parameters.gru_latent_dim == 0
            || !parameters.skin_size.is_multiple_of(3)
            || !parameters.tongue_size.is_multiple_of(3)
            || audio.buffer_len == 0
            || audio.samplerate == 0
        {
            return Err(invalid("invalid diffusion dimensions"));
        }
        let target_offset = audio
            .buffer_len
            .checked_mul(parameters.num_frames_left_truncate)
            .ok_or(Audio2xError::IntegerOverflow {
                field: "diffusion_target_offset",
                value: parameters.num_frames_left_truncate,
                target: "usize",
            })?
            / total_frames;
        let stride_numerator = audio
            .buffer_len
            .checked_mul(parameters.num_frames_center)
            .ok_or(Audio2xError::IntegerOverflow {
                field: "diffusion_stride",
                value: parameters.num_frames_center,
                target: "usize",
            })?;
        let progress_parameters = WindowProgressParameters {
            window_size: audio.buffer_len,
            start_offset: -i64::try_from(audio.padding_left).map_err(|_| {
                Audio2xError::IntegerOverflow {
                    field: "diffusion_padding_left",
                    value: audio.padding_left,
                    target: "i64",
                }
            })?,
            target_offset: i64::try_from(target_offset).map_err(|_| {
                Audio2xError::IntegerOverflow {
                    field: "diffusion_target_offset",
                    value: target_offset,
                    target: "i64",
                }
            })?,
            stride_numerator,
            stride_denominator: total_frames,
        };
        let mut frame_parameters = progress_parameters;
        frame_parameters.stride_denominator = frame_parameters
            .stride_denominator
            .checked_mul(parameters.num_frames_center)
            .ok_or(Audio2xError::IntegerOverflow {
                field: "diffusion_frame_stride",
                value: parameters.num_frames_center,
                target: "usize",
            })?;
        Ok(Self {
            emotion_size: parameters.emotions.len(),
            identity_size: parameters.identities.len(),
            audio_size: audio.buffer_len,
            diffusion_steps: parameters.num_diffusion_steps,
            gru_layers: parameters.num_gru_layers,
            gru_latent_size: parameters.gru_latent_dim,
            left_frames: parameters.num_frames_left_truncate,
            center_frames: parameters.num_frames_center,
            right_frames: parameters.num_frames_right_truncate,
            result_layout: DiffusionResultLayout {
                skin: parameters.skin_size,
                tongue: parameters.tongue_size,
                jaw: parameters.jaw_size,
                eyes: parameters.eyes_size,
            },
            progress: WindowProgress::new(progress_parameters)?,
            frame_progress: WindowProgress::new(frame_parameters)?,
        })
    }

    pub fn total_frames(&self) -> usize {
        self.left_frames + self.center_frames + self.right_frames
    }

    pub fn state_size(&self) -> Result<usize> {
        self.diffusion_steps
            .checked_mul(self.gru_layers)
            .and_then(|value| value.checked_mul(self.gru_latent_size))
            .ok_or(Audio2xError::IntegerOverflow {
                field: "diffusion_state_size",
                value: self.gru_latent_size,
                target: "usize",
            })
    }

    pub fn noise_size(&self) -> Result<usize> {
        (self.diffusion_steps + 1)
            .checked_mul(self.total_frames())
            .and_then(|value| value.checked_mul(self.result_layout.total().ok()?))
            .ok_or(Audio2xError::IntegerOverflow {
                field: "diffusion_noise_size",
                value: self.diffusion_steps,
                target: "usize",
            })
    }

    pub fn schema(&self) -> Result<BindingSchema> {
        let batch = Dimension::Batch;
        let fixed = |value: usize| -> Result<Dimension> { Ok(Dimension::Fixed(value)) };
        BindingSchema::new(vec![
            binding(
                "emotion",
                IoMode::Input,
                vec![batch, fixed(self.center_frames)?, fixed(self.emotion_size)?],
            )?,
            binding(
                "identity",
                IoMode::Input,
                vec![batch, fixed(self.identity_size)?],
            )?,
            binding(
                "input_latents",
                IoMode::Input,
                vec![
                    fixed(self.diffusion_steps)?,
                    fixed(self.gru_layers)?,
                    batch,
                    fixed(self.gru_latent_size)?,
                ],
            )?,
            binding(
                "noise",
                IoMode::Input,
                vec![
                    batch,
                    fixed(self.diffusion_steps + 1)?,
                    fixed(self.total_frames())?,
                    fixed(self.result_layout.total()?)?,
                ],
            )?,
            binding(
                "window",
                IoMode::Input,
                vec![batch, fixed(self.audio_size)?],
            )?,
            binding(
                "output_latents",
                IoMode::Output,
                vec![
                    fixed(self.diffusion_steps)?,
                    fixed(self.gru_layers)?,
                    batch,
                    fixed(self.gru_latent_size)?,
                ],
            )?,
            binding(
                "prediction",
                IoMode::Output,
                vec![
                    batch,
                    fixed(self.total_frames())?,
                    fixed(self.result_layout.total()?)?,
                ],
            )?,
        ])
    }
}

fn binding(name: &str, mode: IoMode, dimensions: Vec<Dimension>) -> Result<Binding> {
    Ok(Binding {
        name: name.into(),
        mode,
        element_type: ElementType::F32,
        shape: Shape::new(dimensions)?,
    })
}

#[derive(Debug, Clone)]
pub struct DiffusionFrameInput {
    pub audio: Vec<f32>,
    pub emotions: Vec<f32>,
    pub identity: Vec<f32>,
    pub noise: Vec<f32>,
    pub input_latents: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiffusionInferenceOutput {
    pub output_latents: Vec<f32>,
    /// All predicted frames, including left/right truncation, in frame-major order.
    pub prediction: Vec<f32>,
}

/// Host parity model of the SDK's interleaved recurrent state buffers.
#[derive(Debug, Clone)]
pub struct DiffusionState {
    input: Vec<f32>,
    output: Vec<f32>,
    tracks: usize,
    slices: usize,
    latent_size: usize,
}

impl DiffusionState {
    pub fn new(contract: &DiffusionContract, tracks: usize) -> Result<Self> {
        if tracks == 0 {
            return Err(invalid("diffusion state requires at least one track"));
        }
        let slices = contract.diffusion_steps * contract.gru_layers;
        let size = slices
            .checked_mul(tracks)
            .and_then(|value| value.checked_mul(contract.gru_latent_size))
            .ok_or(Audio2xError::IntegerOverflow {
                field: "diffusion_batched_state",
                value: tracks,
                target: "usize",
            })?;
        Ok(Self {
            input: vec![0.0; size],
            output: vec![0.0; size],
            tracks,
            slices,
            latent_size: contract.gru_latent_size,
        })
    }

    pub fn input(&self) -> &[f32] {
        &self.input
    }

    pub fn track_input(&self, track: usize) -> Result<Vec<f32>> {
        if track >= self.tracks {
            return Err(invalid("diffusion state track is out of range"));
        }
        let mut output = Vec::with_capacity(self.slices * self.latent_size);
        for slice in 0..self.slices {
            let offset = (slice * self.tracks + track) * self.latent_size;
            output.extend_from_slice(&self.input[offset..offset + self.latent_size]);
        }
        Ok(output)
    }

    pub fn set_output(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.output.len() {
            return Err(invalid("diffusion output state length mismatch"));
        }
        self.output.copy_from_slice(values);
        Ok(())
    }

    /// Swaps inference state and restores non-executed tracks from the old input.
    pub fn commit(&mut self, executed: &[bool]) -> Result<()> {
        if executed.len() != self.tracks {
            return Err(invalid("diffusion executed-track mask length mismatch"));
        }
        std::mem::swap(&mut self.input, &mut self.output);
        for (track, was_executed) in executed.iter().copied().enumerate() {
            if was_executed {
                continue;
            }
            for slice in 0..self.slices {
                let offset = (slice * self.tracks + track) * self.latent_size;
                self.input[offset..offset + self.latent_size]
                    .copy_from_slice(&self.output[offset..offset + self.latent_size]);
            }
        }
        Ok(())
    }

    pub fn commit_tracks(&mut self, outputs: &[(usize, &[f32])]) -> Result<()> {
        self.output.copy_from_slice(&self.input);
        let per_track = self.slices * self.latent_size;
        for &(track, values) in outputs {
            if track >= self.tracks || values.len() != per_track {
                return Err(invalid("diffusion track output state dimensions mismatch"));
            }
            for slice in 0..self.slices {
                let source = slice * self.latent_size;
                let target = (slice * self.tracks + track) * self.latent_size;
                self.output[target..target + self.latent_size]
                    .copy_from_slice(&values[source..source + self.latent_size]);
            }
        }
        std::mem::swap(&mut self.input, &mut self.output);
        Ok(())
    }

    pub fn reset(&mut self, track: usize) -> Result<()> {
        if track >= self.tracks {
            return Err(invalid("diffusion state track is out of range"));
        }
        for slice in 0..self.slices {
            let offset = (slice * self.tracks + track) * self.latent_size;
            self.input[offset..offset + self.latent_size].fill(0.0);
        }
        Ok(())
    }
}

/// Deterministic Philox4x32-10 noise source with one independent stream per track.
#[derive(Debug, Clone)]
pub struct PhiloxNoise {
    seed: u64,
    offsets: Vec<u64>,
    size: usize,
}

impl PhiloxNoise {
    pub fn new(tracks: usize, size: usize, seed: u64) -> Result<Self> {
        if tracks == 0 || size == 0 || !size.is_multiple_of(2) {
            return Err(invalid(
                "Philox track count and even noise size must be non-zero",
            ));
        }
        Ok(Self {
            seed,
            offsets: vec![0; tracks],
            size,
        })
    }

    pub fn generate(&mut self, track: usize) -> Result<Vec<f32>> {
        let offset = *self
            .offsets
            .get(track)
            .ok_or_else(|| invalid("Philox track is out of range"))?;
        let mut output = Vec::with_capacity(self.size);
        for pair in 0..self.size / 2 {
            let counter = offset + u64::try_from(pair).unwrap_or(u64::MAX);
            let words = philox4x32_10(counter, track as u64, self.seed);
            let u1 = unit_open(words[0]);
            let u2 = unit_open(words[1]);
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = std::f32::consts::TAU * u2;
            output.push(radius * angle.cos());
            output.push(radius * angle.sin());
        }
        self.offsets[track] = offset
            .checked_add(u64::try_from(self.size / 2).map_err(|_| {
                Audio2xError::IntegerOverflow {
                    field: "philox_offset",
                    value: self.size,
                    target: "u64",
                }
            })?)
            .ok_or_else(|| invalid("Philox offset overflow"))?;
        Ok(output)
    }

    pub fn reset(&mut self, track: usize, generate_index: usize) -> Result<()> {
        let offset = self
            .offsets
            .get_mut(track)
            .ok_or_else(|| invalid("Philox track is out of range"))?;
        *offset = u64::try_from(generate_index)
            .ok()
            .and_then(|index| index.checked_mul((self.size / 2) as u64))
            .ok_or_else(|| invalid("Philox reset offset overflow"))?;
        Ok(())
    }
}

fn unit_open(word: u32) -> f32 {
    (word as f32 + 1.0) * (1.0 / 4_294_967_296.0)
}

fn philox4x32_10(counter: u64, subsequence: u64, seed: u64) -> [u32; 4] {
    let mut values = [
        counter as u32,
        (counter >> 32) as u32,
        subsequence as u32,
        (subsequence >> 32) as u32,
    ];
    let mut key = [seed as u32, (seed >> 32) as u32];
    for _ in 0..10 {
        let left = 0xd251_1f53_u64 * u64::from(values[0]);
        let right = 0xcd9e_8d57_u64 * u64::from(values[2]);
        values = [
            (right >> 32) as u32 ^ values[1] ^ key[0],
            right as u32,
            (left >> 32) as u32 ^ values[3] ^ key[1],
            left as u32,
        ];
        key[0] = key[0].wrapping_add(0x9e37_79b9);
        key[1] = key[1].wrapping_add(0xbb67_ae85);
    }
    values
}

#[derive(Debug, Clone)]
pub struct DiffusionPostprocessor {
    layout: DiffusionResultLayout,
    skin: SkinAnimator,
    tongue: TongueAnimator,
    jaw: JawTransform,
    jaw_parameters: JawParameters,
    eyes: EyesAnimator,
}

impl DiffusionPostprocessor {
    pub fn new(
        layout: DiffusionResultLayout,
        skin: SkinAnimator,
        tongue: TongueAnimator,
        jaw: JawTransform,
        jaw_parameters: JawParameters,
        eyes: EyesAnimator,
    ) -> Result<Self> {
        if layout.skin != skin.neutral_pose().len()
            || layout.tongue != tongue.neutral_pose().len()
            || layout.jaw != jaw.neutral_pose().len()
            || layout.eyes != 4
        {
            return Err(invalid("diffusion postprocessor dimensions do not match"));
        }
        Ok(Self {
            layout,
            skin,
            tongue,
            jaw,
            jaw_parameters,
            eyes,
        })
    }

    pub fn process(&mut self, prediction: &[f32], dt: f32) -> Result<RegressionGeometry> {
        let result = self.layout.split(prediction)?;
        let eyes: [f32; 4] = result
            .eyes
            .try_into()
            .map_err(|_| invalid("diffusion eyes output must contain four values"))?;
        let geometry = RegressionGeometry {
            skin: self.skin.animate(result.skin, dt)?,
            tongue: self.tongue.animate(result.tongue)?,
            jaw_transform: self.jaw.compute(result.jaw, self.jaw_parameters)?,
            eyes_rotation: self.eyes.compute_rotation(eyes),
        };
        self.eyes.increment_live_time(dt)?;
        Ok(geometry)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.skin.reset();
        self.eyes.reset()
    }
}

fn invalid(message: impl Into<String>) -> Audio2xError {
    Audio2xError::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EyesAnimatorParams, SkinAnimatorParams, TongueAnimatorParams};

    fn parameters() -> (DiffusionParameters, DiffusionAudioParameters) {
        (
            DiffusionParameters {
                emotions: vec!["joy".into(), "anger".into()],
                default_emotion: vec![0.0, 0.0],
                identities: vec!["one".into(), "two".into()],
                skin_size: 3,
                tongue_size: 3,
                jaw_size: 9,
                eyes_size: 4,
                num_diffusion_steps: 2,
                num_gru_layers: 2,
                gru_latent_dim: 3,
                num_frames_left_truncate: 1,
                num_frames_right_truncate: 1,
                num_frames_center: 2,
            },
            DiffusionAudioParameters {
                buffer_len: 8,
                padding_left: 8,
                padding_right: 8,
                samplerate: 8,
            },
        )
    }

    #[test]
    fn schema_and_multi_frame_progress_follow_sdk_layout() {
        let (parameters, audio) = parameters();
        let contract = DiffusionContract::new(&parameters, &audio).unwrap();
        assert_eq!(contract.schema().unwrap().bindings().len(), 7);
        assert_eq!(contract.progress.window(0).unwrap().start, -8);
        assert_eq!(contract.progress.window(0).unwrap().target, -6);
        assert_eq!(contract.progress.window(1).unwrap().target, -2);
        assert_eq!(contract.frame_progress.window(2).unwrap().target, -2);
        assert_eq!(contract.state_size().unwrap(), 12);
        assert_eq!(contract.noise_size().unwrap(), 228);
    }

    #[test]
    fn state_swap_preserves_skipped_tracks_and_reset_is_isolated() {
        let (parameters, audio) = parameters();
        let contract = DiffusionContract::new(&parameters, &audio).unwrap();
        let mut state = DiffusionState::new(&contract, 2).unwrap();
        state.input.fill(1.0);
        state.set_output(&vec![2.0; state.output.len()]).unwrap();
        state.commit(&[true, false]).unwrap();
        for slice in 0..4 {
            let first = (slice * 2) * 3;
            let second = first + 3;
            assert_eq!(&state.input[first..first + 3], &[2.0; 3]);
            assert_eq!(&state.input[second..second + 3], &[1.0; 3]);
        }
        state.reset(0).unwrap();
        assert_eq!(
            state.input.iter().filter(|value| **value == 0.0).count(),
            12
        );
    }

    #[test]
    fn philox_reset_replays_and_tracks_are_independent() {
        assert!(PhiloxNoise::new(1, 3, 7).is_err());
        let mut noise = PhiloxNoise::new(2, 10_000, 7).unwrap();
        let first = noise.generate(0).unwrap();
        let other = noise.generate(1).unwrap();
        assert_ne!(first, other);
        noise.reset(0, 0).unwrap();
        assert_eq!(noise.generate(0).unwrap(), first);
        let mean = first.iter().sum::<f32>() / first.len() as f32;
        let variance = first
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f32>()
            / first.len() as f32;
        assert!(mean.abs() < 0.05, "mean={mean}");
        assert!((variance - 1.0).abs() < 0.08, "variance={variance}");
    }

    #[test]
    fn postprocess_connects_all_geometry_outputs() {
        let layout = DiffusionResultLayout {
            skin: 3,
            tongue: 3,
            jaw: 9,
            eyes: 4,
        };
        let skin = SkinAnimator::new(
            SkinAnimatorParams {
                lower_face_smoothing: 0.0,
                upper_face_smoothing: 0.0,
                lower_face_strength: 1.0,
                upper_face_strength: 1.0,
                face_mask_level: 0.5,
                face_mask_softness: 0.1,
                skin_strength: 1.0,
                blink_strength: 0.0,
                eyelid_open_offset: 0.0,
                lip_open_offset: 0.0,
                blink_offset: 0.0,
            },
            vec![0.0; 3],
            vec![0.0; 3],
            vec![0.0; 3],
        )
        .unwrap();
        let tongue = TongueAnimator::new(
            TongueAnimatorParams {
                tongue_strength: 2.0,
                tongue_height_offset: 1.0,
                tongue_depth_offset: 2.0,
            },
            vec![0.0; 3],
        )
        .unwrap();
        let jaw = JawTransform::new(vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
        let eyes = EyesAnimator::new(
            EyesAnimatorParams {
                eyeballs_strength: 1.0,
                saccade_strength: 0.0,
                right_eyeball_rotation_offset_x: 0.0,
                right_eyeball_rotation_offset_y: 0.0,
                left_eyeball_rotation_offset_x: 0.0,
                left_eyeball_rotation_offset_y: 0.0,
                saccade_seed: 0.0,
            },
            vec![0.0, 0.0],
        )
        .unwrap();
        let mut postprocessor =
            DiffusionPostprocessor::new(layout, skin, tongue, jaw, JawParameters::default(), eyes)
                .unwrap();
        let geometry = postprocessor
            .process(
                &[
                    1.0, 2.0, 3.0, // skin
                    1.0, 2.0, 3.0, // tongue
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // jaw
                    1.0, 2.0, 3.0, 4.0, // eyes
                ],
                1.0 / 30.0,
            )
            .unwrap();
        assert_eq!(geometry.skin, [1.0, 2.0, 3.0]);
        assert_eq!(geometry.tongue, [2.0, 5.0, 8.0]);
        assert_eq!(geometry.eyes_rotation.right, [1.0, 2.0, 0.0]);
        assert_eq!(geometry.eyes_rotation.left, [3.0, 4.0, 0.0]);
        postprocessor.reset().unwrap();
    }
}
