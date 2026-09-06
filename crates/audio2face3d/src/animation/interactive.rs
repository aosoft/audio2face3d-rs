#![cfg_attr(not(feature = "tensorrt"), allow(dead_code))]

//! Offline random-access geometry execution with layer invalidation.

use crate::animation::{
    DiffusionBackend, DiffusionContract, DiffusionFrameInput, DiffusionInferenceOutput,
    LayeredGeometryPostprocessor, PhiloxNoise, RegressionBackend, RegressionContract,
    RegressionGeometry,
};
#[cfg(test)]
use crate::animation::{
    EyesAnimatorParams, JawParameters, SkinAnimatorParams, TongueAnimatorParams,
};
use crate::common::{AudioAccumulator, EmotionAccumulator, Error, Result};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum GeometryInvalidationLayer {
    None = 0,
    All = 1,
    Inference = 2,
    Skin = 3,
    Tongue = 4,
    Teeth = 5,
    Eyes = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InteractiveGeometryMetadata {
    pub frame: usize,
    pub inference: Option<usize>,
    pub timestamp: i64,
    pub next_timestamp: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractiveGeometryStatus {
    Complete { frames: usize, inferences: usize },
    Interrupted { frames: usize, inferences: usize },
}

#[derive(Clone, Debug, Default)]
struct GeometryFrameExecutionState {
    inference: Option<Vec<f32>>,
    skin: Option<Vec<f32>>,
    tongue: Option<Vec<f32>>,
    teeth: Option<[f32; 16]>,
    eyes: Option<crate::animation::EyesRotation>,
}

impl GeometryFrameExecutionState {
    fn clear_geometry(&mut self) {
        self.skin = None;
        self.tongue = None;
        self.teeth = None;
        self.eyes = None;
    }

    fn clear(&mut self, layer: GeometryInvalidationLayer) {
        match layer {
            GeometryInvalidationLayer::None => {}
            GeometryInvalidationLayer::All | GeometryInvalidationLayer::Inference => {
                self.inference = None;
                self.clear_geometry();
            }
            GeometryInvalidationLayer::Skin => self.skin = None,
            GeometryInvalidationLayer::Tongue => self.tongue = None,
            GeometryInvalidationLayer::Teeth => self.teeth = None,
            GeometryInvalidationLayer::Eyes => self.eyes = None,
        }
    }

    fn frame(&self) -> Result<RegressionGeometry> {
        Ok(RegressionGeometry {
            skin: self
                .skin
                .clone()
                .ok_or_else(|| invalid("interactive skin layer is incomplete"))?,
            tongue: self
                .tongue
                .clone()
                .ok_or_else(|| invalid("interactive tongue layer is incomplete"))?,
            jaw_transform: self
                .teeth
                .ok_or_else(|| invalid("interactive teeth layer is incomplete"))?,
            eyes_rotation: self
                .eyes
                .ok_or_else(|| invalid("interactive eyes layer is incomplete"))?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct InteractiveGeometryInterrupt {
    requested: Arc<AtomicBool>,
}

impl InteractiveGeometryInterrupt {
    pub fn interrupt(&self) {
        self.requested.store(true, Ordering::Release);
    }

    pub fn is_interrupted(&self) -> bool {
        self.requested.load(Ordering::Acquire)
    }
}

/// Internal single-track random-access Regression execution.
pub(crate) struct RegressionGeometryInteractiveExecution<B, P> {
    backend: B,
    contract: RegressionContract,
    postprocessor: P,
    audio: Arc<AudioAccumulator>,
    emotions: Arc<EmotionAccumulator>,
    implicit_emotion: Vec<f32>,
    input_strength: f32,
    frames: Vec<GeometryFrameExecutionState>,
    input_signature: Option<(usize, usize)>,
    interrupted: Arc<AtomicBool>,
}

impl<B, P> RegressionGeometryInteractiveExecution<B, P>
where
    B: RegressionBackend<Output = Vec<f32>>,
    P: LayeredGeometryPostprocessor,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: B,
        contract: RegressionContract,
        postprocessor: P,
        audio: Arc<AudioAccumulator>,
        emotions: Arc<EmotionAccumulator>,
        implicit_emotion: Vec<f32>,
        input_strength: f32,
    ) -> Result<Self> {
        if implicit_emotion.len() != contract.implicit_emotion_size {
            return Err(invalid("interactive implicit emotion dimensions differ"));
        }
        finite_strength(input_strength)?;
        Ok(Self {
            backend,
            contract,
            postprocessor,
            audio,
            emotions,
            implicit_emotion,
            input_strength,
            frames: Vec::new(),
            input_signature: None,
            interrupted: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn interrupt_handle(&self) -> InteractiveGeometryInterrupt {
        InteractiveGeometryInterrupt {
            requested: Arc::clone(&self.interrupted),
        }
    }

    #[cfg(feature = "tensorrt")]
    pub(crate) fn prepare_all(&mut self) -> Result<()> {
        self.begin_compute()?;
        self.postprocessor.reset_layers()?;
        for frame in &mut self.frames {
            frame.clear_geometry();
        }
        Ok(())
    }

    #[cfg(feature = "tensorrt")]
    pub(crate) fn compute_frame_stateful<C>(
        &mut self,
        frame: usize,
        mut callback: C,
    ) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &RegressionGeometry) -> bool,
    {
        self.begin_compute()?;
        if frame >= self.frames.len() {
            return Err(invalid("interactive regression frame is out of range"));
        }
        let inferences = usize::from(self.ensure_inference(frame)?);
        self.materialize(frame, false)?;
        let output = self.frames[frame].frame()?;
        let metadata = self.metadata(frame)?;
        if callback(metadata, &output) {
            Ok(InteractiveGeometryStatus::Complete {
                frames: 1,
                inferences,
            })
        } else {
            Ok(InteractiveGeometryStatus::Interrupted {
                frames: 1,
                inferences,
            })
        }
    }

    pub fn total_frames(&self) -> Result<usize> {
        validate_inputs(
            &self.audio,
            &self.emotions,
            self.contract.explicit_emotion_size,
        )?;
        self.contract.progress.available_windows(
            i64::try_from(self.audio.nb_accumulated_samples()).unwrap_or(i64::MAX),
            true,
        )
    }

    pub fn sampling_rate(&self) -> usize {
        self.contract.sample_rate
    }

    pub fn frame_rate(&self) -> (usize, usize) {
        (
            self.contract.frame_rate_numerator,
            self.contract.frame_rate_denominator,
        )
    }

    pub fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        if frame >= self.total_frames()? {
            return Err(invalid("interactive regression frame is out of range"));
        }
        Ok(self.contract.progress.window(frame)?.target)
    }

    pub fn invalidate(&mut self, layer: GeometryInvalidationLayer) {
        for frame in &mut self.frames {
            frame.clear(layer);
        }
        if matches!(
            layer,
            GeometryInvalidationLayer::All | GeometryInvalidationLayer::Inference
        ) {
            self.input_signature = None;
        }
    }

    pub fn is_valid(&self, layer: GeometryInvalidationLayer) -> bool {
        layer_valid(&self.frames, layer)
    }

    pub fn set_input_strength(&mut self, value: f32) -> Result<()> {
        finite_strength(value)?;
        if self.input_strength != value {
            self.input_strength = value;
            self.invalidate(GeometryInvalidationLayer::Inference);
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn skin_parameters(&self) -> SkinAnimatorParams {
        self.postprocessor.skin_parameters()
    }

    #[cfg(test)]
    pub fn set_skin_parameters(&mut self, value: SkinAnimatorParams) -> Result<()> {
        if self.postprocessor.skin_parameters() != value {
            self.postprocessor.set_skin_parameters(value)?;
            self.invalidate(GeometryInvalidationLayer::Skin);
        }
        Ok(())
    }

    pub fn compute_frame<C>(
        &mut self,
        frame: usize,
        mut callback: C,
    ) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &RegressionGeometry) -> bool,
    {
        self.begin_compute()?;
        if frame >= self.frames.len() {
            return Err(invalid("interactive regression frame is out of range"));
        }
        let inferences = usize::from(self.ensure_inference(frame)?);
        self.materialize(frame, true)?;
        let output = self.frames[frame].frame()?;
        let metadata = self.metadata(frame)?;
        let frames = 1;
        if callback(metadata, &output) {
            Ok(InteractiveGeometryStatus::Complete { frames, inferences })
        } else {
            Ok(InteractiveGeometryStatus::Interrupted { frames, inferences })
        }
    }

    #[cfg(test)]
    pub fn compute_all_frames<C>(&mut self, mut callback: C) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &RegressionGeometry) -> bool,
    {
        self.begin_compute()?;
        self.postprocessor.reset_layers()?;
        for frame in &mut self.frames {
            frame.clear_geometry();
        }
        let mut emitted = 0;
        let mut inferences = 0;
        for frame in 0..self.frames.len() {
            if self.interrupted.load(Ordering::Acquire) {
                return Ok(InteractiveGeometryStatus::Interrupted {
                    frames: emitted,
                    inferences,
                });
            }
            inferences += usize::from(self.ensure_inference(frame)?);
            self.materialize(frame, false)?;
            let output = self.frames[frame].frame()?;
            emitted += 1;
            if !callback(self.metadata(frame)?, &output) {
                return Ok(InteractiveGeometryStatus::Interrupted {
                    frames: emitted,
                    inferences,
                });
            }
        }
        Ok(InteractiveGeometryStatus::Complete {
            frames: emitted,
            inferences,
        })
    }

    fn begin_compute(&mut self) -> Result<()> {
        validate_inputs(
            &self.audio,
            &self.emotions,
            self.contract.explicit_emotion_size,
        )?;
        self.interrupted.store(false, Ordering::Release);
        let total = self.total_frames()?;
        let signature = (
            self.audio.nb_accumulated_samples(),
            self.emotions.state().key_count,
        );
        if self.input_signature.is_some_and(|value| value != signature) {
            self.invalidate(GeometryInvalidationLayer::Inference);
        }
        if self.frames.len() != total {
            self.frames
                .resize_with(total, GeometryFrameExecutionState::default);
            self.invalidate(GeometryInvalidationLayer::Inference);
        }
        self.input_signature = Some(signature);
        Ok(())
    }

    fn ensure_inference(&mut self, frame: usize) -> Result<bool> {
        if self.frames[frame].inference.is_some() {
            return Ok(false);
        }
        let input = self.contract.prepare_frame(
            frame,
            &self.audio,
            &self.emotions,
            &self.implicit_emotion,
            self.input_strength,
        )?;
        let output = self.backend.infer(0, &input)?;
        self.contract.result_layout.split(&output)?;
        self.frames[frame].inference = Some(output);
        Ok(true)
    }

    fn materialize(&mut self, frame: usize, stateless: bool) -> Result<()> {
        let dt =
            self.contract.frame_rate_denominator as f32 / self.contract.frame_rate_numerator as f32;
        let live_time = frame as f32 * dt;
        let cache = &mut self.frames[frame];
        let inference = cache
            .inference
            .as_deref()
            .ok_or_else(|| invalid("interactive regression inference is missing"))?;
        if cache.skin.is_none() {
            cache.skin = Some(self.postprocessor.process_skin(inference, dt, stateless)?);
        }
        if cache.tongue.is_none() {
            cache.tongue = Some(self.postprocessor.process_tongue(inference)?);
        }
        if cache.teeth.is_none() {
            cache.teeth = Some(self.postprocessor.process_teeth(inference)?);
        }
        if cache.eyes.is_none() {
            cache.eyes = Some(self.postprocessor.process_eyes(inference, live_time)?);
        }
        Ok(())
    }

    fn metadata(&self, frame: usize) -> Result<InteractiveGeometryMetadata> {
        Ok(InteractiveGeometryMetadata {
            frame,
            inference: None,
            timestamp: self.contract.progress.window(frame)?.target,
            next_timestamp: self.contract.progress.window(frame + 1)?.target,
        })
    }
}

/// Internal single-track random-access Diffusion execution with GRU checkpoints.
pub(crate) struct DiffusionGeometryInteractiveExecution<B, P> {
    backend: B,
    contract: DiffusionContract,
    postprocessor: P,
    audio: Arc<AudioAccumulator>,
    emotions: Arc<EmotionAccumulator>,
    identity_index: usize,
    input_strength: f32,
    preview_inferences: usize,
    seed: u64,
    frames_before_audio: usize,
    frames: Vec<GeometryFrameExecutionState>,
    checkpoints: Vec<Option<Vec<f32>>>,
    exact_checkpoints: bool,
    input_signature: Option<(usize, usize, usize)>,
    interrupted: Arc<AtomicBool>,
}

impl<B, P> DiffusionGeometryInteractiveExecution<B, P>
where
    B: DiffusionBackend,
    P: LayeredGeometryPostprocessor,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        backend: B,
        contract: DiffusionContract,
        postprocessor: P,
        audio: Arc<AudioAccumulator>,
        emotions: Arc<EmotionAccumulator>,
        identity_index: usize,
        input_strength: f32,
        preview_inferences: usize,
        seed: u64,
    ) -> Result<Self> {
        if identity_index >= contract.identity_size {
            return Err(invalid("interactive diffusion identity is out of range"));
        }
        finite_strength(input_strength)?;
        let frames_before_audio = frames_before_audio(&contract)?;
        Ok(Self {
            backend,
            contract,
            postprocessor,
            audio,
            emotions,
            identity_index,
            input_strength,
            preview_inferences,
            seed,
            frames_before_audio,
            frames: Vec::new(),
            checkpoints: Vec::new(),
            exact_checkpoints: true,
            input_signature: None,
            interrupted: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn interrupt_handle(&self) -> InteractiveGeometryInterrupt {
        InteractiveGeometryInterrupt {
            requested: Arc::clone(&self.interrupted),
        }
    }

    #[cfg(test)]
    pub fn set_preview_inferences(&mut self, value: usize) {
        if self.preview_inferences != value {
            self.preview_inferences = value;
            self.invalidate(GeometryInvalidationLayer::Inference);
        }
    }

    pub fn total_frames(&self) -> Result<usize> {
        validate_inputs(&self.audio, &self.emotions, self.contract.emotion_size)?;
        let available = self.contract.frame_progress.available_windows(
            i64::try_from(self.audio.nb_accumulated_samples()).unwrap_or(i64::MAX),
            true,
        )?;
        Ok(available.saturating_sub(self.frames_before_audio))
    }

    pub fn sampling_rate(&self) -> usize {
        self.contract.sample_rate
    }

    pub fn frame_timestamp(&self, frame: usize) -> Result<i64> {
        if frame >= self.total_frames()? {
            return Err(invalid("interactive diffusion frame is out of range"));
        }
        Ok(self
            .contract
            .frame_progress
            .window(self.internal_frame(frame)?)?
            .target)
    }

    pub fn invalidate(&mut self, layer: GeometryInvalidationLayer) {
        for frame in &mut self.frames {
            frame.clear(layer);
        }
        if matches!(
            layer,
            GeometryInvalidationLayer::All | GeometryInvalidationLayer::Inference
        ) {
            self.checkpoints.clear();
            self.exact_checkpoints = true;
            self.input_signature = None;
        }
    }

    pub fn is_valid(&self, layer: GeometryInvalidationLayer) -> bool {
        layer_valid(&self.frames, layer)
            && (layer != GeometryInvalidationLayer::Inference || self.exact_checkpoints)
    }

    pub fn set_input_strength(&mut self, value: f32) -> Result<()> {
        finite_strength(value)?;
        if self.input_strength != value {
            self.input_strength = value;
            self.invalidate(GeometryInvalidationLayer::Inference);
        }
        Ok(())
    }

    pub fn compute_frame<C>(
        &mut self,
        frame: usize,
        mut callback: C,
    ) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &RegressionGeometry) -> bool,
    {
        self.begin_compute()?;
        if frame >= self.frames.len() {
            return Err(invalid("interactive diffusion frame is out of range"));
        }
        let inferences = self.ensure_inference_for_frame(frame, false)?;
        if self.interrupted.load(Ordering::Acquire) && self.frames[frame].inference.is_none() {
            return Ok(InteractiveGeometryStatus::Interrupted {
                frames: 0,
                inferences,
            });
        }
        self.materialize(frame, true)?;
        let output = self.frames[frame].frame()?;
        if callback(self.metadata(frame)?, &output) {
            Ok(InteractiveGeometryStatus::Complete {
                frames: 1,
                inferences,
            })
        } else {
            Ok(InteractiveGeometryStatus::Interrupted {
                frames: 1,
                inferences,
            })
        }
    }

    #[cfg(feature = "tensorrt")]
    pub(crate) fn prepare_all(&mut self) -> Result<()> {
        self.begin_compute()?;
        if !self.exact_checkpoints {
            self.invalidate(GeometryInvalidationLayer::Inference);
            self.begin_compute()?;
        }
        self.postprocessor.reset_layers()?;
        for frame in &mut self.frames {
            frame.clear_geometry();
        }
        Ok(())
    }

    #[cfg(feature = "tensorrt")]
    pub(crate) fn compute_frame_stateful<C>(
        &mut self,
        frame: usize,
        mut callback: C,
    ) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &RegressionGeometry) -> bool,
    {
        self.begin_compute()?;
        if frame >= self.frames.len() {
            return Err(invalid("interactive diffusion frame is out of range"));
        }
        let inferences = self.ensure_inference_for_frame(frame, true)?;
        if self.interrupted.load(Ordering::Acquire) && self.frames[frame].inference.is_none() {
            return Ok(InteractiveGeometryStatus::Interrupted {
                frames: 0,
                inferences,
            });
        }
        self.materialize(frame, false)?;
        let output = self.frames[frame].frame()?;
        if callback(self.metadata(frame)?, &output) {
            Ok(InteractiveGeometryStatus::Complete {
                frames: 1,
                inferences,
            })
        } else {
            Ok(InteractiveGeometryStatus::Interrupted {
                frames: 1,
                inferences,
            })
        }
    }

    #[cfg(test)]
    pub fn compute_all_frames<C>(&mut self, mut callback: C) -> Result<InteractiveGeometryStatus>
    where
        C: FnMut(InteractiveGeometryMetadata, &RegressionGeometry) -> bool,
    {
        self.begin_compute()?;
        if !self.exact_checkpoints {
            self.invalidate(GeometryInvalidationLayer::Inference);
            self.begin_compute()?;
        }
        self.postprocessor.reset_layers()?;
        for frame in &mut self.frames {
            frame.clear_geometry();
        }
        let mut emitted = 0;
        let mut inferences = 0;
        for frame in 0..self.frames.len() {
            if self.interrupted.load(Ordering::Acquire) {
                return Ok(InteractiveGeometryStatus::Interrupted {
                    frames: emitted,
                    inferences,
                });
            }
            inferences += self.ensure_inference_for_frame(frame, true)?;
            if self.interrupted.load(Ordering::Acquire) && self.frames[frame].inference.is_none() {
                return Ok(InteractiveGeometryStatus::Interrupted {
                    frames: emitted,
                    inferences,
                });
            }
            self.materialize(frame, false)?;
            let output = self.frames[frame].frame()?;
            emitted += 1;
            if !callback(self.metadata(frame)?, &output) {
                return Ok(InteractiveGeometryStatus::Interrupted {
                    frames: emitted,
                    inferences,
                });
            }
        }
        self.exact_checkpoints = true;
        Ok(InteractiveGeometryStatus::Complete {
            frames: emitted,
            inferences,
        })
    }

    fn begin_compute(&mut self) -> Result<()> {
        validate_inputs(&self.audio, &self.emotions, self.contract.emotion_size)?;
        self.interrupted.store(false, Ordering::Release);
        let total = self.total_frames()?;
        let signature = (
            self.audio.nb_accumulated_samples(),
            self.emotions.state().key_count,
            self.identity_index,
        );
        if self.input_signature.is_some_and(|value| value != signature) {
            self.invalidate(GeometryInvalidationLayer::Inference);
        }
        if self.frames.len() != total {
            self.frames
                .resize_with(total, GeometryFrameExecutionState::default);
            self.invalidate(GeometryInvalidationLayer::Inference);
        }
        self.input_signature = Some(signature);
        Ok(())
    }

    fn ensure_inference_for_frame(&mut self, frame: usize, exact: bool) -> Result<usize> {
        if self.frames[frame].inference.is_some() {
            return Ok(0);
        }
        let internal = self.internal_frame(frame)?;
        let target = internal / self.contract.center_frames;
        let state_size = self.contract.state_size()?;
        let inference_count = self.contract.progress.available_windows(
            i64::try_from(self.audio.nb_accumulated_samples()).unwrap_or(i64::MAX),
            true,
        )?;
        if self.checkpoints.len() < inference_count + 1 {
            self.checkpoints.resize_with(inference_count + 1, || None);
        }

        let checkpoint = (0..=target)
            .rev()
            .find_map(|index| self.checkpoints[index].clone().map(|state| (index, state)));
        let (start, mut state) = match checkpoint {
            Some(value) if self.exact_checkpoints || !exact => value,
            _ if exact || self.preview_inferences == 0 => (0, vec![0.0; state_size]),
            _ => {
                self.exact_checkpoints = false;
                (
                    target.saturating_sub(self.preview_inferences),
                    vec![0.0; state_size],
                )
            }
        };
        let mut noise = PhiloxNoise::new(1, self.contract.noise_size()?, self.seed)?;
        noise.reset(0, start)?;
        let mut executed = 0;
        for inference in start..=target {
            if self.interrupted.load(Ordering::Acquire) {
                break;
            }
            self.checkpoints[inference] = Some(state.clone());
            let input = self.prepare_diffusion_input(inference, state, noise.generate(0)?)?;
            let mut outputs = self.backend.infer_batch(&[(0, input)])?;
            if outputs.len() != 1 {
                return Err(invalid("interactive diffusion backend batch size differs"));
            }
            let output = outputs.pop().expect("one validated output");
            self.validate_diffusion_output(&output)?;
            state = output.output_latents;
            self.checkpoints[inference + 1] = Some(state.clone());
            self.cache_prediction(inference, &output.prediction)?;
            executed += 1;
        }
        Ok(executed)
    }

    fn prepare_diffusion_input(
        &self,
        inference: usize,
        input_latents: Vec<f32>,
        noise: Vec<f32>,
    ) -> Result<DiffusionFrameInput> {
        let window = self.contract.progress.window(inference)?;
        let audio = self
            .audio
            .read(window.start, self.contract.audio_size, self.input_strength)?;
        let mut emotions =
            Vec::with_capacity(self.contract.center_frames * self.contract.emotion_size);
        for frame in 0..self.contract.center_frames {
            let target = self
                .contract
                .frame_progress
                .window(inference * self.contract.center_frames + frame)?
                .target;
            emotions.extend(
                self.emotions.read(target).map_err(|error| {
                    invalid(format!("interactive emotion unavailable: {error}"))
                })?,
            );
        }
        let mut identity = vec![0.0; self.contract.identity_size];
        identity[self.identity_index] = 1.0;
        Ok(DiffusionFrameInput {
            audio,
            emotions,
            identity,
            noise,
            input_latents,
        })
    }

    fn validate_diffusion_output(&self, output: &DiffusionInferenceOutput) -> Result<()> {
        if output.output_latents.len() != self.contract.state_size()?
            || output.prediction.len()
                != self.contract.total_frames() * self.contract.result_layout.total()?
        {
            return Err(invalid("interactive diffusion output dimensions differ"));
        }
        Ok(())
    }

    fn cache_prediction(&mut self, inference: usize, prediction: &[f32]) -> Result<()> {
        let result_size = self.contract.result_layout.total()?;
        for local in 0..self.contract.center_frames {
            let internal = inference * self.contract.center_frames + local;
            let Some(public) = internal.checked_sub(self.frames_before_audio) else {
                continue;
            };
            if public >= self.frames.len() {
                continue;
            }
            let prediction_frame = self.contract.left_frames + local;
            let offset = prediction_frame * result_size;
            self.frames[public].inference = Some(prediction[offset..offset + result_size].to_vec());
        }
        Ok(())
    }

    fn materialize(&mut self, frame: usize, stateless: bool) -> Result<()> {
        let dt =
            self.contract.frame_rate_denominator as f32 / self.contract.frame_rate_numerator as f32;
        let live_time = frame as f32 * dt;
        let cache = &mut self.frames[frame];
        let inference = cache
            .inference
            .as_deref()
            .ok_or_else(|| invalid("interactive diffusion inference is missing"))?;
        if cache.skin.is_none() {
            cache.skin = Some(self.postprocessor.process_skin(inference, dt, stateless)?);
        }
        if cache.tongue.is_none() {
            cache.tongue = Some(self.postprocessor.process_tongue(inference)?);
        }
        if cache.teeth.is_none() {
            cache.teeth = Some(self.postprocessor.process_teeth(inference)?);
        }
        if cache.eyes.is_none() {
            cache.eyes = Some(self.postprocessor.process_eyes(inference, live_time)?);
        }
        Ok(())
    }

    fn internal_frame(&self, frame: usize) -> Result<usize> {
        self.frames_before_audio
            .checked_add(frame)
            .ok_or_else(|| invalid("interactive diffusion frame index overflow"))
    }

    fn metadata(&self, frame: usize) -> Result<InteractiveGeometryMetadata> {
        let internal = self.internal_frame(frame)?;
        Ok(InteractiveGeometryMetadata {
            frame,
            inference: Some(internal / self.contract.center_frames),
            timestamp: self.contract.frame_progress.window(internal)?.target,
            next_timestamp: self.contract.frame_progress.window(internal + 1)?.target,
        })
    }
}

fn layer_valid(frames: &[GeometryFrameExecutionState], layer: GeometryInvalidationLayer) -> bool {
    if layer == GeometryInvalidationLayer::None {
        return true;
    }
    if frames.is_empty() {
        return false;
    }
    let valid = |frame: &GeometryFrameExecutionState, layer| match layer {
        GeometryInvalidationLayer::None => true,
        GeometryInvalidationLayer::All => {
            frame.inference.is_some()
                && frame.skin.is_some()
                && frame.tongue.is_some()
                && frame.teeth.is_some()
                && frame.eyes.is_some()
        }
        GeometryInvalidationLayer::Inference => frame.inference.is_some(),
        GeometryInvalidationLayer::Skin => frame.skin.is_some(),
        GeometryInvalidationLayer::Tongue => frame.tongue.is_some(),
        GeometryInvalidationLayer::Teeth => frame.teeth.is_some(),
        GeometryInvalidationLayer::Eyes => frame.eyes.is_some(),
    };
    frames.iter().all(|frame| valid(frame, layer))
}

fn frames_before_audio(contract: &DiffusionContract) -> Result<usize> {
    let mut frame = 0;
    loop {
        if contract.frame_progress.window(frame)?.target >= 0 {
            return Ok(frame);
        }
        frame = frame
            .checked_add(1)
            .ok_or_else(|| invalid("diffusion pre-audio frame count overflow"))?;
    }
}

fn validate_inputs(
    audio: &AudioAccumulator,
    emotions: &EmotionAccumulator,
    emotion_size: usize,
) -> Result<()> {
    if !audio.is_closed() || audio.nb_dropped_samples() != 0 {
        return Err(invalid(
            "interactive audio must be closed and retain every sample",
        ));
    }
    let state = emotions.state();
    if state.emotion_size != emotion_size || !state.closed || state.dropped_emotions != 0 {
        return Err(invalid(
            "interactive emotions must be closed, complete, and dimensionally valid",
        ));
    }
    Ok(())
}

fn finite_strength(value: f32) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid("interactive input strength must be finite"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{DiffusionResultLayout, EyesRotation};
    use crate::common::{
        DiffusionAudioParameters, DiffusionParameters, RegressionAudioParameters,
        RegressionParameters,
    };
    use std::cell::Cell;
    use std::rc::Rc;

    #[derive(Clone)]
    struct MockLayers {
        calls: Rc<[Cell<usize>; 4]>,
        skin: SkinAnimatorParams,
        tongue: TongueAnimatorParams,
        teeth: JawParameters,
        eyes: EyesAnimatorParams,
    }

    impl MockLayers {
        fn new() -> Self {
            Self {
                calls: Rc::new(std::array::from_fn(|_| Cell::new(0))),
                skin: SkinAnimatorParams {
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
                tongue: TongueAnimatorParams {
                    tongue_strength: 1.0,
                    tongue_height_offset: 0.0,
                    tongue_depth_offset: 0.0,
                },
                teeth: JawParameters::default(),
                eyes: EyesAnimatorParams {
                    eyeballs_strength: 1.0,
                    saccade_strength: 0.0,
                    right_eyeball_rotation_offset_x: 0.0,
                    right_eyeball_rotation_offset_y: 0.0,
                    left_eyeball_rotation_offset_x: 0.0,
                    left_eyeball_rotation_offset_y: 0.0,
                    saccade_seed: 0.0,
                },
            }
        }
    }

    impl LayeredGeometryPostprocessor for MockLayers {
        fn process_skin(
            &mut self,
            inference: &[f32],
            _dt: f32,
            _stateless: bool,
        ) -> Result<Vec<f32>> {
            self.calls[0].set(self.calls[0].get() + 1);
            Ok(vec![inference[0] * self.skin.skin_strength])
        }

        fn process_tongue(&mut self, inference: &[f32]) -> Result<Vec<f32>> {
            self.calls[1].set(self.calls[1].get() + 1);
            Ok(vec![inference[1] * self.tongue.tongue_strength])
        }

        fn process_teeth(&mut self, inference: &[f32]) -> Result<[f32; 16]> {
            self.calls[2].set(self.calls[2].get() + 1);
            Ok([inference[2] * self.teeth.strength; 16])
        }

        fn process_eyes(&mut self, inference: &[f32], live_time: f32) -> Result<EyesRotation> {
            self.calls[3].set(self.calls[3].get() + 1);
            Ok(EyesRotation {
                right: [inference[3] * self.eyes.eyeballs_strength, live_time, 0.0],
                left: [inference[4] * self.eyes.eyeballs_strength, live_time, 0.0],
            })
        }

        fn reset_layers(&mut self) -> Result<()> {
            Ok(())
        }

        #[cfg(test)]
        fn skin_parameters(&self) -> SkinAnimatorParams {
            self.skin
        }
        #[cfg(test)]
        fn set_skin_parameters(&mut self, parameters: SkinAnimatorParams) -> Result<()> {
            self.skin = parameters;
            Ok(())
        }
    }

    fn closed_inputs(emotion_size: usize) -> (AudioAccumulator, EmotionAccumulator) {
        let audio = AudioAccumulator::new(4, 0).unwrap();
        audio.accumulate(&[1.0; 16]).unwrap();
        audio.close().unwrap();
        let emotions = EmotionAccumulator::new(emotion_size, 4).unwrap();
        emotions.accumulate(-16, &vec![0.0; emotion_size]).unwrap();
        emotions.accumulate(16, &vec![1.0; emotion_size]).unwrap();
        emotions.close().unwrap();
        (audio, emotions)
    }

    fn regression_contract() -> RegressionContract {
        RegressionContract::new(
            &RegressionParameters {
                implicit_emotion_len: 1,
                explicit_emotions: vec!["joy".into()],
                default_emotion: vec![0.0],
                num_shapes_skin: 1,
                num_shapes_tongue: 1,
                num_verts_skin: 1,
                num_verts_tongue: 1,
                result_jaw_size: 3,
                result_eyes_size: 4,
            },
            &RegressionAudioParameters {
                buffer_len: 4,
                buffer_ofs: 2,
                samplerate: 8,
            },
            4,
            1,
        )
        .unwrap()
    }

    #[test]
    fn regression_replays_cached_frame_and_invalidates_only_changed_layer() {
        let calls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&calls);
        let backend = move |_: usize, input: &crate::animation::RegressionFrameInput| {
            observed.set(observed.get() + 1);
            Ok(vec![input.timestamp as f32; 9])
        };
        let layers = MockLayers::new();
        let layer_calls = Rc::clone(&layers.calls);
        let (audio, emotions) = closed_inputs(1);
        let mut executor = RegressionGeometryInteractiveExecution::new(
            backend,
            regression_contract(),
            layers,
            Arc::new(audio),
            Arc::new(emotions),
            vec![0.0],
            1.0,
        )
        .unwrap();
        let mut all = Vec::new();
        let status = executor
            .compute_all_frames(|metadata, frame| {
                all.push((metadata, frame.clone()));
                true
            })
            .unwrap();
        assert!(matches!(status, InteractiveGeometryStatus::Complete { .. }));
        assert!(executor.is_valid(GeometryInvalidationLayer::All));
        let inference_calls = calls.get();
        let before = std::array::from_fn::<_, 4, _>(|index| layer_calls[index].get());
        let mut replay = None;
        executor
            .compute_frame(2, |metadata, frame| {
                replay = Some((metadata, frame.clone()));
                true
            })
            .unwrap();
        assert_eq!(replay.unwrap(), all[2]);
        assert_eq!(calls.get(), inference_calls);
        assert_eq!(
            std::array::from_fn::<_, 4, _>(|index| layer_calls[index].get()),
            before
        );

        let mut skin = executor.skin_parameters();
        skin.skin_strength = 2.0;
        executor.set_skin_parameters(skin).unwrap();
        assert!(!executor.is_valid(GeometryInvalidationLayer::Skin));
        assert!(executor.is_valid(GeometryInvalidationLayer::Inference));
        executor.compute_frame(2, |_, _| true).unwrap();
        assert_eq!(calls.get(), inference_calls);
        assert_eq!(layer_calls[0].get(), before[0] + 1);
        assert_eq!(layer_calls[1].get(), before[1]);

        executor.set_input_strength(0.5).unwrap();
        assert!(!executor.is_valid(GeometryInvalidationLayer::Inference));
        executor.compute_frame(2, |_, _| true).unwrap();
        assert_eq!(calls.get(), inference_calls + 1);
    }

    #[test]
    fn regression_callback_and_external_interrupt_stop_computation() {
        let (audio, emotions) = closed_inputs(1);
        let backend = |_: usize, input: &crate::animation::RegressionFrameInput| {
            Ok(vec![input.timestamp as f32; 9])
        };
        let mut executor = RegressionGeometryInteractiveExecution::new(
            backend,
            regression_contract(),
            MockLayers::new(),
            Arc::new(audio),
            Arc::new(emotions),
            vec![0.0],
            1.0,
        )
        .unwrap();
        assert!(matches!(
            executor.compute_all_frames(|_, _| false).unwrap(),
            InteractiveGeometryStatus::Interrupted { frames: 1, .. }
        ));
        let interrupt = executor.interrupt_handle();
        assert!(matches!(
            executor
                .compute_all_frames(|_, _| {
                    interrupt.interrupt();
                    true
                })
                .unwrap(),
            InteractiveGeometryStatus::Interrupted { frames: 1, .. }
        ));
    }

    fn diffusion_contract() -> DiffusionContract {
        DiffusionContract::new(
            &DiffusionParameters {
                emotions: vec!["joy".into()],
                default_emotion: vec![0.0],
                identities: vec!["actor".into()],
                skin_size: 3,
                tongue_size: 3,
                jaw_size: 3,
                eyes_size: 4,
                num_diffusion_steps: 1,
                num_gru_layers: 1,
                gru_latent_dim: 2,
                num_frames_left_truncate: 1,
                num_frames_right_truncate: 1,
                num_frames_center: 2,
            },
            &DiffusionAudioParameters {
                buffer_len: 8,
                padding_left: 8,
                padding_right: 8,
                samplerate: 8,
            },
        )
        .unwrap()
    }

    #[test]
    fn diffusion_exact_replay_uses_checkpoints_and_preview_limits_history() {
        let contract = diffusion_contract();
        assert_eq!(
            contract.result_layout,
            DiffusionResultLayout {
                skin: 3,
                tongue: 3,
                jaw: 3,
                eyes: 4
            }
        );
        let result_size = contract.result_layout.total().unwrap();
        let total_frames = contract.total_frames();
        let calls = Rc::new(Cell::new(0));
        let observed = Rc::clone(&calls);
        let backend = move |inputs: &[(usize, DiffusionFrameInput)]| {
            observed.set(observed.get() + 1);
            Ok(inputs
                .iter()
                .map(|(_, input)| {
                    let state = input.input_latents[0] + 1.0;
                    let mut prediction = Vec::new();
                    for frame in 0..total_frames {
                        prediction.extend(vec![state + frame as f32; result_size]);
                    }
                    DiffusionInferenceOutput {
                        output_latents: vec![state; input.input_latents.len()],
                        prediction,
                    }
                })
                .collect())
        };
        let (audio, emotions) = closed_inputs(1);
        let mut executor = DiffusionGeometryInteractiveExecution::new(
            backend,
            contract,
            MockLayers::new(),
            Arc::new(audio),
            Arc::new(emotions),
            0,
            1.0,
            0,
            7,
        )
        .unwrap();
        let mut all = Vec::new();
        executor
            .compute_all_frames(|metadata, frame| {
                all.push((metadata, frame.clone()));
                true
            })
            .unwrap();
        assert!(executor.is_valid(GeometryInvalidationLayer::Inference));
        let full_calls = calls.get();
        let last = all.len() - 1;
        let mut replay = None;
        executor
            .compute_frame(last, |metadata, frame| {
                replay = Some((metadata, frame.clone()));
                true
            })
            .unwrap();
        assert_eq!(replay.unwrap(), all[last]);
        assert_eq!(calls.get(), full_calls);

        executor.invalidate(GeometryInvalidationLayer::Inference);
        executor.compute_frame(last, |_, _| true).unwrap();
        let exact_replay_calls = calls.get() - full_calls;
        assert!(exact_replay_calls > 1);

        executor.invalidate(GeometryInvalidationLayer::Inference);
        executor.set_preview_inferences(1);
        let before_preview = calls.get();
        executor.compute_frame(last, |_, _| true).unwrap();
        assert!(calls.get() - before_preview <= 2);
        assert!(!executor.is_valid(GeometryInvalidationLayer::Inference));
    }
}
