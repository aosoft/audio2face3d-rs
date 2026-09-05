use crate::animation::{
    DiffusionContract, DiffusionResultLayout, RegressionContract, RegressionResultLayout,
};
#[cfg(feature = "cuda")]
use crate::animation::{DiffusionFrameInput, RegressionFrameInput};
use crate::common::{BindingSchema, Dimension, ElementType, Error, IoMode, Result};
#[cfg(feature = "cuda")]
use crate::cuda::{CudaStream, DeviceBuffer, DeviceView, GpuDevice};
#[cfg(feature = "cuda")]
use std::sync::Arc;

/// Offset, component size, and distance between consecutive batch elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorBatchInfo {
    pub offset: usize,
    pub size: usize,
    pub stride: usize,
}

/// Post-processed geometry layout used by both model families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeometryResultLayout {
    pub skin: usize,
    pub tongue: usize,
    pub jaw_transform: usize,
    pub eyes_rotation: usize,
}

impl GeometryResultLayout {
    pub const JAW_TRANSFORM_SIZE: usize = 16;
    pub const EYES_ROTATION_SIZE: usize = 6;

    pub const fn new(skin: usize, tongue: usize) -> Self {
        Self {
            skin,
            tongue,
            jaw_transform: Self::JAW_TRANSFORM_SIZE,
            eyes_rotation: Self::EYES_ROTATION_SIZE,
        }
    }

    pub fn total(self) -> Result<usize> {
        checked_sum(
            [
                self.skin,
                self.tongue,
                self.jaw_transform,
                self.eyes_rotation,
            ],
            "geometry_result_size",
        )
    }

    pub fn skin_info(self) -> Result<TensorBatchInfo> {
        component_info(
            [
                self.skin,
                self.tongue,
                self.jaw_transform,
                self.eyes_rotation,
            ],
            0,
        )
    }

    pub fn tongue_info(self) -> Result<TensorBatchInfo> {
        component_info(
            [
                self.skin,
                self.tongue,
                self.jaw_transform,
                self.eyes_rotation,
            ],
            1,
        )
    }

    pub fn jaw_info(self) -> Result<TensorBatchInfo> {
        component_info(
            [
                self.skin,
                self.tongue,
                self.jaw_transform,
                self.eyes_rotation,
            ],
            2,
        )
    }

    pub fn eyes_info(self) -> Result<TensorBatchInfo> {
        component_info(
            [
                self.skin,
                self.tongue,
                self.jaw_transform,
                self.eyes_rotation,
            ],
            3,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeBinding {
    pub name: String,
    pub mode: IoMode,
    pub element_type: ElementType,
    pub dimensions: Vec<usize>,
    pub element_count: usize,
}

/// Resolved runtime shapes and allocation sizes derived from one binding schema.
#[derive(Debug, Clone)]
pub struct ModelBindingContract {
    schema: BindingSchema,
    count: usize,
    bindings: Vec<RuntimeBinding>,
}

impl ModelBindingContract {
    pub fn new(schema: BindingSchema, count: usize) -> Result<Self> {
        if count == 0 {
            return Err(invalid("model buffers require at least one track"));
        }
        let bindings = schema
            .bindings()
            .iter()
            .map(|binding| {
                let dimensions = binding
                    .shape
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Fixed(value) => Ok(*value),
                        Dimension::Batch => Ok(count),
                        Dimension::Dynamic { .. } => Err(invalid(
                            "model buffer schemas must resolve dynamic dimensions before allocation",
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let element_count = dimensions.iter().try_fold(1_usize, |total, value| {
                    total.checked_mul(*value).ok_or(Error::IntegerOverflow {
                        field: "runtime_binding_elements",
                        value: *value,
                        target: "usize",
                    })
                })?;
                Ok(RuntimeBinding {
                    name: binding.name.clone(),
                    mode: binding.mode,
                    element_type: binding.element_type,
                    dimensions,
                    element_count,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            schema,
            count,
            bindings,
        })
    }

    pub const fn count(&self) -> usize {
        self.count
    }

    pub fn schema(&self) -> &BindingSchema {
        &self.schema
    }

    pub fn bindings(&self) -> &[RuntimeBinding] {
        &self.bindings
    }

    pub fn binding(&self, name: &str) -> Result<&RuntimeBinding> {
        self.bindings
            .iter()
            .find(|binding| binding.name == name)
            .ok_or_else(|| invalid(format!("missing model binding {name}")))
    }

    /// Validates an engine schema against the same contract used for allocation.
    pub fn validate_engine_schema(&self, actual: &BindingSchema) -> Result<()> {
        if actual.bindings().len() != self.schema.bindings().len() {
            return Err(invalid(
                "engine binding count does not match model contract",
            ));
        }
        for expected in self.schema.bindings() {
            let actual = actual
                .get(&expected.name)
                .ok_or_else(|| invalid(format!("missing engine binding {}", expected.name)))?;
            if actual.mode != expected.mode || actual.element_type != expected.element_type {
                return Err(invalid(format!(
                    "engine binding {} type or mode does not match model contract",
                    expected.name
                )));
            }
            let expected_dimensions = expected.shape.dimensions();
            let actual_dimensions = actual.shape.dimensions();
            if expected_dimensions.len() != actual_dimensions.len()
                || !expected_dimensions
                    .iter()
                    .zip(actual_dimensions)
                    .all(|(expected, actual)| match (expected, actual) {
                        (Dimension::Fixed(expected), Dimension::Fixed(actual)) => {
                            expected == actual
                        }
                        (Dimension::Batch, Dimension::Batch) => true,
                        (Dimension::Batch, Dimension::Dynamic { min, max }) => {
                            (*min..=*max).contains(&self.count)
                        }
                        _ => false,
                    })
            {
                return Err(invalid(format!(
                    "engine binding {} shape does not match model contract",
                    expected.name
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct RegressionBufferContract {
    bindings: ModelBindingContract,
    pub implicit_emotion_size: usize,
    pub explicit_emotion_size: usize,
    pub inference_layout: RegressionResultLayout,
    pub result_layout: GeometryResultLayout,
}

impl RegressionBufferContract {
    pub fn new(model: &RegressionContract, count: usize) -> Result<Self> {
        Ok(Self {
            bindings: ModelBindingContract::new(model.binding_schema()?, count)?,
            implicit_emotion_size: model.implicit_emotion_size,
            explicit_emotion_size: model.explicit_emotion_size,
            inference_layout: model.result_layout,
            result_layout: GeometryResultLayout::new(
                model.result_skin_size,
                model.result_tongue_size,
            ),
        })
    }

    pub fn bindings(&self) -> &ModelBindingContract {
        &self.bindings
    }

    pub const fn count(&self) -> usize {
        self.bindings.count()
    }

    pub fn skin_inference_info(&self) -> Result<TensorBatchInfo> {
        regression_component_info(self.inference_layout, 0)
    }

    pub fn tongue_inference_info(&self) -> Result<TensorBatchInfo> {
        regression_component_info(self.inference_layout, 1)
    }

    pub fn jaw_inference_info(&self) -> Result<TensorBatchInfo> {
        regression_component_info(self.inference_layout, 2)
    }

    pub fn eyes_inference_info(&self) -> Result<TensorBatchInfo> {
        regression_component_info(self.inference_layout, 3)
    }
}

#[derive(Debug, Clone)]
pub struct DiffusionBufferContract {
    bindings: ModelBindingContract,
    pub inference_layout: DiffusionResultLayout,
    pub result_layout: GeometryResultLayout,
    pub diffusion_steps: usize,
    pub gru_layers: usize,
    pub gru_latent_size: usize,
    pub left_frames: usize,
    pub center_frames: usize,
    pub right_frames: usize,
}

impl DiffusionBufferContract {
    pub fn new(model: &DiffusionContract, count: usize) -> Result<Self> {
        Ok(Self {
            bindings: ModelBindingContract::new(model.schema()?, count)?,
            inference_layout: model.result_layout,
            result_layout: GeometryResultLayout::new(
                model.result_layout.skin,
                model.result_layout.tongue,
            ),
            diffusion_steps: model.diffusion_steps,
            gru_layers: model.gru_layers,
            gru_latent_size: model.gru_latent_size,
            left_frames: model.left_frames,
            center_frames: model.center_frames,
            right_frames: model.right_frames,
        })
    }

    pub fn bindings(&self) -> &ModelBindingContract {
        &self.bindings
    }

    pub const fn count(&self) -> usize {
        self.bindings.count()
    }

    pub const fn total_frames(&self) -> usize {
        self.left_frames + self.center_frames + self.right_frames
    }

    pub fn prediction_stride(&self) -> Result<usize> {
        self.total_frames()
            .checked_mul(self.inference_layout.total()?)
            .ok_or(Error::IntegerOverflow {
                field: "diffusion_prediction_stride",
                value: self.total_frames(),
                target: "usize",
            })
    }

    pub fn component_info(&self, frame: usize, component: usize) -> Result<TensorBatchInfo> {
        if frame >= self.center_frames {
            return Err(invalid("diffusion center frame is out of range"));
        }
        let sizes = [
            self.inference_layout.skin,
            self.inference_layout.tongue,
            self.inference_layout.jaw,
            self.inference_layout.eyes,
        ];
        let base = self
            .left_frames
            .checked_add(frame)
            .and_then(|value| value.checked_mul(self.inference_layout.total().ok()?))
            .ok_or(Error::IntegerOverflow {
                field: "diffusion_component_offset",
                value: frame,
                target: "usize",
            })?;
        let component = component_info(sizes, component)?;
        Ok(TensorBatchInfo {
            offset: base
                .checked_add(component.offset)
                .ok_or(Error::IntegerOverflow {
                    field: "diffusion_component_offset",
                    value: component.offset,
                    target: "usize",
                })?,
            size: component.size,
            stride: self.prediction_stride()?,
        })
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct RegressionInferenceInputBuffers {
    count: usize,
    implicit_emotion_size: usize,
    explicit_emotion_size: usize,
    emotion: DeviceBuffer<f32>,
    input: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl RegressionInferenceInputBuffers {
    pub fn allocate(device: &Arc<GpuDevice>, contract: &RegressionBufferContract) -> Result<Self> {
        Ok(Self {
            count: contract.count(),
            implicit_emotion_size: contract.implicit_emotion_size,
            explicit_emotion_size: contract.explicit_emotion_size,
            emotion: device.allocate(contract.bindings.binding("emotion")?.element_count)?,
            input: device.allocate(contract.bindings.binding("input")?.element_count)?,
        })
    }

    pub const fn count(&self) -> usize {
        self.count
    }
    pub fn emotion_tensor(&self) -> DeviceView<'_, f32> {
        self.emotion.view()
    }
    pub fn input_tensor(&self) -> DeviceView<'_, f32> {
        self.input.view()
    }

    pub fn implicit_emotions(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.emotion.view().slice(
            self.track_emotion_offset(index)?,
            self.implicit_emotion_size,
        )
    }

    pub fn explicit_emotions(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.emotion.view().slice(
            self.track_emotion_offset(index)? + self.implicit_emotion_size,
            self.explicit_emotion_size,
        )
    }

    pub fn input(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        checked_track_view(
            &self.input,
            index,
            self.count,
            self.input.len() / self.count,
        )
    }

    pub fn copy_frames(
        &mut self,
        frames: &[RegressionFrameInput],
        stream: &CudaStream,
    ) -> Result<()> {
        if frames.len() != self.count {
            return Err(invalid(
                "regression frame count does not match buffer count",
            ));
        }
        let emotion_stride = self.emotion.len() / self.count;
        let input_stride = self.input.len() / self.count;
        if frames
            .iter()
            .any(|frame| frame.emotion.len() != emotion_stride || frame.audio.len() != input_stride)
        {
            return Err(invalid(
                "regression frame dimensions do not match buffer contract",
            ));
        }
        let emotion = frames
            .iter()
            .flat_map(|frame| frame.emotion.iter().copied())
            .collect::<Vec<_>>();
        let input = frames
            .iter()
            .flat_map(|frame| frame.audio.iter().copied())
            .collect::<Vec<_>>();
        self.emotion.copy_from(&emotion, stream)?;
        self.input.copy_from(&input, stream)
    }

    fn track_emotion_offset(&self, index: usize) -> Result<usize> {
        checked_track_offset(index, self.count, self.emotion.len() / self.count)
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct RegressionInferenceOutputBuffers {
    count: usize,
    layout: RegressionResultLayout,
    result: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl RegressionInferenceOutputBuffers {
    pub fn allocate(device: &Arc<GpuDevice>, contract: &RegressionBufferContract) -> Result<Self> {
        Ok(Self {
            count: contract.count(),
            layout: contract.inference_layout,
            result: device.allocate(contract.bindings.binding("result")?.element_count)?,
        })
    }

    pub const fn count(&self) -> usize {
        self.count
    }
    pub fn result_tensor(&self) -> DeviceView<'_, f32> {
        self.result.view()
    }
    pub fn result(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        checked_track_view(&self.result, index, self.count, self.layout.total()?)
    }
    pub fn skin(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(index, 0)
    }
    pub fn tongue(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(index, 1)
    }
    pub fn jaw(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(index, 2)
    }
    pub fn eyes(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(index, 3)
    }
    pub fn skin_batch_info(&self) -> Result<TensorBatchInfo> {
        regression_component_info(self.layout, 0)
    }
    pub fn tongue_batch_info(&self) -> Result<TensorBatchInfo> {
        regression_component_info(self.layout, 1)
    }
    pub fn jaw_batch_info(&self) -> Result<TensorBatchInfo> {
        regression_component_info(self.layout, 2)
    }
    pub fn eyes_batch_info(&self) -> Result<TensorBatchInfo> {
        regression_component_info(self.layout, 3)
    }
    pub fn copy_to_host(&self, stream: &CudaStream) -> Result<Vec<f32>> {
        let mut values = vec![0.0; self.result.len()];
        self.result.copy_to(&mut values, stream)?;
        Ok(values)
    }
    fn component(&self, index: usize, component: usize) -> Result<DeviceView<'_, f32>> {
        let info = regression_component_info(self.layout, component)?;
        let base = checked_track_offset(index, self.count, info.stride)?;
        self.result.view().slice(base + info.offset, info.size)
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct GeometryResultBuffers {
    count: usize,
    layout: GeometryResultLayout,
    result: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl GeometryResultBuffers {
    pub fn allocate(
        device: &Arc<GpuDevice>,
        layout: GeometryResultLayout,
        count: usize,
    ) -> Result<Self> {
        if count == 0 {
            return Err(invalid(
                "geometry result buffers require at least one track",
            ));
        }
        let len = layout
            .total()?
            .checked_mul(count)
            .ok_or(Error::IntegerOverflow {
                field: "geometry_result_allocation",
                value: count,
                target: "usize",
            })?;
        Ok(Self {
            count,
            layout,
            result: device.allocate(len)?,
        })
    }
    pub const fn count(&self) -> usize {
        self.count
    }
    pub const fn layout(&self) -> GeometryResultLayout {
        self.layout
    }
    pub fn result_tensor(&self) -> DeviceView<'_, f32> {
        self.result.view()
    }
    pub fn result(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        checked_track_view(&self.result, index, self.count, self.layout.total()?)
    }
    pub fn skin(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(index, self.layout.skin_info()?)
    }
    pub fn tongue(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(index, self.layout.tongue_info()?)
    }
    pub fn jaw_transform(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(index, self.layout.jaw_info()?)
    }
    pub fn eyes_rotation(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(index, self.layout.eyes_info()?)
    }
    pub fn skin_batch_info(&self) -> Result<TensorBatchInfo> {
        self.layout.skin_info()
    }
    pub fn tongue_batch_info(&self) -> Result<TensorBatchInfo> {
        self.layout.tongue_info()
    }
    pub fn jaw_batch_info(&self) -> Result<TensorBatchInfo> {
        self.layout.jaw_info()
    }
    pub fn eyes_batch_info(&self) -> Result<TensorBatchInfo> {
        self.layout.eyes_info()
    }
    fn component(&self, index: usize, info: TensorBatchInfo) -> Result<DeviceView<'_, f32>> {
        let base = checked_track_offset(index, self.count, info.stride)?;
        self.result.view().slice(base + info.offset, info.size)
    }
}

#[cfg(feature = "cuda")]
pub type RegressionResultBuffers = GeometryResultBuffers;
#[cfg(feature = "cuda")]
pub type DiffusionResultBuffers = GeometryResultBuffers;

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DiffusionInferenceInputBuffers {
    count: usize,
    emotion_frames: usize,
    emotion_size: usize,
    window: DeviceBuffer<f32>,
    emotion: DeviceBuffer<f32>,
    identity: DeviceBuffer<f32>,
    noise: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl DiffusionInferenceInputBuffers {
    pub fn allocate(device: &Arc<GpuDevice>, contract: &DiffusionBufferContract) -> Result<Self> {
        let binding = |name| {
            contract
                .bindings
                .binding(name)
                .map(|value| value.element_count)
        };
        Ok(Self {
            count: contract.count(),
            emotion_frames: contract.center_frames,
            emotion_size: contract.bindings.binding("emotion")?.dimensions[2],
            window: device.allocate(binding("window")?)?,
            emotion: device.allocate(binding("emotion")?)?,
            identity: device.allocate(binding("identity")?)?,
            noise: device.allocate(binding("noise")?)?,
        })
    }
    pub const fn count(&self) -> usize {
        self.count
    }
    pub fn window_tensor(&self) -> DeviceView<'_, f32> {
        self.window.view()
    }
    pub fn emotion_tensor(&self) -> DeviceView<'_, f32> {
        self.emotion.view()
    }
    pub fn identity_tensor(&self) -> DeviceView<'_, f32> {
        self.identity.view()
    }
    pub fn noise_tensor(&self) -> DeviceView<'_, f32> {
        self.noise.view()
    }
    pub fn window(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        checked_track_view(
            &self.window,
            index,
            self.count,
            self.window.len() / self.count,
        )
    }
    pub fn emotions(&self, frame: usize, index: usize) -> Result<DeviceView<'_, f32>> {
        let per_track = self.emotion.len() / self.count;
        if frame >= self.emotion_frames {
            return Err(invalid("diffusion emotion frame is out of range"));
        }
        let base = checked_track_offset(index, self.count, per_track)?;
        self.emotion
            .view()
            .slice(base + frame * self.emotion_size, self.emotion_size)
    }
    pub fn identity(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        checked_track_view(
            &self.identity,
            index,
            self.count,
            self.identity.len() / self.count,
        )
    }
    pub fn noise(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        checked_track_view(
            &self.noise,
            index,
            self.count,
            self.noise.len() / self.count,
        )
    }
    pub fn copy_frames(
        &mut self,
        frames: &[DiffusionFrameInput],
        stream: &CudaStream,
    ) -> Result<()> {
        if frames.len() != self.count {
            return Err(invalid("diffusion frame count does not match buffer count"));
        }
        copy_frame_field(
            &mut self.window,
            self.count,
            frames,
            |frame| &frame.audio,
            stream,
        )?;
        copy_frame_field(
            &mut self.emotion,
            self.count,
            frames,
            |frame| &frame.emotions,
            stream,
        )?;
        copy_frame_field(
            &mut self.identity,
            self.count,
            frames,
            |frame| &frame.identity,
            stream,
        )?;
        copy_frame_field(
            &mut self.noise,
            self.count,
            frames,
            |frame| &frame.noise,
            stream,
        )
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DiffusionInferenceStateBuffers {
    count: usize,
    slices: usize,
    latent_size: usize,
    input: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl DiffusionInferenceStateBuffers {
    pub fn allocate(device: &Arc<GpuDevice>, contract: &DiffusionBufferContract) -> Result<Self> {
        Ok(Self {
            count: contract.count(),
            slices: contract.diffusion_steps * contract.gru_layers,
            latent_size: contract.gru_latent_size,
            input: device.allocate(contract.bindings.binding("input_latents")?.element_count)?,
            output: device.allocate(contract.bindings.binding("output_latents")?.element_count)?,
        })
    }
    pub const fn count(&self) -> usize {
        self.count
    }
    pub fn input_tensor(&self) -> DeviceView<'_, f32> {
        self.input.view()
    }
    pub fn output_tensor(&self) -> DeviceView<'_, f32> {
        self.output.view()
    }
    pub fn copy_track_inputs(
        &mut self,
        frames: &[DiffusionFrameInput],
        stream: &CudaStream,
    ) -> Result<()> {
        if frames.len() != self.count {
            return Err(invalid(
                "diffusion state frame count does not match buffer count",
            ));
        }
        let per_track = self.slices * self.latent_size;
        if frames
            .iter()
            .any(|frame| frame.input_latents.len() != per_track)
        {
            return Err(invalid(
                "diffusion input state dimensions do not match buffer contract",
            ));
        }
        let mut interleaved = Vec::with_capacity(self.input.len());
        for slice in 0..self.slices {
            let offset = slice * self.latent_size;
            for frame in frames {
                interleaved
                    .extend_from_slice(&frame.input_latents[offset..offset + self.latent_size]);
            }
        }
        self.input.copy_from(&interleaved, stream)
    }
    pub fn copy_output_tracks_to_host(&self, stream: &CudaStream) -> Result<Vec<Vec<f32>>> {
        let mut interleaved = vec![0.0; self.output.len()];
        self.output.copy_to(&mut interleaved, stream)?;
        let mut tracks = vec![Vec::with_capacity(self.slices * self.latent_size); self.count];
        for slice in 0..self.slices {
            for (track, values) in tracks.iter_mut().enumerate() {
                let offset = (slice * self.count + track) * self.latent_size;
                values.extend_from_slice(&interleaved[offset..offset + self.latent_size]);
            }
        }
        Ok(tracks)
    }
    pub fn copy_output_to_input(&mut self, stream: &CudaStream, index: usize) -> Result<()> {
        if index >= self.count {
            return Err(invalid("diffusion state track is out of range"));
        }
        for slice in 0..self.slices {
            let offset = (slice * self.count + index) * self.latent_size;
            self.input.copy_from_device_range(
                offset,
                &self.output,
                offset,
                self.latent_size,
                stream,
            )?;
        }
        Ok(())
    }
    pub fn reset(&mut self, stream: &CudaStream, index: usize) -> Result<()> {
        if index >= self.count {
            return Err(invalid("diffusion state track is out of range"));
        }
        for slice in 0..self.slices {
            let offset = (slice * self.count + index) * self.latent_size;
            self.input
                .memset_zero_range(offset, self.latent_size, stream)?;
        }
        Ok(())
    }
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.input, &mut self.output);
    }
    /// Commits inference output and restores old state for inactive tracks.
    pub fn commit_active(&mut self, stream: &CudaStream, active: &[bool]) -> Result<()> {
        if active.len() != self.count {
            return Err(invalid("diffusion active-track mask length mismatch"));
        }
        self.swap();
        for (index, active) in active.iter().copied().enumerate() {
            if !active {
                self.copy_output_to_input(stream, index)?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct DiffusionInferenceOutputBuffers {
    count: usize,
    layout: DiffusionResultLayout,
    left_frames: usize,
    center_frames: usize,
    total_frames: usize,
    prediction: DeviceBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl DiffusionInferenceOutputBuffers {
    pub fn allocate(device: &Arc<GpuDevice>, contract: &DiffusionBufferContract) -> Result<Self> {
        Ok(Self {
            count: contract.count(),
            layout: contract.inference_layout,
            left_frames: contract.left_frames,
            center_frames: contract.center_frames,
            total_frames: contract.total_frames(),
            prediction: device.allocate(contract.bindings.binding("prediction")?.element_count)?,
        })
    }
    pub const fn count(&self) -> usize {
        self.count
    }
    pub const fn frame_count(&self) -> usize {
        self.total_frames
    }
    pub fn result_tensor(&self) -> DeviceView<'_, f32> {
        self.prediction.view()
    }
    pub fn inference_result(&self, index: usize) -> Result<DeviceView<'_, f32>> {
        let result_size = self.layout.total()?;
        let stride = self.total_frames * result_size;
        let base = checked_track_offset(index, self.count, stride)?;
        self.prediction.view().slice(
            base + self.left_frames * result_size,
            self.center_frames * result_size,
        )
    }
    pub fn skin(&self, frame: usize, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(frame, index, 0)
    }
    pub fn tongue(&self, frame: usize, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(frame, index, 1)
    }
    pub fn jaw(&self, frame: usize, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(frame, index, 2)
    }
    pub fn eyes(&self, frame: usize, index: usize) -> Result<DeviceView<'_, f32>> {
        self.component(frame, index, 3)
    }
    pub fn skin_batch_info(&self, frame: usize) -> Result<TensorBatchInfo> {
        self.component_batch_info(frame, 0)
    }
    pub fn tongue_batch_info(&self, frame: usize) -> Result<TensorBatchInfo> {
        self.component_batch_info(frame, 1)
    }
    pub fn jaw_batch_info(&self, frame: usize) -> Result<TensorBatchInfo> {
        self.component_batch_info(frame, 2)
    }
    pub fn eyes_batch_info(&self, frame: usize) -> Result<TensorBatchInfo> {
        self.component_batch_info(frame, 3)
    }
    pub fn copy_to_host(&self, stream: &CudaStream) -> Result<Vec<f32>> {
        let mut values = vec![0.0; self.prediction.len()];
        self.prediction.copy_to(&mut values, stream)?;
        Ok(values)
    }
    fn component(
        &self,
        frame: usize,
        index: usize,
        component: usize,
    ) -> Result<DeviceView<'_, f32>> {
        if frame >= self.center_frames {
            return Err(invalid("diffusion center frame is out of range"));
        }
        let sizes = [
            self.layout.skin,
            self.layout.tongue,
            self.layout.jaw,
            self.layout.eyes,
        ];
        let info = component_info(sizes, component)?;
        let stride = self.total_frames * info.stride;
        let base = checked_track_offset(index, self.count, stride)?;
        let frame_offset = (self.left_frames + frame) * info.stride;
        self.prediction
            .view()
            .slice(base + frame_offset + info.offset, info.size)
    }

    fn component_batch_info(&self, frame: usize, component: usize) -> Result<TensorBatchInfo> {
        if frame >= self.center_frames {
            return Err(invalid("diffusion center frame is out of range"));
        }
        let sizes = [
            self.layout.skin,
            self.layout.tongue,
            self.layout.jaw,
            self.layout.eyes,
        ];
        let info = component_info(sizes, component)?;
        Ok(TensorBatchInfo {
            offset: (self.left_frames + frame) * info.stride + info.offset,
            size: info.size,
            stride: self.total_frames * info.stride,
        })
    }
}

#[cfg(all(feature = "cuda", feature = "tensorrt"))]
impl RegressionBufferContract {
    pub fn device_bindings<'a>(
        &self,
        input: &'a RegressionInferenceInputBuffers,
        output: &'a RegressionInferenceOutputBuffers,
    ) -> Result<crate::tensorrt::DeviceBindings<'a>> {
        ensure_counts(self.count(), [input.count(), output.count()])?;
        let mut bindings = crate::tensorrt::DeviceBindings::new();
        insert_binding(&mut bindings, "emotion", input.emotion_tensor())?;
        insert_binding(&mut bindings, "input", input.input_tensor())?;
        insert_binding(&mut bindings, "result", output.result_tensor())?;
        set_input_shapes(&mut bindings, &self.bindings)?;
        Ok(bindings)
    }
}

#[cfg(all(feature = "cuda", feature = "tensorrt"))]
impl DiffusionBufferContract {
    pub fn device_bindings<'a>(
        &self,
        input: &'a DiffusionInferenceInputBuffers,
        state: &'a DiffusionInferenceStateBuffers,
        output: &'a DiffusionInferenceOutputBuffers,
    ) -> Result<crate::tensorrt::DeviceBindings<'a>> {
        ensure_counts(self.count(), [input.count(), state.count(), output.count()])?;
        let mut bindings = crate::tensorrt::DeviceBindings::new();
        insert_binding(&mut bindings, "emotion", input.emotion_tensor())?;
        insert_binding(&mut bindings, "identity", input.identity_tensor())?;
        insert_binding(&mut bindings, "input_latents", state.input_tensor())?;
        insert_binding(&mut bindings, "noise", input.noise_tensor())?;
        insert_binding(&mut bindings, "window", input.window_tensor())?;
        insert_binding(&mut bindings, "output_latents", state.output_tensor())?;
        insert_binding(&mut bindings, "prediction", output.result_tensor())?;
        set_input_shapes(&mut bindings, &self.bindings)?;
        Ok(bindings)
    }
}

#[cfg(all(feature = "cuda", feature = "tensorrt"))]
fn insert_binding<'a>(
    bindings: &mut crate::tensorrt::DeviceBindings<'a>,
    name: &str,
    view: DeviceView<'a, f32>,
) -> Result<()> {
    bindings
        .insert(
            name,
            crate::tensorrt::BindingBuffer::from_view(view, ElementType::F32),
        )
        .map_err(|error| invalid(format!("unable to create {name} binding: {error}")))
}

#[cfg(all(feature = "cuda", feature = "tensorrt"))]
fn set_input_shapes(
    bindings: &mut crate::tensorrt::DeviceBindings<'_>,
    contract: &ModelBindingContract,
) -> Result<()> {
    for binding in contract
        .bindings()
        .iter()
        .filter(|binding| binding.mode == IoMode::Input)
    {
        let shape = binding
            .dimensions
            .iter()
            .copied()
            .map(|value| {
                i64::try_from(value).map_err(|_| invalid("runtime binding dimension exceeds i64"))
            })
            .collect::<Result<Vec<_>>>()?;
        bindings
            .set_input_shape(&binding.name, shape)
            .map_err(|error| {
                invalid(format!(
                    "unable to set {} runtime shape: {error}",
                    binding.name
                ))
            })?;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn copy_frame_field(
    target: &mut DeviceBuffer<f32>,
    count: usize,
    frames: &[DiffusionFrameInput],
    select: fn(&DiffusionFrameInput) -> &[f32],
    stream: &CudaStream,
) -> Result<()> {
    let stride = target.len() / count;
    if frames.iter().any(|frame| select(frame).len() != stride) {
        return Err(invalid(
            "diffusion frame dimensions do not match buffer contract",
        ));
    }
    let values = frames
        .iter()
        .flat_map(|frame| select(frame).iter().copied())
        .collect::<Vec<_>>();
    target.copy_from(&values, stream)
}

#[cfg(feature = "cuda")]
fn checked_track_view<T>(
    buffer: &DeviceBuffer<T>,
    index: usize,
    count: usize,
    stride: usize,
) -> Result<DeviceView<'_, T>> {
    buffer
        .view()
        .slice(checked_track_offset(index, count, stride)?, stride)
}

#[cfg(feature = "cuda")]
fn checked_track_offset(index: usize, count: usize, stride: usize) -> Result<usize> {
    if index >= count {
        return Err(invalid("buffer track is out of range"));
    }
    index.checked_mul(stride).ok_or(Error::IntegerOverflow {
        field: "buffer_track_offset",
        value: index,
        target: "usize",
    })
}

fn regression_component_info(
    layout: RegressionResultLayout,
    component: usize,
) -> Result<TensorBatchInfo> {
    component_info(
        [layout.skin, layout.tongue, layout.jaw, layout.eyes],
        component,
    )
}

fn component_info(sizes: [usize; 4], component: usize) -> Result<TensorBatchInfo> {
    let size = *sizes
        .get(component)
        .ok_or_else(|| invalid("geometry component is out of range"))?;
    let offset = checked_sum(sizes[..component].iter().copied(), "component_offset")?;
    Ok(TensorBatchInfo {
        offset,
        size,
        stride: checked_sum(sizes, "component_stride")?,
    })
}

fn checked_sum(values: impl IntoIterator<Item = usize>, field: &'static str) -> Result<usize> {
    values.into_iter().try_fold(0_usize, |total, value| {
        total.checked_add(value).ok_or(Error::IntegerOverflow {
            field,
            value,
            target: "usize",
        })
    })
}

#[cfg(all(feature = "cuda", feature = "tensorrt"))]
fn ensure_counts<const N: usize>(expected: usize, counts: [usize; N]) -> Result<()> {
    if counts.into_iter().all(|count| count == expected) {
        Ok(())
    } else {
        Err(invalid("typed buffer counts do not match binding contract"))
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{
        DiffusionAudioParameters, DiffusionParameters, RegressionAudioParameters,
        RegressionParameters,
    };

    fn regression_contract() -> RegressionContract {
        RegressionContract::new(
            &RegressionParameters {
                implicit_emotion_len: 2,
                explicit_emotions: vec!["joy".into()],
                default_emotion: vec![0.0],
                num_shapes_skin: 3,
                num_shapes_tongue: 2,
                num_verts_skin: 5,
                num_verts_tongue: 2,
                result_jaw_size: 15,
                result_eyes_size: 4,
            },
            &RegressionAudioParameters {
                buffer_len: 8,
                buffer_ofs: 4,
                samplerate: 16_000,
            },
            30,
            1,
        )
        .unwrap()
    }

    fn diffusion_contract() -> DiffusionContract {
        DiffusionContract::new(
            &DiffusionParameters {
                emotions: vec!["joy".into(), "anger".into()],
                default_emotion: vec![0.0; 2],
                identities: vec!["one".into()],
                skin_size: 6,
                tongue_size: 3,
                jaw_size: 9,
                eyes_size: 4,
                num_diffusion_steps: 2,
                num_gru_layers: 3,
                gru_latent_dim: 4,
                num_frames_left_truncate: 1,
                num_frames_right_truncate: 2,
                num_frames_center: 3,
            },
            &DiffusionAudioParameters {
                buffer_len: 12,
                padding_left: 2,
                padding_right: 2,
                samplerate: 16_000,
            },
        )
        .unwrap()
    }

    #[test]
    fn regression_contract_drives_shapes_allocations_and_component_strides() {
        let buffers = RegressionBufferContract::new(&regression_contract(), 4).unwrap();
        assert_eq!(
            buffers.bindings().binding("emotion").unwrap().dimensions,
            [4, 1, 3]
        );
        assert_eq!(
            buffers.bindings().binding("emotion").unwrap().element_count,
            12
        );
        assert_eq!(
            buffers.bindings().binding("result").unwrap().dimensions,
            [4, 1, 24]
        );
        assert_eq!(
            buffers.jaw_inference_info().unwrap(),
            TensorBatchInfo {
                offset: 5,
                size: 15,
                stride: 24
            }
        );
        assert_eq!(
            buffers.result_layout,
            GeometryResultLayout {
                skin: 15,
                tongue: 6,
                jaw_transform: 16,
                eyes_rotation: 6
            }
        );
        assert_eq!(buffers.result_layout.total().unwrap(), 43);
    }

    #[test]
    fn diffusion_runtime_shapes_and_center_frame_batch_info_match_sdk_layout() {
        let buffers = DiffusionBufferContract::new(&diffusion_contract(), 5).unwrap();
        assert_eq!(
            buffers
                .bindings()
                .binding("input_latents")
                .unwrap()
                .dimensions,
            [2, 3, 5, 4]
        );
        assert_eq!(
            buffers.bindings().binding("prediction").unwrap().dimensions,
            [5, 6, 22]
        );
        assert_eq!(
            buffers.component_info(1, 2).unwrap(),
            TensorBatchInfo {
                offset: 53,
                size: 9,
                stride: 132
            }
        );
        assert!(buffers.component_info(3, 0).is_err());
    }

    #[test]
    fn engine_schema_validation_uses_the_allocation_contract() {
        let buffers = RegressionBufferContract::new(&regression_contract(), 4).unwrap();
        assert!(
            buffers
                .bindings()
                .validate_engine_schema(buffers.bindings().schema())
                .is_ok()
        );
        let mut incompatible = regression_contract();
        incompatible.audio_size += 1;
        assert!(
            buffers
                .bindings()
                .validate_engine_schema(&incompatible.binding_schema().unwrap())
                .is_err()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn device_state_commit_and_reset_preserve_inactive_tracks() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let contract = DiffusionBufferContract::new(&diffusion_contract(), 2).unwrap();
        let mut state = DiffusionInferenceStateBuffers::allocate(&device, &contract).unwrap();
        state
            .input
            .copy_from(&vec![1.0; state.input.len()], &stream)
            .unwrap();
        state
            .output
            .copy_from(&vec![2.0; state.output.len()], &stream)
            .unwrap();

        state.commit_active(&stream, &[true, false]).unwrap();
        let mut values = vec![0.0; state.input.len()];
        state.input.copy_to(&mut values, &stream).unwrap();
        for slice in 0..state.slices {
            let first = (slice * 2) * state.latent_size;
            let second = first + state.latent_size;
            assert_eq!(&values[first..first + state.latent_size], &[2.0; 4]);
            assert_eq!(&values[second..second + state.latent_size], &[1.0; 4]);
        }

        state.reset(&stream, 0).unwrap();
        state.input.copy_to(&mut values, &stream).unwrap();
        for slice in 0..state.slices {
            let first = (slice * 2) * state.latent_size;
            let second = first + state.latent_size;
            assert_eq!(&values[first..first + state.latent_size], &[0.0; 4]);
            assert_eq!(&values[second..second + state.latent_size], &[1.0; 4]);
        }
    }
}
