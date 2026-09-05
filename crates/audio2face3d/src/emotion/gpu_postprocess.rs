use crate::common::{Error, Result};
use crate::cuda::{
    CudaEvent, CudaModule, CudaStream, DeviceBuffer, GpuDevice, emotion_postprocess_ptx,
    ensure_same_device,
};
use crate::emotion::{EmotionPostProcessData, EmotionPostProcessParameters};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

const PARAMETER_PREFIX: usize = 8;
const BLOCK_SIZE: u32 = 128;

macro_rules! params {
    ($($value:ident),+ $(,)?) => {
        [$(std::ptr::from_mut(&mut $value).cast::<c_void>()),+]
    };
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuEmotionKernelPath {
    Local,
    Generic,
}

/// Persistent, per-track CUDA implementation of Audio2Emotion post-processing.
pub struct GpuEmotionPostProcessor {
    module: CudaModule,
    data: EmotionPostProcessData,
    host_parameters: Vec<EmotionPostProcessParameters>,
    correspondence: DeviceBuffer<i32>,
    parameters: DeviceBuffer<f32>,
    preferred: DeviceBuffer<f32>,
    state: DeviceBuffer<f32>,
    active: DeviceBuffer<u64>,
    parameter_stride: usize,
    state_stride: usize,
    track_count: usize,
}

impl GpuEmotionPostProcessor {
    pub fn new(
        device: &Arc<GpuDevice>,
        stream: &CudaStream,
        data: EmotionPostProcessData,
        parameters: &[EmotionPostProcessParameters],
    ) -> Result<Self> {
        data.validate()?;
        if parameters.is_empty() {
            return Err(invalid("GPU emotion track count must be non-zero"));
        }
        u32::try_from(parameters.len())
            .map_err(|_| invalid("GPU emotion track count exceeds CUDA index range"))?;
        for value in parameters {
            value.validate(&data)?;
        }
        let parameter_stride = PARAMETER_PREFIX
            .checked_add(data.output_emotion_length)
            .ok_or_else(|| invalid("GPU emotion parameter stride overflow"))?;
        let state_stride = data
            .output_emotion_length
            .checked_mul(3)
            .and_then(|value| value.checked_add(data.inference_emotion_length))
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| invalid("GPU emotion state stride overflow"))?;
        let state_elements = state_stride
            .checked_mul(parameters.len())
            .ok_or_else(|| invalid("GPU emotion state allocation overflow"))?;
        let active_words = parameters.len().div_ceil(u64::BITS as usize);
        let correspondence = data
            .emotion_correspondence
            .iter()
            .map(|value| i32::try_from(*value).map_err(|_| invalid("correspondence exceeds i32")))
            .collect::<Result<Vec<_>>>()?;
        let packed = pack_parameters(parameters, data.output_emotion_length);
        let preferred = parameters
            .iter()
            .flat_map(|value| value.preferred_emotion.iter().copied())
            .collect::<Vec<_>>();
        let mut result = Self {
            module: device.load_module(emotion_postprocess_ptx())?,
            data,
            host_parameters: parameters.to_vec(),
            correspondence: upload(device, stream, &correspondence)?,
            parameters: upload(device, stream, &packed)?,
            preferred: upload(device, stream, &preferred)?,
            state: device.allocate(state_elements)?,
            active: device.allocate(active_words)?,
            parameter_stride,
            state_stride,
            track_count: parameters.len(),
        };
        result.active.copy_from(&vec![0; active_words], stream)?;
        result.reset_all(stream)?;
        Ok(result)
    }

    pub const fn track_count(&self) -> usize {
        self.track_count
    }

    pub fn selected_path(&self) -> GpuEmotionKernelPath {
        if self.data.inference_emotion_length <= 32 && self.data.output_emotion_length <= 32 {
            GpuEmotionKernelPath::Local
        } else {
            GpuEmotionKernelPath::Generic
        }
    }

    pub fn set_parameters(
        &mut self,
        track: usize,
        parameters: EmotionPostProcessParameters,
        stream: &CudaStream,
    ) -> Result<()> {
        parameters.validate(&self.data)?;
        let current = self
            .host_parameters
            .get_mut(track)
            .ok_or_else(|| invalid("GPU emotion parameter track is out of range"))?;
        *current = parameters;
        self.parameters.copy_from(
            &pack_parameters(&self.host_parameters, self.data.output_emotion_length),
            stream,
        )?;
        let preferred = self
            .host_parameters
            .iter()
            .flat_map(|value| value.preferred_emotion.iter().copied())
            .collect::<Vec<_>>();
        self.preferred.copy_from(&preferred, stream)
    }

    pub fn set_preferred(
        &mut self,
        track: usize,
        preferred: &[f32],
        stream: &CudaStream,
    ) -> Result<()> {
        if preferred.len() != self.data.output_emotion_length
            || preferred.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("invalid GPU preferred emotion"));
        }
        self.host_parameters
            .get_mut(track)
            .ok_or_else(|| invalid("GPU preferred emotion track is out of range"))?
            .preferred_emotion
            .clone_from_slice(preferred);
        let values = self
            .host_parameters
            .iter()
            .flat_map(|value| value.preferred_emotion.iter().copied())
            .collect::<Vec<_>>();
        self.preferred.copy_from(&values, stream)
    }

    pub fn reset_all(&mut self, stream: &CudaStream) -> Result<()> {
        let function = self.module.function("emotion_postprocess_reset")?;
        let mut state = self.state.view().as_raw();
        let mut state_stride = self.state_stride as u64;
        let mut track_count = u32::try_from(self.track_count)
            .map_err(|_| invalid("GPU emotion track count exceeds CUDA index range"))?;
        let mut arguments = params![state, state_stride, track_count];
        // SAFETY: the ABI matches the kernel and state has track_count strides.
        unsafe {
            function.launch_raw(
                (track_count.div_ceil(BLOCK_SIZE), 1, 1),
                (BLOCK_SIZE, 1, 1),
                0,
                stream,
                &mut arguments,
            )?;
        }
        stream.synchronize()
    }

    /// Enqueues post-processing. Input rows are packed in `active_tracks`
    /// order; output rows retain their full track index.
    ///
    /// The returned fence borrows processor, input, output, and stream until
    /// completion, preventing mutation or destruction of asynchronous state.
    pub fn enqueue<'a>(
        &'a mut self,
        input: &'a DeviceBuffer<f32>,
        input_stride: usize,
        output: &'a mut DeviceBuffer<f32>,
        output_stride: usize,
        active_tracks: &[usize],
        stream: &'a CudaStream,
    ) -> Result<GpuEmotionPostProcessFence<'a>> {
        ensure_same_device(stream.device_id(), input.device_id())?;
        ensure_same_device(stream.device_id(), output.device_id())?;
        if input_stride < self.data.inference_emotion_length
            || output_stride < self.data.output_emotion_length
            || input.len() < input_stride.saturating_mul(active_tracks.len())
            || output.len() < output_stride.saturating_mul(self.track_count)
        {
            return Err(invalid("GPU emotion input/output layout is too small"));
        }
        let mut active_bits = vec![0_u64; self.active.len()];
        for &track in active_tracks {
            if track >= self.track_count {
                return Err(invalid("GPU emotion active track is invalid or duplicated"));
            }
            let word = track / u64::BITS as usize;
            let bit = 1_u64 << (track % u64::BITS as usize);
            if active_bits[word] & bit != 0 {
                return Err(invalid("GPU emotion active track is invalid or duplicated"));
            }
            active_bits[word] |= bit;
        }
        let set = self.module.function("emotion_postprocess_set")?;
        for (word, value) in active_bits.into_iter().enumerate() {
            let mut active_pointer = self.active.view().as_raw();
            let mut active_word = u32::try_from(word)
                .map_err(|_| invalid("GPU emotion active mask exceeds CUDA index range"))?;
            let mut active_value = value;
            let mut set_arguments = params![active_pointer, active_word, active_value];
            // SAFETY: the scalar ABI matches and active owns the indexed writable u64.
            unsafe { set.launch_raw((1, 1, 1), (1, 1, 1), 0, stream, &mut set_arguments)? };
        }

        let path = self.selected_path();
        let function = self.module.function(match path {
            GpuEmotionKernelPath::Local => "emotion_postprocess_local",
            GpuEmotionKernelPath::Generic => "emotion_postprocess_generic",
        })?;
        let mut output_pointer = output.view().as_raw();
        let mut output_stride = output_stride as u64;
        let mut input_pointer = input.view().as_raw();
        let mut input_stride = input_stride as u64;
        let mut correspondence = self.correspondence.view().as_raw();
        let mut parameters = self.parameters.view().as_raw();
        let mut parameter_stride = self.parameter_stride as u64;
        let mut preferred = self.preferred.view().as_raw();
        let mut state = self.state.view().as_raw();
        let mut state_stride = self.state_stride as u64;
        let mut active = self.active.view().as_raw();
        let mut input_length = self.data.inference_emotion_length as u32;
        let mut output_length = self.data.output_emotion_length as u32;
        let mut track_count = u32::try_from(self.track_count)
            .map_err(|_| invalid("GPU emotion track count exceeds CUDA index range"))?;
        let mut arguments = params![
            output_pointer,
            output_stride,
            input_pointer,
            input_stride,
            correspondence,
            parameters,
            parameter_stride,
            preferred,
            state,
            state_stride,
            active,
            input_length,
            output_length,
            track_count
        ];
        let (grid, block) = match path {
            GpuEmotionKernelPath::Local => ((track_count, 1, 1), (32, 1, 1)),
            GpuEmotionKernelPath::Generic => {
                ((track_count.div_ceil(BLOCK_SIZE), 1, 1), (BLOCK_SIZE, 1, 1))
            }
        };
        // SAFETY: all kernel arguments point to validated, same-device allocations;
        // the returned fence retains exclusive state/output access through completion.
        unsafe { function.launch_raw(grid, block, 0, stream, &mut arguments)? };
        let event = stream.create_event()?;
        event.record(stream)?;
        Ok(GpuEmotionPostProcessFence {
            event,
            path,
            _resources: PhantomData,
        })
    }
}

pub struct GpuEmotionPostProcessFence<'a> {
    event: CudaEvent,
    path: GpuEmotionKernelPath,
    _resources: PhantomData<(
        &'a mut GpuEmotionPostProcessor,
        &'a DeviceBuffer<f32>,
        &'a mut DeviceBuffer<f32>,
        &'a CudaStream,
    )>,
}

impl GpuEmotionPostProcessFence<'_> {
    pub const fn path(&self) -> GpuEmotionKernelPath {
        self.path
    }

    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }
}

fn upload<T: Copy>(
    device: &Arc<GpuDevice>,
    stream: &CudaStream,
    values: &[T],
) -> Result<DeviceBuffer<T>> {
    let mut result = device.allocate(values.len())?;
    result.copy_from(values, stream)?;
    Ok(result)
}

fn pack_parameters(parameters: &[EmotionPostProcessParameters], output_length: usize) -> Vec<f32> {
    let mut packed = Vec::with_capacity(parameters.len() * (PARAMETER_PREFIX + output_length));
    for value in parameters {
        packed.extend([
            value.emotion_contrast,
            value.max_emotions as f32,
            value.live_blend_coefficient,
            u8::from(value.enable_preferred_emotion) as f32,
            value.preferred_emotion_strength,
            value.live_transition_time,
            value.fixed_dt,
            value.emotion_strength,
        ]);
        packed.extend_from_slice(&value.beginning_emotion);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emotion::EmotionPostProcessor;

    fn run_case(size: usize) {
        let Ok(device) = GpuDevice::new(0) else {
            return;
        };
        let stream = device.create_stream().unwrap();
        let data = EmotionPostProcessData {
            inference_emotion_length: size,
            output_emotion_length: size,
            emotion_correspondence: (0..size as i64).collect(),
        };
        let parameters = EmotionPostProcessParameters {
            max_emotions: size.saturating_sub(1),
            beginning_emotion: vec![0.1; size],
            preferred_emotion: vec![0.2; size],
            enable_preferred_emotion: true,
            live_transition_time: 0.25,
            fixed_dt: 1.0 / 30.0,
            ..EmotionPostProcessParameters::default()
        };
        let input = (0..size)
            .map(|index| index as f32 / size as f32)
            .collect::<Vec<_>>();
        let expected = EmotionPostProcessor::new(data.clone(), parameters.clone())
            .unwrap()
            .process(&input)
            .unwrap();
        let mut gpu = GpuEmotionPostProcessor::new(&device, &stream, data, &[parameters]).unwrap();
        let device_input = upload(&device, &stream, &input).unwrap();
        let mut device_output = device.allocate::<f32>(size).unwrap();
        let fence = gpu
            .enqueue(&device_input, size, &mut device_output, size, &[0], &stream)
            .unwrap();
        assert_eq!(
            fence.path(),
            if size <= 32 {
                GpuEmotionKernelPath::Local
            } else {
                GpuEmotionKernelPath::Generic
            }
        );
        fence.synchronize().unwrap();
        drop(fence);
        let mut actual = vec![0.0; size];
        device_output.copy_to(&mut actual, &stream).unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn selects_local_at_32_and_generic_above_32_with_cpu_parity() {
        run_case(32);
        run_case(33);
    }

    #[test]
    fn inactive_tracks_preserve_state_and_reset_replays_first_frame() {
        let Ok(device) = GpuDevice::new(0) else {
            return;
        };
        let stream = device.create_stream().unwrap();
        let data = EmotionPostProcessData {
            inference_emotion_length: 3,
            output_emotion_length: 3,
            emotion_correspondence: vec![0, 1, 2],
        };
        let parameters = EmotionPostProcessParameters {
            max_emotions: 3,
            beginning_emotion: vec![0.1, 0.2, 0.3],
            preferred_emotion: vec![0.0; 3],
            live_transition_time: 0.25,
            fixed_dt: 1.0 / 30.0,
            emotion_strength: 1.0,
            ..EmotionPostProcessParameters::default()
        };
        let mut first_cpu = EmotionPostProcessor::new(data.clone(), parameters.clone()).unwrap();
        let mut second_cpu = EmotionPostProcessor::new(data.clone(), parameters.clone()).unwrap();
        let first_input = [1.0, 0.0, -1.0];
        let next_input = [-1.0, 0.0, 1.0];
        let first_expected = first_cpu.process(&first_input).unwrap();
        let second_expected = second_cpu.process(&next_input).unwrap();
        let continued_expected = first_cpu.process(&next_input).unwrap();
        let mut gpu =
            GpuEmotionPostProcessor::new(&device, &stream, data, &[parameters.clone(), parameters])
                .unwrap();
        let mut input = upload(&device, &stream, &first_input).unwrap();
        let mut output = device.allocate::<f32>(6).unwrap();
        output.copy_from(&[0.0; 6], &stream).unwrap();

        let fence = gpu
            .enqueue(&input, 3, &mut output, 3, &[0], &stream)
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        input.copy_from(&next_input, &stream).unwrap();
        let fence = gpu
            .enqueue(&input, 3, &mut output, 3, &[1], &stream)
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        let mut actual = vec![0.0; 6];
        output.copy_to(&mut actual, &stream).unwrap();
        assert_close(&actual[..3], &first_expected);
        assert_close(&actual[3..], &second_expected);

        let fence = gpu
            .enqueue(&input, 3, &mut output, 3, &[0], &stream)
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        output.copy_to(&mut actual, &stream).unwrap();
        assert_close(&actual[..3], &continued_expected);

        gpu.reset_all(&stream).unwrap();
        input.copy_from(&first_input, &stream).unwrap();
        let fence = gpu
            .enqueue(&input, 3, &mut output, 3, &[0], &stream)
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        output.copy_to(&mut actual, &stream).unwrap();
        assert_close(&actual[..3], &first_expected);
    }

    #[test]
    fn active_mask_and_packed_input_cross_u64_boundaries() {
        let Ok(device) = GpuDevice::new(0) else {
            return;
        };
        let stream = device.create_stream().unwrap();
        let data = EmotionPostProcessData {
            inference_emotion_length: 3,
            output_emotion_length: 3,
            emotion_correspondence: vec![0, 1, 2],
        };
        let parameters = EmotionPostProcessParameters {
            max_emotions: 3,
            beginning_emotion: vec![0.0; 3],
            preferred_emotion: vec![0.0; 3],
            live_blend_coefficient: 0.0,
            live_transition_time: 0.0,
            fixed_dt: 1.0 / 30.0,
            emotion_strength: 1.0,
            ..EmotionPostProcessParameters::default()
        };
        let active_tracks = [0, 63, 64, 127];
        let packed_input = active_tracks
            .iter()
            .flat_map(|track| [*track as f32 / 64.0, 0.25, -(*track as f32) / 128.0])
            .collect::<Vec<_>>();
        let expected = packed_input
            .chunks_exact(3)
            .map(|input| {
                EmotionPostProcessor::new(data.clone(), parameters.clone())
                    .unwrap()
                    .process(input)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut gpu =
            GpuEmotionPostProcessor::new(&device, &stream, data, &vec![parameters; 128]).unwrap();
        let input = upload(&device, &stream, &packed_input).unwrap();
        let mut output = device.allocate::<f32>(128 * 3).unwrap();
        output.copy_from(&vec![-1.0; 128 * 3], &stream).unwrap();

        let fence = gpu
            .enqueue(&input, 3, &mut output, 3, &active_tracks, &stream)
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);

        let mut actual = vec![0.0; 128 * 3];
        output.copy_to(&mut actual, &stream).unwrap();
        for (packed, track) in active_tracks.into_iter().enumerate() {
            assert_close(&actual[track * 3..track * 3 + 3], &expected[packed]);
        }
        for track in 0..128 {
            if !active_tracks.contains(&track) {
                assert_eq!(&actual[track * 3..track * 3 + 3], &[-1.0; 3]);
            }
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 2.0e-6, "{actual} != {expected}");
        }
    }
}
