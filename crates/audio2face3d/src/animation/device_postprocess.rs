use crate::animation::{
    EyesAnimatorParams, JawParameters, RegressionResultLayout, SkinAnimatorParams,
    TongueAnimatorParams,
};
use crate::common::{Error, Result};
use crate::cuda::{
    CudaEvent, CudaModule, CudaStream, DeviceBuffer, DeviceView, GpuDevice, ensure_same_device,
    regression_jaw_ptx, regression_postprocess_ptx,
};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 256;
const SKIN_PARAM_STRIDE: usize = 9;
const TONGUE_PARAM_STRIDE: usize = 3;
const EYES_PARAM_STRIDE: usize = 7;
const JAW_PARAM_STRIDE: usize = 3;

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

/// Host-side geometry shared by every track in one regression model.
pub struct GpuRegressionModel<'a> {
    pub skin_neutral_pose: &'a [f32],
    pub skin_lip_open_delta: &'a [f32],
    pub skin_eye_close_delta: &'a [f32],
    pub tongue_neutral_pose: &'a [f32],
    pub jaw_neutral_pose: &'a [f32],
    pub saccade_rotation: &'a [f32],
}

/// Per-track animator controls uploaded once and retained on the device.
#[derive(Clone, Copy)]
pub struct GpuRegressionTrackParams {
    pub skin: SkinAnimatorParams,
    pub tongue: TongueAnimatorParams,
    pub jaw: JawParameters,
    pub eyes: EyesAnimatorParams,
}

/// Device output allocations. Each allocation uses track-major layout.
pub struct GpuRegressionOutputs<'a> {
    pub skin: &'a mut DeviceBuffer<f32>,
    pub tongue: &'a mut DeviceBuffer<f32>,
    pub jaw_transforms: &'a mut DeviceBuffer<f32>,
    pub eyes_rotations: &'a mut DeviceBuffer<f32>,
}

/// Owns immutable model data and persistent per-track animation state.
pub struct GpuRegressionPostprocessor {
    module: CudaModule,
    jaw_module: CudaModule,
    skin_animator_data: DeviceBuffer<f32>,
    face_mask_lower: DeviceBuffer<f32>,
    tongue_neutral_pose: DeviceBuffer<f32>,
    jaw_neutral_pose: DeviceBuffer<f32>,
    saccade_rotation: DeviceBuffer<f32>,
    skin_params: DeviceBuffer<f32>,
    tongue_params: DeviceBuffer<f32>,
    jaw_params: DeviceBuffer<f32>,
    eyes_params: DeviceBuffer<f32>,
    active_tracks: DeviceBuffer<u64>,
    initialized_tracks: DeviceBuffer<u64>,
    skin_interp: DeviceBuffer<f32>,
    eyes_live_time: DeviceBuffer<f32>,
    active_staging: Vec<u64>,
    track_count: usize,
    skin_pose_size: usize,
    tongue_pose_size: usize,
    jaw_pose_size: usize,
    dt: f32,
}

/// Device-resident Regression PCA reconstruction followed by the shared
/// geometry animator pipeline.
pub(crate) struct GpuRegressionPcaPostprocessor {
    postprocessor: GpuRegressionPostprocessor,
    blas: crate::cuda::CublasHandle,
    skin_shapes: DeviceBuffer<f32>,
    tongue_shapes: DeviceBuffer<f32>,
    skin_temporary: DeviceBuffer<f32>,
    tongue_temporary: DeviceBuffer<f32>,
    expanded: DeviceBuffer<f32>,
    raw_layout: RegressionResultLayout,
    expanded_stride: usize,
    track_count: usize,
}

impl GpuRegressionPcaPostprocessor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        device: &Arc<GpuDevice>,
        stream: &CudaStream,
        model: GpuRegressionModel<'_>,
        tracks: &[GpuRegressionTrackParams],
        dt: f32,
        skin_shapes: &[f32],
        tongue_shapes: &[f32],
        raw_layout: RegressionResultLayout,
    ) -> Result<Self> {
        let track_count = tracks.len();
        let skin_size = model.skin_neutral_pose.len();
        let tongue_size = model.tongue_neutral_pose.len();
        if skin_shapes.len() != skin_size.saturating_mul(raw_layout.skin)
            || tongue_shapes.len() != tongue_size.saturating_mul(raw_layout.tongue)
        {
            return Err(invalid("Regression PCA matrix dimensions mismatch"));
        }
        let expanded_stride = skin_size
            .checked_add(tongue_size)
            .and_then(|value| value.checked_add(raw_layout.jaw))
            .and_then(|value| value.checked_add(raw_layout.eyes))
            .ok_or_else(|| invalid("Regression expanded result size overflow"))?;
        let upload = |values: &[f32]| -> Result<DeviceBuffer<f32>> {
            let mut output = device.allocate(values.len())?;
            output.copy_from(values, stream)?;
            Ok(output)
        };
        Ok(Self {
            postprocessor: GpuRegressionPostprocessor::new(device, stream, model, tracks, dt)?,
            blas: crate::cuda::CublasHandle::new(stream)?,
            skin_shapes: upload(skin_shapes)?,
            tongue_shapes: upload(tongue_shapes)?,
            skin_temporary: device.allocate(skin_size)?,
            tongue_temporary: device.allocate(tongue_size)?,
            expanded: device.allocate(expanded_stride.saturating_mul(track_count))?,
            raw_layout,
            expanded_stride,
            track_count,
        })
    }

    pub(crate) fn reset_track(&mut self, track: usize, stream: &CudaStream) -> Result<()> {
        self.postprocessor.reset_track(track, stream)
    }

    pub(crate) fn enqueue<'a>(
        &'a mut self,
        raw: DeviceView<'a, f32>,
        active_tracks: &[usize],
        outputs: GpuRegressionOutputs<'a>,
        stream: &'a CudaStream,
    ) -> Result<GpuRegressionPostprocessFence<'a>> {
        let raw_stride = self.raw_layout.total()?;
        if raw.len() < raw_stride.saturating_mul(self.track_count) {
            return Err(invalid("Regression device result dimensions mismatch"));
        }
        let skin_size = self.skin_temporary.len();
        let tongue_size = self.tongue_temporary.len();
        for &track in active_tracks {
            if track >= self.track_count {
                return Err(invalid("Regression active track is out of range"));
            }
            let raw_base = track * raw_stride;
            let skin_coefficients = raw.slice(raw_base, self.raw_layout.skin)?;
            let fence = self.blas.pca_reconstruct_views(
                self.skin_shapes.view(),
                skin_coefficients,
                &mut self.skin_temporary,
                crate::cuda::PcaDimensions {
                    shape_size: skin_size,
                    shape_count: self.raw_layout.skin,
                    batch_size: 1,
                },
                stream,
            )?;
            fence.synchronize()?;
            drop(fence);
            self.expanded.copy_from_device_range(
                track * self.expanded_stride,
                &self.skin_temporary,
                0,
                skin_size,
                stream,
            )?;

            let tongue_coefficients =
                raw.slice(raw_base + self.raw_layout.skin, self.raw_layout.tongue)?;
            let fence = self.blas.pca_reconstruct_views(
                self.tongue_shapes.view(),
                tongue_coefficients,
                &mut self.tongue_temporary,
                crate::cuda::PcaDimensions {
                    shape_size: tongue_size,
                    shape_count: self.raw_layout.tongue,
                    batch_size: 1,
                },
                stream,
            )?;
            fence.synchronize()?;
            drop(fence);
            let expanded_base = track * self.expanded_stride;
            self.expanded.copy_from_device_range(
                expanded_base + skin_size,
                &self.tongue_temporary,
                0,
                tongue_size,
                stream,
            )?;
            let raw_tail = self.raw_layout.jaw + self.raw_layout.eyes;
            self.expanded.copy_from_device_view_range(
                expanded_base + skin_size + tongue_size,
                raw,
                raw_base + self.raw_layout.skin + self.raw_layout.tongue,
                raw_tail,
                stream,
            )?;
        }
        self.postprocessor.enqueue_view(
            self.expanded.view(),
            self.expanded_stride,
            0,
            active_tracks,
            outputs,
            stream,
        )
    }
}

impl GpuRegressionPostprocessor {
    pub fn new(
        device: &Arc<GpuDevice>,
        stream: &CudaStream,
        model: GpuRegressionModel<'_>,
        tracks: &[GpuRegressionTrackParams],
        dt: f32,
    ) -> Result<Self> {
        if tracks.is_empty() || !dt.is_finite() || dt <= 0.0 {
            return Err(invalid(
                "GPU postprocessor requires tracks and a positive finite dt",
            ));
        }
        validate_pose("skin neutral pose", model.skin_neutral_pose)?;
        validate_pose("tongue neutral pose", model.tongue_neutral_pose)?;
        validate_pose("jaw neutral pose", model.jaw_neutral_pose)?;
        if model.skin_lip_open_delta.len() != model.skin_neutral_pose.len()
            || model.skin_eye_close_delta.len() != model.skin_neutral_pose.len()
        {
            return Err(invalid("skin animator pose lengths must match"));
        }
        if model.saccade_rotation.is_empty() || !model.saccade_rotation.len().is_multiple_of(2) {
            return Err(invalid("saccade rotation must contain XY pairs"));
        }
        let track_count = tracks.len();
        let skin_pose_size = model.skin_neutral_pose.len();
        let tongue_pose_size = model.tongue_neutral_pose.len();
        let jaw_pose_size = model.jaw_neutral_pose.len();
        let word_count = track_count.div_ceil(64);

        let mut animator_data = Vec::with_capacity(skin_pose_size * 3);
        for index in 0..skin_pose_size {
            animator_data.extend([
                model.skin_eye_close_delta[index],
                model.skin_lip_open_delta[index],
                model.skin_neutral_pose[index],
            ]);
        }
        let face_mask_lower = make_face_mask(model.skin_neutral_pose, tracks[0].skin)?;
        let mut skin_params = Vec::with_capacity(track_count * SKIN_PARAM_STRIDE);
        let mut tongue_params = Vec::with_capacity(track_count * TONGUE_PARAM_STRIDE);
        let mut jaw_params = Vec::with_capacity(track_count * JAW_PARAM_STRIDE);
        let mut eyes_params = Vec::with_capacity(track_count * EYES_PARAM_STRIDE);
        for track in tracks {
            if track.skin.face_mask_level != tracks[0].skin.face_mask_level
                || track.skin.face_mask_softness != tracks[0].skin.face_mask_softness
            {
                return Err(invalid("GPU tracks must share face mask controls"));
            }
            skin_params.extend([
                track.skin.skin_strength,
                track.skin.eyelid_open_offset,
                track.skin.blink_offset,
                track.skin.blink_strength,
                track.skin.lip_open_offset,
                smoothing_alpha(track.skin.lower_face_smoothing, dt)?,
                smoothing_alpha(track.skin.upper_face_smoothing, dt)?,
                track.skin.lower_face_strength,
                track.skin.upper_face_strength,
            ]);
            tongue_params.extend([
                track.tongue.tongue_strength,
                track.tongue.tongue_height_offset,
                track.tongue.tongue_depth_offset,
            ]);
            jaw_params.extend([
                track.jaw.strength,
                track.jaw.height_offset,
                track.jaw.depth_offset,
            ]);
            eyes_params.extend([
                track.eyes.eyeballs_strength,
                track.eyes.saccade_strength,
                track.eyes.right_eyeball_rotation_offset_x,
                track.eyes.right_eyeball_rotation_offset_y,
                track.eyes.left_eyeball_rotation_offset_x,
                track.eyes.left_eyeball_rotation_offset_y,
                track.eyes.saccade_seed,
            ]);
        }

        let upload = |values: &[f32]| -> Result<DeviceBuffer<f32>> {
            let mut buffer = device.allocate(values.len())?;
            buffer.copy_from(values, stream)?;
            Ok(buffer)
        };
        let mut active_tracks = device.allocate(word_count)?;
        active_tracks.copy_from(&vec![0_u64; word_count], stream)?;
        let initialized_tracks = zeroed::<u64>(device, stream, word_count)?;
        let skin_interp = zeroed::<f32>(device, stream, track_count * skin_pose_size * 4)?;
        let eyes_live_time = zeroed::<f32>(device, stream, track_count)?;

        Ok(Self {
            module: device.load_module(regression_postprocess_ptx())?,
            jaw_module: device.load_module(regression_jaw_ptx())?,
            skin_animator_data: upload(&animator_data)?,
            face_mask_lower: upload(&face_mask_lower)?,
            tongue_neutral_pose: upload(model.tongue_neutral_pose)?,
            jaw_neutral_pose: upload(model.jaw_neutral_pose)?,
            saccade_rotation: upload(model.saccade_rotation)?,
            skin_params: upload(&skin_params)?,
            tongue_params: upload(&tongue_params)?,
            jaw_params: upload(&jaw_params)?,
            eyes_params: upload(&eyes_params)?,
            active_tracks,
            initialized_tracks,
            skin_interp,
            eyes_live_time,
            active_staging: vec![0; word_count],
            track_count,
            skin_pose_size,
            tongue_pose_size,
            jaw_pose_size,
            dt,
        })
    }

    pub const fn track_count(&self) -> usize {
        self.track_count
    }

    /// Resets IIR initialization and eye time for every track.
    pub fn reset(&mut self, stream: &CudaStream) -> Result<()> {
        zero_buffer(&mut self.initialized_tracks, stream)?;
        zero_buffer(&mut self.skin_interp, stream)?;
        zero_buffer(&mut self.eyes_live_time, stream)
    }

    pub(crate) fn reset_track(&mut self, track: usize, stream: &CudaStream) -> Result<()> {
        if track >= self.track_count {
            return Err(invalid("GPU postprocessor reset track is out of range"));
        }
        let function = self.module.function("audio2face3d_reset_track")?;
        let mut args = [
            self.initialized_tracks.view().as_raw(),
            self.skin_interp.view().as_raw(),
            4,
            self.skin_pose_size as u64,
            self.eyes_live_time.view().as_raw(),
            track as u64,
        ];
        launch(&function, self.skin_pose_size.max(1), stream, &mut args)?;
        stream.synchronize()
    }

    /// Enqueues skin, tongue, jaw and eyes kernels on one stream.
    ///
    /// The returned fence borrows all input/output/state allocations until the
    /// recorded completion event has synchronized.
    pub fn enqueue<'a>(
        &'a mut self,
        network_result: &'a DeviceBuffer<f32>,
        input_stride: usize,
        active_tracks: &[usize],
        outputs: GpuRegressionOutputs<'a>,
        stream: &'a CudaStream,
    ) -> Result<GpuRegressionPostprocessFence<'a>> {
        self.enqueue_view(
            network_result.view(),
            input_stride,
            0,
            active_tracks,
            outputs,
            stream,
        )
    }

    pub(crate) fn enqueue_view<'a>(
        &'a mut self,
        network_result: DeviceView<'a, f32>,
        input_stride: usize,
        input_offset: usize,
        active_tracks: &[usize],
        outputs: GpuRegressionOutputs<'a>,
        stream: &'a CudaStream,
    ) -> Result<GpuRegressionPostprocessFence<'a>> {
        let minimum_stride = self.skin_pose_size + self.tongue_pose_size + self.jaw_pose_size + 4;
        let expected_input = self
            .track_count
            .checked_sub(1)
            .and_then(|tracks| tracks.checked_mul(input_stride))
            .and_then(|elements| elements.checked_add(input_offset))
            .and_then(|elements| elements.checked_add(minimum_stride))
            .ok_or_else(|| invalid("network result size overflow"))?;
        ensure_same_device(stream.device_id(), network_result.device_id())?;
        ensure_same_device(stream.device_id(), outputs.skin.device_id())?;
        ensure_same_device(stream.device_id(), outputs.tongue.device_id())?;
        ensure_same_device(stream.device_id(), outputs.jaw_transforms.device_id())?;
        ensure_same_device(stream.device_id(), outputs.eyes_rotations.device_id())?;
        if input_stride < minimum_stride || network_result.len() < expected_input {
            return Err(invalid("network result layout is too small"));
        }
        validate_output(outputs.skin, self.skin_pose_size, self.track_count, "skin")?;
        validate_output(
            outputs.tongue,
            self.tongue_pose_size,
            self.track_count,
            "tongue",
        )?;
        validate_output(outputs.jaw_transforms, 16, self.track_count, "jaw")?;
        validate_output(outputs.eyes_rotations, 6, self.track_count, "eyes")?;
        self.active_staging.fill(0);
        for &track in active_tracks {
            if track >= self.track_count {
                return Err(invalid(format!("active track {track} is out of range")));
            }
            self.active_staging[track / 64] |= 1_u64 << (track % 64);
        }
        // SAFETY: the returned fence borrows self, retaining both staging and
        // device storage until all queued work has completed.
        unsafe {
            self.active_tracks
                .copy_from_async(&self.active_staging, stream)?;
        }

        let skin_offset = input_offset;
        let tongue_offset = skin_offset + self.skin_pose_size;
        let jaw_offset = tongue_offset + self.tongue_pose_size;
        let eyes_offset = jaw_offset + self.jaw_pose_size;
        self.launch_skin(
            network_result,
            input_stride,
            outputs.skin,
            skin_offset,
            stream,
        )?;
        self.mark_initialized(stream)?;
        self.launch_tongue(
            network_result,
            input_stride,
            outputs.tongue,
            tongue_offset,
            stream,
        )?;
        self.launch_jaw(
            network_result,
            input_stride,
            outputs.jaw_transforms,
            jaw_offset,
            active_tracks,
            stream,
        )?;
        self.launch_eyes(
            network_result,
            input_stride,
            outputs.eyes_rotations,
            eyes_offset,
            stream,
        )?;
        let event = stream.create_event()?;
        event.record(stream)?;
        Ok(GpuRegressionPostprocessFence {
            event,
            _resources: PhantomData,
        })
    }

    fn launch_skin(
        &mut self,
        input: DeviceView<'_, f32>,
        stride: usize,
        output: &mut DeviceBuffer<f32>,
        offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self.module.function("audio2face3d_skin_postprocess")?;
        let mut args = [
            output.view().as_raw(),
            0,
            self.skin_pose_size as u64,
            input.as_raw(),
            offset as u64,
            stride as u64,
            self.skin_animator_data.view().as_raw(),
            3,
            self.face_mask_lower.view().as_raw(),
            self.skin_interp.view().as_raw(),
            4,
            self.skin_params.view().as_raw(),
            SKIN_PARAM_STRIDE as u64,
            self.active_tracks.view().as_raw(),
            self.initialized_tracks.view().as_raw(),
            self.skin_pose_size as u64,
            self.track_count as u64,
        ];
        launch(
            &function,
            self.skin_pose_size * self.track_count,
            stream,
            &mut args,
        )
    }
    fn mark_initialized(&self, stream: &CudaStream) -> Result<()> {
        let function = self.module.function("audio2face3d_mark_initialized")?;
        let mut args = [
            self.initialized_tracks.view().as_raw(),
            self.active_tracks.view().as_raw(),
            self.active_staging.len() as u64,
        ];
        launch(&function, self.active_staging.len(), stream, &mut args)
    }
    fn launch_tongue(
        &self,
        input: DeviceView<'_, f32>,
        stride: usize,
        output: &mut DeviceBuffer<f32>,
        offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self.module.function("audio2face3d_tongue_postprocess")?;
        let mut args = [
            output.view().as_raw(),
            0,
            self.tongue_pose_size as u64,
            input.as_raw(),
            offset as u64,
            stride as u64,
            self.tongue_neutral_pose.view().as_raw(),
            self.tongue_params.view().as_raw(),
            TONGUE_PARAM_STRIDE as u64,
            self.active_tracks.view().as_raw(),
            self.tongue_pose_size as u64,
            self.track_count as u64,
        ];
        launch(
            &function,
            self.tongue_pose_size * self.track_count,
            stream,
            &mut args,
        )
    }
    fn launch_jaw(
        &self,
        input: DeviceView<'_, f32>,
        stride: usize,
        output: &mut DeviceBuffer<f32>,
        offset: usize,
        active_tracks: &[usize],
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self.jaw_module.function("audio2face_regression_jaw")?;
        let point_count = u32::try_from(self.jaw_pose_size / 3)
            .map_err(|_| invalid("jaw point count exceeds u32"))?;
        for &track in active_tracks {
            let input_element = track
                .checked_mul(stride)
                .and_then(|base| base.checked_add(offset))
                .ok_or_else(|| invalid("jaw input offset overflow"))?;
            let input_pointer = input.as_raw() + (input_element * size_of::<f32>()) as u64;
            let params_pointer = self.jaw_params.view().as_raw()
                + (track * JAW_PARAM_STRIDE * size_of::<f32>()) as u64;
            let output_pointer = output.view().as_raw() + (track * 16 * size_of::<f32>()) as u64;
            // One launch per active track keeps the jaw delta slice contiguous;
            // the kernel itself intentionally has no input-stride parameter.
            let mut args = [
                self.jaw_neutral_pose.view().as_raw(),
                input_pointer,
                u64::from(point_count),
                1,
                params_pointer,
                JAW_PARAM_STRIDE as u64,
                0,
                output_pointer,
            ];
            launch(&function, 1, stream, &mut args)?;
        }
        Ok(())
    }
    fn launch_eyes(
        &self,
        input: DeviceView<'_, f32>,
        stride: usize,
        output: &mut DeviceBuffer<f32>,
        offset: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self.module.function("audio2face3d_eyes_postprocess")?;
        let dt_bits = u64::from(self.dt.to_bits());
        let mut args = [
            output.view().as_raw(),
            0,
            6,
            input.as_raw(),
            offset as u64,
            stride as u64,
            self.eyes_params.view().as_raw(),
            EYES_PARAM_STRIDE as u64,
            self.saccade_rotation.view().as_raw(),
            self.saccade_rotation.len() as u64,
            dt_bits,
            self.eyes_live_time.view().as_raw(),
            self.active_tracks.view().as_raw(),
            self.track_count as u64,
        ];
        launch(&function, self.track_count * 6, stream, &mut args)
    }
}

pub struct GpuRegressionPostprocessFence<'a> {
    event: CudaEvent,
    _resources: PhantomData<(
        &'a mut GpuRegressionPostprocessor,
        &'a CudaStream,
        DeviceView<'a, f32>,
        &'a mut DeviceBuffer<f32>,
    )>,
}
impl GpuRegressionPostprocessFence<'_> {
    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }
}

fn validate_pose(name: &str, values: &[f32]) -> Result<()> {
    if values.is_empty() || !values.len().is_multiple_of(3) || values.iter().any(|v| !v.is_finite())
    {
        Err(invalid(format!("{name} must contain finite XYZ vertices")))
    } else {
        Ok(())
    }
}
fn smoothing_alpha(smoothing: f32, dt: f32) -> Result<f32> {
    if !smoothing.is_finite() || smoothing < 0.0 {
        return Err(invalid("skin smoothing must be finite and non-negative"));
    }
    Ok(if smoothing > 0.0 {
        1.0 - 0.5_f32.powf(dt / smoothing)
    } else {
        0.0
    })
}
fn make_face_mask(pose: &[f32], params: SkinAnimatorParams) -> Result<Vec<f32>> {
    if !params.face_mask_softness.is_finite() || params.face_mask_softness <= 0.0 {
        return Err(invalid("face mask softness must be positive"));
    }
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for point in pose.chunks_exact(3) {
        min = min.min(point[1]);
        max = max.max(point[1]);
    }
    Ok(pose
        .chunks_exact(3)
        .map(|point| {
            let normalized = if max > min {
                (point[1] - min) / (max - min)
            } else {
                0.0
            };
            1.0 / (1.0 + (-(params.face_mask_level - normalized) / params.face_mask_softness).exp())
        })
        .collect())
}
fn zeroed<T>(device: &Arc<GpuDevice>, stream: &CudaStream, len: usize) -> Result<DeviceBuffer<T>> {
    let mut buffer = device.allocate(len)?;
    zero_buffer(&mut buffer, stream)?;
    Ok(buffer)
}
fn zero_buffer<T>(buffer: &mut DeviceBuffer<T>, stream: &CudaStream) -> Result<()> {
    // SAFETY: `buffer` owns the complete writable allocation and synchronization
    // below keeps it alive until the memset has completed.
    unsafe {
        stream.memset_device_zero(buffer.view().as_raw(), buffer.len() * size_of::<T>())?;
    }
    stream.synchronize()
}
fn validate_output(
    buffer: &DeviceBuffer<f32>,
    stride: usize,
    tracks: usize,
    name: &str,
) -> Result<()> {
    if buffer.len() < stride * tracks {
        Err(invalid(format!("{name} output is too small")))
    } else {
        Ok(())
    }
}
fn launch(
    function: &crate::cuda::CudaFunction<'_>,
    work: usize,
    stream: &CudaStream,
    args: &mut [u64],
) -> Result<()> {
    let blocks = u32::try_from(work.div_ceil(BLOCK_SIZE as usize))
        .map_err(|_| invalid("CUDA grid exceeds u32"))?;
    let mut params: Vec<*mut c_void> = args
        .iter_mut()
        .map(|arg| (arg as *mut u64).cast())
        .collect();
    // SAFETY: every argument occupies an aligned u64 slot (the CUDA driver reads
    // the low 32 bits for f32/u32 arguments), and enqueue's fence retains every
    // referenced allocation until the recorded stream work completes.
    unsafe { function.launch_raw((blocks, 1, 1), (BLOCK_SIZE, 1, 1), 0, stream, &mut params) }
}
use std::mem::size_of;

#[cfg(test)]
mod tests {
    use super::*;

    fn track_params() -> GpuRegressionTrackParams {
        GpuRegressionTrackParams {
            skin: SkinAnimatorParams {
                lower_face_smoothing: 0.0,
                upper_face_smoothing: 0.0,
                lower_face_strength: 1.0,
                upper_face_strength: 1.0,
                face_mask_level: 0.5,
                face_mask_softness: 0.1,
                skin_strength: 2.0,
                blink_strength: 0.0,
                eyelid_open_offset: 0.0,
                lip_open_offset: 0.0,
                blink_offset: 0.0,
            },
            tongue: TongueAnimatorParams {
                tongue_strength: 2.0,
                tongue_height_offset: 10.0,
                tongue_depth_offset: 20.0,
            },
            jaw: JawParameters::default(),
            eyes: EyesAnimatorParams {
                eyeballs_strength: 2.0,
                saccade_strength: 0.0,
                right_eyeball_rotation_offset_x: 1.0,
                right_eyeball_rotation_offset_y: 2.0,
                left_eyeball_rotation_offset_x: 3.0,
                left_eyeball_rotation_offset_y: 4.0,
                saccade_seed: 0.0,
            },
        }
    }

    #[test]
    fn gpu_matches_small_cpu_oracles() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let skin_neutral = [0.0, 0.0, 0.0];
        let tongue_neutral = [1.0, 2.0, 3.0];
        let jaw_neutral = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let mut postprocessor = GpuRegressionPostprocessor::new(
            &device,
            &stream,
            GpuRegressionModel {
                skin_neutral_pose: &skin_neutral,
                skin_lip_open_delta: &[0.0; 3],
                skin_eye_close_delta: &[0.0; 3],
                tongue_neutral_pose: &tongue_neutral,
                jaw_neutral_pose: &jaw_neutral,
                saccade_rotation: &[0.0, 0.0],
            },
            &[track_params()],
            1.0 / 30.0,
        )
        .unwrap();
        let host_input = [
            1.0, 2.0, 3.0, // skin
            0.5, 1.0, 1.5, // tongue
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // jaw
            1.0, 2.0, 3.0, 4.0, // eyes
        ];
        let mut input = device.allocate(host_input.len()).unwrap();
        input.copy_from(&host_input, &stream).unwrap();
        let mut skin = device.allocate(3).unwrap();
        let mut tongue = device.allocate(3).unwrap();
        let mut jaw = device.allocate(16).unwrap();
        let mut eyes = device.allocate(6).unwrap();
        let fence = postprocessor
            .enqueue(
                &input,
                host_input.len(),
                &[0],
                GpuRegressionOutputs {
                    skin: &mut skin,
                    tongue: &mut tongue,
                    jaw_transforms: &mut jaw,
                    eyes_rotations: &mut eyes,
                },
                &stream,
            )
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        let mut actual_skin = [0.0; 3];
        let mut actual_tongue = [0.0; 3];
        let mut actual_jaw = [0.0; 16];
        let mut actual_eyes = [0.0; 6];
        skin.copy_to(&mut actual_skin, &stream).unwrap();
        tongue.copy_to(&mut actual_tongue, &stream).unwrap();
        jaw.copy_to(&mut actual_jaw, &stream).unwrap();
        eyes.copy_to(&mut actual_eyes, &stream).unwrap();
        assert_eq!(actual_skin, [2.0, 4.0, 6.0]);
        assert_eq!(actual_tongue, [2.0, 14.0, 26.0]);
        assert_eq!(actual_eyes, [3.0, 6.0, 0.0, 9.0, 12.0, 0.0]);
        let expected_jaw = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        for (actual, expected) in actual_jaw.into_iter().zip(expected_jaw) {
            assert!((actual - expected).abs() < 1.0e-5);
        }

        postprocessor.reset_track(0, &stream).unwrap();
        let mut framed_input = vec![0.0; host_input.len()];
        framed_input.extend(host_input);
        let mut framed = device.allocate(framed_input.len()).unwrap();
        framed.copy_from(&framed_input, &stream).unwrap();
        let fence = postprocessor
            .enqueue_view(
                framed.view(),
                framed_input.len(),
                host_input.len(),
                &[0],
                GpuRegressionOutputs {
                    skin: &mut skin,
                    tongue: &mut tongue,
                    jaw_transforms: &mut jaw,
                    eyes_rotations: &mut eyes,
                },
                &stream,
            )
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        skin.copy_to(&mut actual_skin, &stream).unwrap();
        tongue.copy_to(&mut actual_tongue, &stream).unwrap();
        assert_eq!(actual_skin, [2.0, 4.0, 6.0]);
        assert_eq!(actual_tongue, [2.0, 14.0, 26.0]);

        // The standalone animator uses the same jaw contract without owning a
        // Regression executor or any Skin/Tongue/Eyes state.
        let mut standalone = crate::animation::GpuMultiTrackTeethAnimator::new(
            &device,
            &stream,
            &jaw_neutral,
            JawParameters::default(),
            1,
        )
        .unwrap();
        let mut standalone_input = device.allocate(jaw_neutral.len()).unwrap();
        standalone_input
            .copy_from(&host_input[6..15], &stream)
            .unwrap();
        let mut standalone_output = device.allocate(16).unwrap();
        let standalone_input_info = standalone.input_batch_info();
        let standalone_output_info = standalone.output_batch_info();
        let fence = standalone
            .compute(
                crate::animation::GpuTeethInputBatch::new(&standalone_input, standalone_input_info),
                crate::animation::GpuTeethOutputBatch::new(
                    &mut standalone_output,
                    standalone_output_info,
                ),
                &stream,
            )
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        let mut standalone_jaw = [0.0; 16];
        standalone_output
            .copy_to(&mut standalone_jaw, &stream)
            .unwrap();
        for (standalone, pipeline) in standalone_jaw.into_iter().zip(actual_jaw) {
            assert!((standalone - pipeline).abs() < 1.0e-5);
        }
    }

    #[test]
    fn regression_pca_stays_on_device_before_animation() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let skin_neutral = [0.0, 0.0, 0.0];
        let tongue_neutral = [1.0, 2.0, 3.0];
        let jaw_neutral = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let layout = RegressionResultLayout {
            skin: 1,
            tongue: 1,
            jaw: 9,
            eyes: 4,
        };
        let mut pipeline = GpuRegressionPcaPostprocessor::new(
            &device,
            &stream,
            GpuRegressionModel {
                skin_neutral_pose: &skin_neutral,
                skin_lip_open_delta: &[0.0; 3],
                skin_eye_close_delta: &[0.0; 3],
                tongue_neutral_pose: &tongue_neutral,
                jaw_neutral_pose: &jaw_neutral,
                saccade_rotation: &[0.0, 0.0],
            },
            &[track_params()],
            1.0 / 30.0,
            &[1.0, 2.0, 3.0],
            &[0.5, 1.0, 1.5],
            layout,
        )
        .unwrap();
        let raw = [
            1.0, 1.0, // PCA coefficients
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // jaw
            1.0, 2.0, 3.0, 4.0, // eyes
        ];
        let mut input = device.allocate(raw.len()).unwrap();
        input.copy_from(&raw, &stream).unwrap();
        let mut skin = device.allocate(3).unwrap();
        let mut tongue = device.allocate(3).unwrap();
        let mut jaw = device.allocate(16).unwrap();
        let mut eyes = device.allocate(6).unwrap();
        let fence = pipeline
            .enqueue(
                input.view(),
                &[0],
                GpuRegressionOutputs {
                    skin: &mut skin,
                    tongue: &mut tongue,
                    jaw_transforms: &mut jaw,
                    eyes_rotations: &mut eyes,
                },
                &stream,
            )
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        let mut actual_skin = [0.0; 3];
        let mut actual_tongue = [0.0; 3];
        skin.copy_to(&mut actual_skin, &stream).unwrap();
        tongue.copy_to(&mut actual_tongue, &stream).unwrap();
        assert_eq!(actual_skin, [2.0, 4.0, 6.0]);
        assert_eq!(actual_tongue, [2.0, 14.0, 26.0]);
    }
}
