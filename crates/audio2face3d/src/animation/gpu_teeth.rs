use crate::animation::{JawParameters, TensorBatchInfo};
use crate::common::{Error, Result, checked_u32};
use crate::cuda::{
    CudaEvent, CudaModule, CudaStream, DeviceBuffer, DeviceId, DeviceView, GpuDevice,
    ensure_same_device, regression_jaw_ptx,
};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

const PARAMETER_STRIDE: usize = 4;
const TRANSFORM_SIZE: usize = 16;

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

/// Borrowed device input for standalone multi-track teeth animation.
///
/// Each batch element contains interleaved XYZ jaw deltas. `info.offset` is
/// relative to the beginning of each `info.stride`-element track row.
pub struct GpuTeethInputBatch<'a> {
    buffer: &'a DeviceBuffer<f32>,
    info: TensorBatchInfo,
}

impl<'a> GpuTeethInputBatch<'a> {
    pub const fn new(buffer: &'a DeviceBuffer<f32>, info: TensorBatchInfo) -> Self {
        Self { buffer, info }
    }

    pub const fn info(&self) -> TensorBatchInfo {
        self.info
    }

    pub fn tensor(&self) -> DeviceView<'_, f32> {
        self.buffer.view()
    }
}

/// Exclusively borrowed device output for standalone teeth transforms.
///
/// Every batch element stores one column-major 4x4 transform (16 floats).
pub struct GpuTeethOutputBatch<'a> {
    buffer: &'a mut DeviceBuffer<f32>,
    info: TensorBatchInfo,
}

impl<'a> GpuTeethOutputBatch<'a> {
    pub const fn new(buffer: &'a mut DeviceBuffer<f32>, info: TensorBatchInfo) -> Self {
        Self { buffer, info }
    }

    pub const fn info(&self) -> TensorBatchInfo {
        self.info
    }
}

/// Standalone CUDA implementation of the original `IMultiTrackAnimatorTeeth`.
///
/// The animator owns the neutral jaw and one parameter row per track, but uses
/// caller-owned input/output buffers and CUDA streams. It is stateless: reset
/// is a validated no-op and active-track masks are intentionally ignored, as
/// in the original SDK. Every call computes every configured track.
pub struct GpuMultiTrackTeethAnimator {
    module: CudaModule,
    neutral_pose: DeviceBuffer<f32>,
    device_parameters: DeviceBuffer<f32>,
    parameters: Vec<JawParameters>,
    jaw_pose_size: usize,
}

impl GpuMultiTrackTeethAnimator {
    /// Creates a standalone teeth animator and uploads immutable model data.
    ///
    /// `parameters` is copied to every track, matching the original `Init`
    /// contract. Uploads complete before this method returns.
    pub fn new(
        device: &Arc<GpuDevice>,
        stream: &CudaStream,
        neutral_pose: &[f32],
        parameters: JawParameters,
        track_count: usize,
    ) -> Result<Self> {
        if track_count == 0 {
            return Err(invalid("teeth animator requires at least one track"));
        }
        validate_pose(neutral_pose)?;
        parameters.validate()?;
        ensure_same_device(device.id(), stream.device_id())?;
        let parameter_len =
            track_count
                .checked_mul(PARAMETER_STRIDE)
                .ok_or(Error::IntegerOverflow {
                    field: "teeth_parameter_count",
                    value: track_count,
                    target: "usize",
                })?;
        let mut neutral = device.allocate(neutral_pose.len())?;
        neutral.copy_from(neutral_pose, stream)?;
        let parameters_host = vec![parameters; track_count];
        let packed = pack_parameters(&parameters_host, parameter_len);
        let mut device_parameters = device.allocate(parameter_len)?;
        device_parameters.copy_from(&packed, stream)?;
        Ok(Self {
            module: device.load_module(regression_jaw_ptx())?,
            neutral_pose: neutral,
            device_parameters,
            parameters: parameters_host,
            jaw_pose_size: neutral_pose.len(),
        })
    }

    pub const fn track_count(&self) -> usize {
        self.parameters.len()
    }

    pub const fn jaw_pose_size(&self) -> usize {
        self.jaw_pose_size
    }

    pub fn device_id(&self) -> DeviceId {
        self.neutral_pose.device_id()
    }

    /// Contiguous input layout for one jaw-delta row per track.
    pub const fn input_batch_info(&self) -> TensorBatchInfo {
        TensorBatchInfo {
            offset: 0,
            size: self.jaw_pose_size,
            stride: self.jaw_pose_size,
        }
    }

    /// Contiguous output layout for one column-major transform per track.
    pub const fn output_batch_info(&self) -> TensorBatchInfo {
        TensorBatchInfo {
            offset: 0,
            size: TRANSFORM_SIZE,
            stride: TRANSFORM_SIZE,
        }
    }

    pub fn parameters(&self, track: usize) -> Result<JawParameters> {
        self.parameters
            .get(track)
            .copied()
            .ok_or_else(|| invalid(format!("teeth track {track} is out of range")))
    }

    /// Replaces one track's parameters and completes the device upload.
    pub fn set_parameters(
        &mut self,
        track: usize,
        parameters: JawParameters,
        stream: &CudaStream,
    ) -> Result<()> {
        parameters.validate()?;
        ensure_same_device(self.device_id(), stream.device_id())?;
        let previous = *self
            .parameters
            .get(track)
            .ok_or_else(|| invalid(format!("teeth track {track} is out of range")))?;
        self.parameters[track] = parameters;
        let packed = pack_parameters(&self.parameters, self.device_parameters.len());
        if let Err(error) = self.device_parameters.copy_from(&packed, stream) {
            self.parameters[track] = previous;
            return Err(error);
        }
        Ok(())
    }

    /// Validates a track index. Teeth animation has no temporal state to reset.
    pub fn reset(&mut self, track: usize) -> Result<()> {
        self.parameters(track).map(|_| ())
    }

    /// Accepts the executor-style active mask for API composition.
    ///
    /// The mask is deliberately ignored. The original teeth animator is
    /// stateless and computes every track even when other animator layers mark
    /// a track inactive.
    pub fn set_active_tracks(&mut self, _active_tracks: Option<&[u64]>) {}

    /// Enqueues every configured track on `stream` without host synchronization.
    ///
    /// Both tensors must have exactly `stride * track_count` elements. Within
    /// each row, `offset + size` must not exceed `stride`. Input size must equal
    /// the neutral jaw size and output size must be 16.
    pub fn compute<'a>(
        &'a mut self,
        input: GpuTeethInputBatch<'a>,
        output: GpuTeethOutputBatch<'a>,
        stream: &'a CudaStream,
    ) -> Result<GpuMultiTrackTeethFence<'a>> {
        ensure_same_device(self.device_id(), stream.device_id())?;
        ensure_same_device(self.device_id(), input.buffer.device_id())?;
        ensure_same_device(self.device_id(), output.buffer.device_id())?;
        validate_batch(
            "teeth input",
            input.buffer.len(),
            input.info,
            self.jaw_pose_size,
            self.track_count(),
        )?;
        validate_batch(
            "teeth output",
            output.buffer.len(),
            output.info,
            TRANSFORM_SIZE,
            self.track_count(),
        )?;

        let point_count = checked_u32(self.jaw_pose_size / 3, "teeth_point_count")?;
        let function = self.module.function("audio2face_regression_jaw")?;
        for track in 0..self.track_count() {
            let input_element = track
                .checked_mul(input.info.stride)
                .and_then(|base| base.checked_add(input.info.offset))
                .ok_or_else(|| invalid("teeth input offset overflow"))?;
            let output_element = track
                .checked_mul(output.info.stride)
                .and_then(|base| base.checked_add(output.info.offset))
                .ok_or_else(|| invalid("teeth output offset overflow"))?;
            let parameter_element = track
                .checked_mul(PARAMETER_STRIDE)
                .ok_or_else(|| invalid("teeth parameter offset overflow"))?;
            let mut arguments = [
                self.neutral_pose.view().as_raw(),
                input.buffer.view().as_raw() + byte_offset(input_element)?,
                u64::from(point_count),
                1,
                self.device_parameters.view().as_raw() + byte_offset(parameter_element)?,
                PARAMETER_STRIDE as u64,
                0,
                output.buffer.view().as_raw() + byte_offset(output_element)?,
            ];
            let mut kernel_parameters: Vec<*mut c_void> = arguments
                .iter_mut()
                .map(|argument| (argument as *mut u64).cast())
                .collect();
            // SAFETY: layouts and devices were validated above. Each launch
            // addresses disjoint track rows, and the returned fence retains all
            // buffers, animator storage, and the stream until completion.
            unsafe {
                function.launch_raw((1, 1, 1), (32, 1, 1), 0, stream, &mut kernel_parameters)?;
            }
        }
        let event = stream.create_event()?;
        event.record(stream)?;
        Ok(GpuMultiTrackTeethFence {
            event,
            output: output.buffer.view(),
            output_info: output.info,
            track_count: self.track_count(),
            stream,
            _resources: PhantomData,
        })
    }
}

/// Completion token retaining all resources borrowed by a teeth animation.
pub struct GpuMultiTrackTeethFence<'a> {
    event: CudaEvent,
    output: DeviceView<'a, f32>,
    output_info: TensorBatchInfo,
    track_count: usize,
    stream: &'a CudaStream,
    _resources: PhantomData<(
        &'a mut GpuMultiTrackTeethAnimator,
        &'a DeviceBuffer<f32>,
        &'a mut DeviceBuffer<f32>,
    )>,
}

impl GpuMultiTrackTeethFence<'_> {
    pub fn stream(&self) -> &CudaStream {
        self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }

    /// Returns one track's column-major 4x4 transform on the device.
    pub fn transform(&self, track: usize) -> Result<DeviceView<'_, f32>> {
        if track >= self.track_count {
            return Err(invalid(format!("teeth track {track} is out of range")));
        }
        let offset = track
            .checked_mul(self.output_info.stride)
            .and_then(|base| base.checked_add(self.output_info.offset))
            .ok_or_else(|| invalid("teeth output offset overflow"))?;
        self.output.slice(offset, TRANSFORM_SIZE)
    }
}

impl Drop for GpuMultiTrackTeethFence<'_> {
    fn drop(&mut self) {
        // A dropped fence must not release the exclusive buffer/animator borrows
        // while CUDA still references them.
        let _ = self.event.synchronize();
    }
}

fn validate_pose(pose: &[f32]) -> Result<()> {
    if pose.is_empty() || !pose.len().is_multiple_of(3) {
        return Err(invalid("neutral jaw size must be a non-zero multiple of 3"));
    }
    if pose.iter().any(|value| !value.is_finite()) {
        return Err(invalid("neutral jaw must contain only finite values"));
    }
    Ok(())
}

fn validate_batch(
    name: &str,
    buffer_len: usize,
    info: TensorBatchInfo,
    expected_size: usize,
    track_count: usize,
) -> Result<()> {
    if info.size != expected_size {
        return Err(invalid(format!(
            "{name} component size is {}, expected {expected_size}",
            info.size
        )));
    }
    let row_end = info
        .offset
        .checked_add(info.size)
        .ok_or_else(|| invalid(format!("{name} row range overflow")))?;
    if info.stride == 0 || row_end > info.stride {
        return Err(invalid(format!("{name} offset and size exceed its stride")));
    }
    let expected_len = info
        .stride
        .checked_mul(track_count)
        .ok_or_else(|| invalid(format!("{name} batch size overflow")))?;
    if buffer_len != expected_len {
        return Err(invalid(format!(
            "{name} buffer has {buffer_len} elements, expected {expected_len}"
        )));
    }
    Ok(())
}

fn byte_offset(element: usize) -> Result<u64> {
    let bytes = element
        .checked_mul(size_of::<f32>())
        .ok_or(Error::IntegerOverflow {
            field: "teeth_buffer_byte_offset",
            value: element,
            target: "u64",
        })?;
    u64::try_from(bytes).map_err(|_| Error::IntegerOverflow {
        field: "teeth_buffer_byte_offset",
        value: bytes,
        target: "u64",
    })
}

fn pack_parameters(parameters: &[JawParameters], capacity: usize) -> Vec<f32> {
    let mut packed = Vec::with_capacity(capacity);
    for parameter in parameters {
        // The fourth slot is reserved to keep a stable aligned device row.
        packed.extend([
            parameter.strength,
            parameter.height_offset,
            parameter.depth_offset,
            0.0,
        ]);
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{GeometryModelData, JawTransform};
    use crate::{Model, ModelKind, ModelParameters};

    fn assert_close(actual: &[f32], expected: &[f32]) {
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "element {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn batch_contract_rejects_invalid_size_stride_length_and_overflow() {
        assert!(
            validate_batch(
                "input",
                18,
                TensorBatchInfo {
                    offset: 0,
                    size: 8,
                    stride: 9,
                },
                9,
                2,
            )
            .is_err()
        );
        assert!(
            validate_batch(
                "input",
                18,
                TensorBatchInfo {
                    offset: 2,
                    size: 9,
                    stride: 10,
                },
                9,
                2,
            )
            .is_err()
        );
        assert!(
            validate_batch(
                "input",
                17,
                TensorBatchInfo {
                    offset: 0,
                    size: 9,
                    stride: 9,
                },
                9,
                2,
            )
            .is_err()
        );
        assert!(
            validate_batch(
                "input",
                1,
                TensorBatchInfo {
                    offset: 0,
                    size: 9,
                    stride: usize::MAX,
                },
                9,
                2,
            )
            .is_err()
        );
    }

    #[test]
    fn standalone_multitrack_matches_host_with_strided_batches() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let neutral = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert!(
            GpuMultiTrackTeethAnimator::new(
                &device,
                &stream,
                &neutral,
                JawParameters::default(),
                0,
            )
            .is_err()
        );
        assert!(
            GpuMultiTrackTeethAnimator::new(&device, &stream, &[], JawParameters::default(), 1,)
                .is_err()
        );
        let host = JawTransform::new(neutral.to_vec()).unwrap();
        let mut animator = GpuMultiTrackTeethAnimator::new(
            &device,
            &stream,
            &neutral,
            JawParameters::default(),
            2,
        )
        .unwrap();
        let second_parameters = JawParameters {
            strength: 0.5,
            height_offset: 2.0,
            depth_offset: -1.0,
        };
        animator
            .set_parameters(1, second_parameters, &stream)
            .unwrap();
        animator.set_active_tracks(Some(&[1]));
        animator.reset(0).unwrap();

        let input_info = TensorBatchInfo {
            offset: 2,
            size: 9,
            stride: 12,
        };
        let first = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let second = [2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0];
        let mut input_host = vec![-9.0; input_info.stride * 2];
        input_host[2..11].copy_from_slice(&first);
        input_host[14..23].copy_from_slice(&second);
        let mut input = device.allocate(input_host.len()).unwrap();
        input.copy_from(&input_host, &stream).unwrap();

        let output_info = TensorBatchInfo {
            offset: 3,
            size: TRANSFORM_SIZE,
            stride: 20,
        };
        let mut output = device.allocate(output_info.stride * 2).unwrap();
        output
            .copy_from(&vec![-7.0; output_info.stride * 2], &stream)
            .unwrap();
        let fence = animator
            .compute(
                GpuTeethInputBatch::new(&input, input_info),
                GpuTeethOutputBatch::new(&mut output, output_info),
                &stream,
            )
            .unwrap();
        assert_eq!(fence.stream().device_id(), device.id());
        assert_eq!(fence.transform(0).unwrap().len(), TRANSFORM_SIZE);
        fence.synchronize().unwrap();
        drop(fence);

        let mut actual = vec![0.0; output_info.stride * 2];
        output.copy_to(&mut actual, &stream).unwrap();
        assert_close(
            &actual[3..19],
            &host.compute(&first, JawParameters::default()).unwrap(),
        );
        assert_close(
            &actual[23..39],
            &host.compute(&second, second_parameters).unwrap(),
        );
        assert_eq!(actual[0], -7.0);
        assert_eq!(actual[20], -7.0);
        assert_eq!(animator.parameters(1).unwrap(), second_parameters);
        assert!(animator.reset(2).is_err());
    }

    #[test]
    fn original_maximum_test_boundary_of_128_tracks_is_supported() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let neutral = [0.0, 0.0, 0.0];
        let mut animator = GpuMultiTrackTeethAnimator::new(
            &device,
            &stream,
            &neutral,
            JawParameters::default(),
            128,
        )
        .unwrap();
        let input_info = animator.input_batch_info();
        let output_info = animator.output_batch_info();
        let mut input = device.allocate(input_info.stride * 128).unwrap();
        input
            .copy_from(&vec![0.0; input_info.stride * 128], &stream)
            .unwrap();
        let mut output = device.allocate(output_info.stride * 128).unwrap();
        let fence = animator
            .compute(
                GpuTeethInputBatch::new(&input, input_info),
                GpuTeethOutputBatch::new(&mut output, output_info),
                &stream,
            )
            .unwrap();
        fence.synchronize().unwrap();
        drop(fence);
        let mut result = vec![0.0; output_info.stride * 128];
        output.copy_to(&mut result, &stream).unwrap();
        for transform in result.chunks_exact(TRANSFORM_SIZE) {
            assert_close(
                transform,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
            );
        }
    }

    #[test]
    fn runs_installed_geometry_model_data_when_configured() {
        let Some(paths) = std::env::var_os("AUDIO2FACE3D_TEST_MODEL_DIRS") else {
            return;
        };
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        for root in std::env::split_paths(&paths) {
            let model = Model::load(root.join("model.json")).unwrap();
            let data = match model.kind() {
                ModelKind::Regression => {
                    GeometryModelData::load_regression(model.model_data_path(0).unwrap()).unwrap()
                }
                ModelKind::Diffusion => {
                    GeometryModelData::load_diffusion(model.model_data_path(0).unwrap()).unwrap()
                }
                ModelKind::Emotion => continue,
            };
            let geometry = match model.parameters(0).unwrap() {
                ModelParameters::Geometry(value) => value,
                ModelParameters::Emotion(_) => unreachable!(),
            };
            let parameters = JawParameters {
                strength: geometry.lower_teeth_strength,
                height_offset: geometry.lower_teeth_height_offset,
                depth_offset: geometry.lower_teeth_depth_offset,
            };
            let host = JawTransform::new(data.jaw_neutral_pose.clone()).unwrap();
            let mut animator = GpuMultiTrackTeethAnimator::new(
                &device,
                &stream,
                &data.jaw_neutral_pose,
                parameters,
                1,
            )
            .unwrap();
            let input_info = animator.input_batch_info();
            let output_info = animator.output_batch_info();
            let deltas = vec![0.0; data.jaw_neutral_pose.len()];
            let mut input = device.allocate(deltas.len()).unwrap();
            input.copy_from(&deltas, &stream).unwrap();
            let mut output = device.allocate(TRANSFORM_SIZE).unwrap();
            let fence = animator
                .compute(
                    GpuTeethInputBatch::new(&input, input_info),
                    GpuTeethOutputBatch::new(&mut output, output_info),
                    &stream,
                )
                .unwrap();
            fence.synchronize().unwrap();
            drop(fence);
            let mut actual = [0.0; TRANSFORM_SIZE];
            output.copy_to(&mut actual, &stream).unwrap();
            assert_close(&actual, &host.compute(&deltas, parameters).unwrap());
        }
    }
}
