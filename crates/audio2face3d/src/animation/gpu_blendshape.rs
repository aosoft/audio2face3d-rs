//! Device-resident blendshape solver using the SDK's four-step ADMM path.

use crate::animation::{BlendshapeData, CpuBlendshapeSolver};
use crate::common::{BlendshapeConfig, Error, Result};
use crate::cuda::{
    CublasHandle, CublasTranspose, CudaEvent, CudaModule, CudaStream, DeviceBuffer, DeviceView,
    GpuDevice, blendshape_solver_ptx, ensure_same_device,
};
use std::ffi::c_void;
use std::marker::PhantomData;
use std::rc::Rc;

const BLOCK_SIZE: u32 = 256;

macro_rules! params {
    ($($value:ident),+ $(,)?) => {
        [$(
            std::ptr::from_mut(&mut $value).cast::<c_void>()
        ),+]
    };
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

pub struct GpuBlendshapeSolver {
    data: BlendshapeData,
    module: CudaModule,
    blas: CublasHandle,
    coordinate_count: usize,
    active_count: usize,
    pose_count: usize,
    temporal_beta: f32,
    coordinate_indices: DeviceBuffer<u32>,
    active_indices: DeviceBuffer<u32>,
    cancel_first: Option<DeviceBuffer<u32>>,
    cancel_second: Option<DeviceBuffer<u32>>,
    neutral: DeviceBuffer<f32>,
    deltas: DeviceBuffer<f32>,
    matrix: DeviceBuffer<f32>,
    matrix_inverse: DeviceBuffer<f32>,
    admm_weights: DeviceBuffer<f32>,
    admm_inverse: DeviceBuffer<f32>,
    multipliers: DeviceBuffer<f32>,
    offsets: DeviceBuffer<f32>,
    target_delta: DeviceBuffer<f32>,
    rhs: DeviceBuffer<f32>,
    lower: DeviceBuffer<f32>,
    upper: DeviceBuffer<f32>,
    solved: DeviceBuffer<f32>,
    atb: DeviceBuffer<f32>,
    u1: DeviceBuffer<f32>,
    z2: DeviceBuffer<f32>,
    u2: DeviceBuffer<f32>,
    previous: DeviceBuffer<f32>,
}

impl GpuBlendshapeSolver {
    pub fn new(
        device: &Rc<GpuDevice>,
        stream: &CudaStream,
        data: BlendshapeData,
        config: &BlendshapeConfig,
    ) -> Result<Self> {
        ensure_same_device(device.id(), stream.device_id())?;
        let mut cpu = CpuBlendshapeSolver::from_config(data, config)?;
        cpu.prepare()?;
        let prepared = cpu
            .prepared
            .take()
            .ok_or_else(|| invalid("blendshape GPU preparation failed"))?;
        let coordinate_count = prepared.coordinate_indices.len();
        let active_count = prepared.active_indices.len();
        let pose_count = cpu.data.pose_count();

        let matrix = prepared
            .matrix
            .iter()
            .map(|value| *value as f32)
            .collect::<Vec<_>>();
        let matrix_inverse = invert_matrix(&prepared.matrix, active_count)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let mut admm_system = multiply_transpose_self(&prepared.matrix, active_count);
        let admm_weights = (0..active_count)
            .map(|index| 0.25 * admm_system[index * active_count + index].sqrt())
            .collect::<Vec<_>>();
        for index in 0..active_count {
            admm_system[index * active_count + index] += admm_weights[index].powi(2);
        }
        let admm_inverse = invert_matrix(&admm_system, active_count)?
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let admm_weights = admm_weights
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let coordinate_indices = prepared
            .coordinate_indices
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| invalid("blendshape coordinate index overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let active_indices = prepared
            .active_indices
            .iter()
            .map(|value| {
                u32::try_from(*value).map_err(|_| invalid("blendshape active index overflow"))
            })
            .collect::<Result<Vec<_>>>()?;
        let cancel_first = prepared
            .cancel_pairs
            .iter()
            .map(|pair| u32::try_from(pair.0).map_err(|_| invalid("cancel pair index overflow")))
            .collect::<Result<Vec<_>>>()?;
        let cancel_second = prepared
            .cancel_pairs
            .iter()
            .map(|pair| u32::try_from(pair.1).map_err(|_| invalid("cancel pair index overflow")))
            .collect::<Result<Vec<_>>>()?;

        let upload = |values: &[f32]| -> Result<DeviceBuffer<f32>> {
            let mut buffer = device.allocate(values.len())?;
            buffer.copy_from(values, stream)?;
            Ok(buffer)
        };
        let upload_u32 = |values: &[u32]| -> Result<DeviceBuffer<u32>> {
            let mut buffer = device.allocate(values.len())?;
            buffer.copy_from(values, stream)?;
            Ok(buffer)
        };
        let cancel_first = (!cancel_first.is_empty())
            .then(|| upload_u32(&cancel_first))
            .transpose()?;
        let cancel_second = (!cancel_second.is_empty())
            .then(|| upload_u32(&cancel_second))
            .transpose()?;
        let mut lower = device.allocate(active_count)?;
        lower.memset_zero(stream)?;
        let mut previous = device.allocate(active_count)?;
        previous.memset_zero(stream)?;

        Ok(Self {
            data: cpu.data,
            module: device.load_module(blendshape_solver_ptx())?,
            blas: CublasHandle::new(stream)?,
            coordinate_count,
            active_count,
            pose_count,
            temporal_beta: cpu.parameters.temporal_regularization * prepared.scale_factor as f32,
            coordinate_indices: upload_u32(&coordinate_indices)?,
            active_indices: upload_u32(&active_indices)?,
            cancel_first,
            cancel_second,
            neutral: upload(&prepared.active_neutral)?,
            deltas: upload(&prepared.active_deltas)?,
            matrix: upload(&matrix)?,
            matrix_inverse: upload(&matrix_inverse)?,
            admm_weights: upload(&admm_weights)?,
            admm_inverse: upload(&admm_inverse)?,
            multipliers: upload(&cpu.multipliers)?,
            offsets: upload(&cpu.offsets)?,
            target_delta: device.allocate(coordinate_count)?,
            rhs: device.allocate(active_count)?,
            lower,
            upper: device.allocate(active_count)?,
            solved: device.allocate(active_count)?,
            atb: device.allocate(active_count)?,
            u1: device.allocate(active_count)?,
            z2: device.allocate(active_count)?,
            u2: device.allocate(active_count)?,
            previous,
        })
    }

    pub fn evaluate_pose(&self, weights: &[f32]) -> Result<Vec<f32>> {
        self.data.evaluate_pose(weights)
    }

    pub fn target_len(&self) -> usize {
        self.data.neutral_pose.len()
    }

    pub fn pose_count(&self) -> usize {
        self.pose_count
    }

    pub fn device_id(&self) -> crate::cuda::DeviceId {
        self.neutral.device_id()
    }

    pub fn set_multipliers(&mut self, values: &[f32], stream: &CudaStream) -> Result<()> {
        validate_post_parameters(values, self.pose_count, "multipliers")?;
        self.multipliers.copy_from(values, stream)
    }

    pub fn set_offsets(&mut self, values: &[f32], stream: &CudaStream) -> Result<()> {
        validate_post_parameters(values, self.pose_count, "offsets")?;
        self.offsets.copy_from(values, stream)
    }

    pub fn reset(&mut self, stream: &CudaStream) -> Result<()> {
        self.previous.memset_zero(stream)
    }

    /// Enqueues the complete device solver without a host round trip.
    ///
    /// All work uses `stream`. The returned fence borrows the solver, target,
    /// output, and stream until [`GpuBlendshapeSolveFence::synchronize`] has
    /// observed completion. Device work that consumes [`GpuBlendshapeSolveFence::output`]
    /// may be enqueued on [`GpuBlendshapeSolveFence::stream`] before host
    /// synchronization; direct host access must wait for completion.
    pub fn solve_async<'a>(
        &'a mut self,
        target: &'a DeviceBuffer<f32>,
        output: &'a mut DeviceBuffer<f32>,
        stream: &'a CudaStream,
    ) -> Result<GpuBlendshapeSolveFence<'a>> {
        ensure_same_device(stream.device_id(), target.device_id())?;
        ensure_same_device(stream.device_id(), output.device_id())?;
        if target.len() != self.data.neutral_pose.len() || output.len() != self.pose_count {
            return Err(invalid(
                "blendshape GPU input/output dimensions do not match",
            ));
        }
        let mut coordinate_count = checked_u32(self.coordinate_count, "coordinate count")?;
        let mut active_count = checked_u32(self.active_count, "active count")?;
        let mut pose_count = checked_u32(self.pose_count, "pose count")?;

        let mut target_delta = self.target_delta.view().as_raw();
        let mut target_pointer = target.view().as_raw();
        let mut neutral = self.neutral.view().as_raw();
        let mut coordinate_indices = self.coordinate_indices.view().as_raw();
        let mut gather_params = params![
            target_delta,
            target_pointer,
            neutral,
            coordinate_indices,
            coordinate_count
        ];
        // SAFETY: kernel arguments match the PTX ABI and every referenced
        // allocation is retained by the returned fence.
        unsafe {
            launch(
                &self.module,
                "blendshape_gather_subtract",
                coordinate_count,
                stream,
                &mut gather_params,
            )?;
        }

        launch_copy(
            &self.module,
            &mut self.rhs,
            &self.previous,
            active_count,
            stream,
        )?;
        // SAFETY: the solver fence retains all cuBLAS resources and buffers;
        // dimensions exactly describe the uploaded column-major delta matrix.
        unsafe {
            self.blas.enqueue_matrix_vector(
                self.deltas.view(),
                self.target_delta.view(),
                &mut self.rhs,
                self.coordinate_count,
                self.active_count,
                CublasTranspose::Transpose,
                1.0,
                self.temporal_beta,
                stream,
            )?;
        }
        launch_fill(&self.module, &mut self.upper, 1.0, active_count, stream)?;
        self.enqueue_admm(active_count, stream)?;
        if let (Some(first), Some(second)) = (&self.cancel_first, &self.cancel_second) {
            let mut pair_count = checked_u32(first.len(), "cancel pair count")?;
            let mut upper = self.upper.view().as_raw();
            let mut solved = self.solved.view().as_raw();
            let mut first = first.view().as_raw();
            let mut second = second.view().as_raw();
            let mut cancel_params = params![upper, solved, first, second, pair_count];
            // SAFETY: validated pair indices address active weight/upper buffers.
            unsafe {
                launch(
                    &self.module,
                    "blendshape_cancel_upper",
                    pair_count,
                    stream,
                    &mut cancel_params,
                )?;
            }
            self.enqueue_admm(active_count, stream)?;
        }
        launch_copy(
            &self.module,
            &mut self.previous,
            &self.solved,
            active_count,
            stream,
        )?;
        // SAFETY: output owns pose_count f32 values and remains borrowed by the fence.
        unsafe {
            stream.memset_device_zero(
                output.view().as_raw(),
                self.pose_count * std::mem::size_of::<f32>(),
            )?;
        }
        let mut output_pointer = output.view().as_raw();
        let mut solved = self.solved.view().as_raw();
        let mut active_indices = self.active_indices.view().as_raw();
        let mut unmap_params = params![output_pointer, solved, active_indices, active_count];
        // SAFETY: active indices were validated against the full output at construction.
        unsafe {
            launch(
                &self.module,
                "blendshape_unmap",
                active_count,
                stream,
                &mut unmap_params,
            )?;
        }
        let mut output_pointer = output.view().as_raw();
        let mut multipliers = self.multipliers.view().as_raw();
        let mut offsets = self.offsets.view().as_raw();
        let mut apply_params = params![output_pointer, multipliers, offsets, pose_count];
        // SAFETY: all arrays contain pose_count elements and the fence retains them.
        unsafe {
            launch(
                &self.module,
                "blendshape_apply",
                pose_count,
                stream,
                &mut apply_params,
            )?;
        }
        let event = stream.create_event()?;
        event.record(stream)?;
        Ok(GpuBlendshapeSolveFence {
            event,
            output: output.view(),
            stream,
            _resources: PhantomData,
        })
    }

    fn enqueue_admm(&mut self, count: u32, stream: &CudaStream) -> Result<()> {
        // SAFETY: persistent buffers all have active_count elements and remain
        // owned by self until the outer solve fence completes.
        unsafe {
            self.blas.enqueue_matrix_vector(
                self.matrix.view(),
                self.rhs.view(),
                &mut self.atb,
                self.active_count,
                self.active_count,
                CublasTranspose::Transpose,
                1.0,
                0.0,
                stream,
            )?;
            self.blas.enqueue_matrix_vector(
                self.matrix_inverse.view(),
                self.rhs.view(),
                &mut self.solved,
                self.active_count,
                self.active_count,
                CublasTranspose::None,
                1.0,
                0.0,
                stream,
            )?;
        }
        launch_fill(&self.module, &mut self.u1, 0.0, count, stream)?;
        launch_clip(
            &self.module,
            &mut self.solved,
            &self.lower,
            &self.upper,
            count,
            stream,
        )?;
        launch_admm(
            &self.module,
            &self.solved,
            &self.u1,
            &mut self.z2,
            &mut self.u2,
            &self.admm_weights,
            &self.atb,
            &self.admm_inverse,
            &self.lower,
            &self.upper,
            count,
            stream,
        )?;
        launch_admm(
            &self.module,
            &self.z2,
            &self.u2,
            &mut self.solved,
            &mut self.u1,
            &self.admm_weights,
            &self.atb,
            &self.admm_inverse,
            &self.lower,
            &self.upper,
            count,
            stream,
        )?;
        launch_admm(
            &self.module,
            &self.solved,
            &self.u1,
            &mut self.z2,
            &mut self.u2,
            &self.admm_weights,
            &self.atb,
            &self.admm_inverse,
            &self.lower,
            &self.upper,
            count,
            stream,
        )?;
        launch_admm(
            &self.module,
            &self.z2,
            &self.u2,
            &mut self.solved,
            &mut self.u1,
            &self.admm_weights,
            &self.atb,
            &self.admm_inverse,
            &self.lower,
            &self.upper,
            count,
            stream,
        )
    }
}

pub struct GpuBlendshapeSolveFence<'a> {
    event: CudaEvent,
    output: DeviceView<'a, f32>,
    stream: &'a CudaStream,
    _resources: PhantomData<(
        &'a mut GpuBlendshapeSolver,
        &'a DeviceBuffer<f32>,
        &'a mut DeviceBuffer<f32>,
        &'a CudaStream,
    )>,
}

impl<'a> GpuBlendshapeSolveFence<'a> {
    /// Device-resident output ordered after the solve on [`Self::stream`].
    pub fn output(&self) -> DeviceView<'_, f32> {
        self.output
    }

    /// CUDA stream on which the solve and completion event were recorded.
    pub fn stream(&self) -> &CudaStream {
        self.stream
    }

    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }
}

impl Drop for GpuBlendshapeSolveFence<'_> {
    fn drop(&mut self) {
        // Keep the borrowed solver and buffers alive until their enqueued work
        // finishes even when the caller does not synchronize explicitly.
        let _ = self.event.synchronize();
    }
}

fn launch_fill(
    module: &CudaModule,
    output: &mut DeviceBuffer<f32>,
    value: f32,
    count: u32,
    stream: &CudaStream,
) -> Result<()> {
    let mut output = output.view().as_raw();
    let mut value = value;
    let mut count = count;
    let mut kernel_params = params![output, value, count];
    // SAFETY: output has count elements and the caller retains it through stream completion.
    unsafe { launch(module, "blendshape_fill", count, stream, &mut kernel_params) }
}

fn launch_copy(
    module: &CudaModule,
    output: &mut DeviceBuffer<f32>,
    input: &DeviceBuffer<f32>,
    count: u32,
    stream: &CudaStream,
) -> Result<()> {
    let mut output = output.view().as_raw();
    let mut input = input.view().as_raw();
    let mut count = count;
    let mut kernel_params = params![output, input, count];
    // SAFETY: both allocations have count elements and remain alive through completion.
    unsafe { launch(module, "blendshape_copy", count, stream, &mut kernel_params) }
}

fn launch_clip(
    module: &CudaModule,
    values: &mut DeviceBuffer<f32>,
    lower: &DeviceBuffer<f32>,
    upper: &DeviceBuffer<f32>,
    count: u32,
    stream: &CudaStream,
) -> Result<()> {
    let mut values = values.view().as_raw();
    let mut lower = lower.view().as_raw();
    let mut upper = upper.view().as_raw();
    let mut count = count;
    let mut kernel_params = params![values, lower, upper, count];
    // SAFETY: each allocation has count elements and remains alive through completion.
    unsafe { launch(module, "blendshape_clip", count, stream, &mut kernel_params) }
}

#[allow(clippy::too_many_arguments)]
fn launch_admm(
    module: &CudaModule,
    z: &DeviceBuffer<f32>,
    u: &DeviceBuffer<f32>,
    z_out: &mut DeviceBuffer<f32>,
    u_out: &mut DeviceBuffer<f32>,
    weights: &DeviceBuffer<f32>,
    atb: &DeviceBuffer<f32>,
    inverse: &DeviceBuffer<f32>,
    lower: &DeviceBuffer<f32>,
    upper: &DeviceBuffer<f32>,
    count: u32,
    stream: &CudaStream,
) -> Result<()> {
    let mut u_out = u_out.view().as_raw();
    let mut z_out = z_out.view().as_raw();
    let mut u = u.view().as_raw();
    let mut z = z.view().as_raw();
    let mut weights = weights.view().as_raw();
    let mut atb = atb.view().as_raw();
    let mut inverse = inverse.view().as_raw();
    let mut lower = lower.view().as_raw();
    let mut upper = upper.view().as_raw();
    let mut count = count;
    let mut kernel_params = params![
        u_out, z_out, u, z, weights, atb, inverse, lower, upper, count
    ];
    // SAFETY: argument order matches the kernel and all dimensions equal count.
    unsafe {
        launch(
            module,
            "blendshape_admm_update",
            count,
            stream,
            &mut kernel_params,
        )
    }
}

unsafe fn launch(
    module: &CudaModule,
    name: &str,
    count: u32,
    stream: &CudaStream,
    parameters: &mut [*mut c_void],
) -> Result<()> {
    let function = module.function(name)?;
    // SAFETY: the caller guarantees the named kernel ABI, argument storage,
    // allocation ownership, and asynchronous lifetime.
    unsafe {
        function.launch_raw(
            (count.div_ceil(BLOCK_SIZE), 1, 1),
            (BLOCK_SIZE, 1, 1),
            0,
            stream,
            parameters,
        )
    }
}

fn invert_matrix(matrix: &[f64], count: usize) -> Result<Vec<f64>> {
    let width = count * 2;
    let mut augmented = vec![0.0_f64; count * width];
    for row in 0..count {
        for column in 0..count {
            augmented[row * width + column] = matrix[row * count + column];
        }
        augmented[row * width + count + row] = 1.0;
    }
    for column in 0..count {
        let pivot = (column..count)
            .max_by(|left, right| {
                augmented[*left * width + column]
                    .abs()
                    .total_cmp(&augmented[*right * width + column].abs())
            })
            .ok_or_else(|| invalid("blendshape matrix is empty"))?;
        if augmented[pivot * width + column].abs() <= f64::EPSILON {
            return Err(invalid("blendshape GPU matrix is singular"));
        }
        if pivot != column {
            for index in 0..width {
                augmented.swap(column * width + index, pivot * width + index);
            }
        }
        let divisor = augmented[column * width + column];
        for index in 0..width {
            augmented[column * width + index] /= divisor;
        }
        for row in 0..count {
            if row == column {
                continue;
            }
            let factor = augmented[row * width + column];
            for index in 0..width {
                augmented[row * width + index] -= factor * augmented[column * width + index];
            }
        }
    }
    Ok((0..count)
        .flat_map(|row| {
            let augmented = &augmented;
            (0..count).map(move |column| augmented[row * width + count + column])
        })
        .collect())
}

fn multiply_transpose_self(matrix: &[f64], count: usize) -> Vec<f64> {
    let mut result = vec![0.0; count * count];
    for row in 0..count {
        for column in 0..count {
            result[row * count + column] = (0..count)
                .map(|inner| matrix[inner * count + row] * matrix[inner * count + column])
                .sum();
        }
    }
    result
}

fn validate_post_parameters(values: &[f32], count: usize, name: &str) -> Result<()> {
    if values.len() != count || values.iter().any(|value| !value.is_finite()) {
        Err(invalid(format!("blendshape GPU {name} are invalid")))
    } else {
        Ok(())
    }
}

fn checked_u32(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("blendshape {name} exceeds u32")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BlendshapeConfig {
        BlendshapeConfig {
            l2_regularization: 0.01,
            temporal_regularization: 0.0,
            l1_regularization: 0.0,
            symmetry_regularization: 0.0,
            num_poses: 3,
            active_poses: vec![1, 1, 1],
            cancel_poses: vec![-1, -1, -1],
            symmetry_poses: vec![-1, -1, -1],
            multipliers: vec![1.0; 3],
            offsets: vec![0.0; 3],
            template_bb_size: 54.7,
            tolerance: 1.0e-10,
        }
    }

    fn data() -> BlendshapeData {
        BlendshapeData {
            neutral_pose: vec![0.0, 0.0, 0.0, 2.0, 3.0, 4.0],
            delta_poses: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0,
            ],
            pose_names: vec!["x".into(), "y".into(), "z".into()],
            pose_mask: None,
        }
    }

    #[test]
    fn gpu_admm_produces_bounded_device_output_and_reset_replays() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let data = data();
        let target = data.evaluate_pose(&[0.25, 0.5, 0.75]).unwrap();
        let mut solver = GpuBlendshapeSolver::new(&device, &stream, data, &config()).unwrap();
        let mut target_device = device.allocate(target.len()).unwrap();
        target_device.copy_from(&target, &stream).unwrap();
        let mut output_device = device.allocate(3).unwrap();
        solver
            .solve_async(&target_device, &mut output_device, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let mut first = vec![0.0; 3];
        output_device.copy_to(&mut first, &stream).unwrap();
        assert!(first.iter().all(|value| (0.0..=1.0).contains(value)));
        assert!((first[0] - 0.25).abs() < 0.03);
        assert!((first[1] - 0.5).abs() < 0.03);
        assert!((first[2] - 0.75).abs() < 0.03);
        solver.reset(&stream).unwrap();
        solver
            .solve_async(&target_device, &mut output_device, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let mut replay = vec![0.0; 3];
        output_device.copy_to(&mut replay, &stream).unwrap();
        assert_eq!(first, replay);
    }
}
