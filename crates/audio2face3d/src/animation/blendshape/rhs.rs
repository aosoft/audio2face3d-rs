//! Native CPU solver RHS: GPU gather/subtract and the SDK's transposed SGEMV.
//! Only the small active-pose vector is copied to the CPU BVLS stage.

use super::PreparedBlendshape;
use crate::common::{Error, Result};
use crate::cuda::{
    CublasHandle, CublasTranspose, CudaModule, CudaStream, DeviceBuffer, DeviceView, GpuDevice,
    blendshape_solver_ptx, ensure_same_device,
};
use std::ffi::c_void;
use std::sync::Arc;

pub(crate) struct GpuRhs {
    stream: Arc<CudaStream>,
    module: CudaModule,
    blas: CublasHandle,
    indices: DeviceBuffer<u32>,
    neutral: DeviceBuffer<f32>,
    deltas: DeviceBuffer<f32>,
    target_delta: DeviceBuffer<f32>,
    atb: DeviceBuffer<f32>,
    source_len: usize,
}

impl GpuRhs {
    pub(crate) fn active_count(&self) -> usize {
        self.atb.len()
    }
    pub(crate) fn new(
        prepared: &PreparedBlendshape,
        source_len: usize,
        device: &Arc<GpuDevice>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        ensure_same_device(device.id(), stream.device_id())?;
        if prepared
            .coordinate_indices
            .iter()
            .any(|index| *index >= source_len)
        {
            return Err(Error::InvalidSchema(
                "BlendShape mask exceeds target size".into(),
            ));
        }
        let indices = prepared
            .coordinate_indices
            .iter()
            .map(|index| {
                u32::try_from(*index)
                    .map_err(|_| Error::InvalidSchema("BlendShape coordinate exceeds u32".into()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut index_buffer = device.allocate(indices.len())?;
        index_buffer.copy_from(&indices, &stream)?;
        let mut neutral = device.allocate(prepared.active_neutral.len())?;
        neutral.copy_from(&prepared.active_neutral, &stream)?;
        let mut deltas = device.allocate(prepared.active_deltas.len())?;
        deltas.copy_from(&prepared.active_deltas, &stream)?;
        let blas = CublasHandle::new(&stream)?;
        Ok(Self {
            stream,
            module: device.load_module(blendshape_solver_ptx())?,
            blas,
            indices: index_buffer,
            neutral,
            deltas,
            target_delta: device.allocate(indices.len())?,
            atb: device.allocate(prepared.active_indices.len())?,
            source_len,
        })
    }

    /// Target must be ready on the retained stream before this call. Resources
    /// remain borrowed until the copy/synchronization completes, including errors.
    pub(crate) fn compute(
        &mut self,
        target: DeviceView<'_, f32>,
        output: &mut [f32],
    ) -> Result<()> {
        ensure_same_device(target.device_id(), self.stream.device_id())?;
        if target.len() != self.source_len || output.len() != self.atb.len() {
            return Err(Error::InvalidSchema(
                "BlendShape GPU RHS dimensions do not match".into(),
            ));
        }
        let mut count = u32::try_from(self.indices.len())
            .map_err(|_| Error::InvalidSchema("BlendShape coordinate count exceeds u32".into()))?;
        let mut delta = self.target_delta.view().as_raw();
        let mut input = target.as_raw();
        let mut neutral = self.neutral.view().as_raw();
        let mut indices = self.indices.view().as_raw();
        let mut params = [
            std::ptr::from_mut(&mut delta).cast::<c_void>(),
            std::ptr::from_mut(&mut input).cast::<c_void>(),
            std::ptr::from_mut(&mut neutral).cast::<c_void>(),
            std::ptr::from_mut(&mut indices).cast::<c_void>(),
            std::ptr::from_mut(&mut count).cast::<c_void>(),
        ];
        let result = (|| {
            let gather = self.module.function("blendshape_gather_subtract")?;
            // SAFETY: arguments match the kernel ABI; the prepared mask indexes
            // the checked target size. All allocations and the module live until
            // the blocking copy or error-path synchronization below finishes.
            unsafe {
                gather.launch_raw(
                    (count.div_ceil(256), 1, 1),
                    (256, 1, 1),
                    0,
                    &self.stream,
                    &mut params,
                )?;
                self.blas.enqueue_matrix_vector(
                    self.deltas.view(),
                    self.target_delta.view(),
                    &mut self.atb,
                    self.indices.len(),
                    output.len(),
                    CublasTranspose::Transpose,
                    1.0,
                    0.0,
                    &self.stream,
                )?;
            }
            self.atb.copy_to(output, &self.stream)
        })();
        if result.is_err() {
            // A successful gather may precede a later enqueue/copy failure.
            // Do not release the borrowed target with work still queued.
            let _ = self.stream.synchronize();
        }
        result
    }
}
