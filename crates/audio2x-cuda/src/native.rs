use crate::{DeviceId, ensure_same_device};
use audio2x_core::{Audio2xError, Result};
use cudarc::driver::sys::*;
use std::ffi::CString;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr;
use std::rc::Rc;

fn check(code: CUresult, operation: &'static str) -> Result<()> {
    if code == cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Audio2xError::Cuda {
            operation,
            code: code as u32,
        })
    }
}

#[derive(Debug)]
pub struct GpuDevice {
    id: DeviceId,
    raw_device: CUdevice,
    context: CUcontext,
    _not_send_sync: PhantomData<*mut ()>,
}

impl GpuDevice {
    pub fn new(ordinal: i32) -> Result<Rc<Self>> {
        let id = DeviceId::new(ordinal)?;
        let mut raw_device = 0;
        let mut context = ptr::null_mut();
        // SAFETY: output pointers are valid, and CUDA initialization precedes device access.
        unsafe {
            check(cuInit(0), "cuInit")?;
            check(cuDeviceGet(&mut raw_device, ordinal), "cuDeviceGet")?;
            check(
                cuDevicePrimaryCtxRetain(&mut context, raw_device),
                "cuDevicePrimaryCtxRetain",
            )?;
        }
        tracing::debug!(device = ordinal, "retained CUDA primary context");
        Ok(Rc::new(Self {
            id,
            raw_device,
            context,
            _not_send_sync: PhantomData,
        }))
    }

    pub const fn id(&self) -> DeviceId {
        self.id
    }

    fn make_current(&self) -> Result<()> {
        // SAFETY: this object retains the primary context until Drop.
        unsafe { check(cuCtxSetCurrent(self.context), "cuCtxSetCurrent") }
    }

    pub fn create_stream(self: &Rc<Self>) -> Result<CudaStream> {
        self.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: current context is retained and the output pointer is valid.
        unsafe { check(cuStreamCreate(&mut raw, 0), "cuStreamCreate")? };
        tracing::debug!(device = self.id().ordinal(), "created CUDA stream");
        Ok(CudaStream {
            device: Rc::clone(self),
            raw,
            _not_send_sync: PhantomData,
        })
    }

    pub fn allocate<T>(self: &Rc<Self>, len: usize) -> Result<DeviceBuffer<T>> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(Audio2xError::IntegerOverflow {
                field: "device_allocation_bytes",
                value: len,
                target: "usize",
            })?;
        if bytes == 0 {
            return Err(Audio2xError::CudaUnavailable(
                "zero-sized device allocations are unsupported".into(),
            ));
        }
        self.make_current()?;
        let mut pointer = 0;
        // SAFETY: the current context is retained and pointer is a valid output.
        unsafe { check(cuMemAlloc_v2(&mut pointer, bytes), "cuMemAlloc")? };
        tracing::debug!(
            device = self.id().ordinal(),
            elements = len,
            bytes,
            "allocated CUDA device memory"
        );
        Ok(DeviceBuffer {
            device: Rc::clone(self),
            pointer,
            len,
            _type: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    pub fn load_module(self: &Rc<Self>, ptx: &str) -> Result<CudaModule> {
        self.make_current()?;
        let ptx = CString::new(ptx)
            .map_err(|_| Audio2xError::CudaUnavailable("PTX contains a NUL byte".into()))?;
        let mut raw = ptr::null_mut();
        // SAFETY: PTX is NUL terminated and remains alive for the duration of the call.
        unsafe {
            check(
                cuModuleLoadData(&mut raw, ptx.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
        }
        Ok(CudaModule {
            device: Rc::clone(self),
            raw,
            _not_send_sync: PhantomData,
        })
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        // SAFETY: this object owns one primary-context retain count.
        unsafe {
            let _ = cuDevicePrimaryCtxRelease_v2(self.raw_device);
        }
    }
}

#[derive(Debug)]
pub struct CudaStream {
    device: Rc<GpuDevice>,
    raw: CUstream,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CudaStream {
    pub fn device_id(&self) -> DeviceId {
        self.device.id()
    }

    /// Returns the native CUDA stream handle for FFI interoperability.
    ///
    /// The returned handle is borrowed from this object and must not be
    /// destroyed. Any queued work must complete before this stream is dropped.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        self.raw.cast()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.make_current()?;
        // SAFETY: raw is owned by this object and remains valid for the call.
        unsafe { check(cuStreamSynchronize(self.raw), "cuStreamSynchronize") }
    }

    pub fn create_event(&self) -> Result<CudaEvent> {
        self.device.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: output pointer is valid and the current context is retained.
        unsafe { check(cuEventCreate(&mut raw, 0), "cuEventCreate")? };
        Ok(CudaEvent {
            device: Rc::clone(&self.device),
            raw,
            _not_send_sync: PhantomData,
        })
    }

    /// Enqueues a zero-fill for an arbitrary device allocation.
    ///
    /// # Safety
    ///
    /// `pointer..pointer + bytes` must be a writable allocation owned by this
    /// stream's CUDA context and must remain alive until the stream completes.
    pub unsafe fn memset_device_zero(&self, pointer: u64, bytes: usize) -> Result<()> {
        self.device.make_current()?;
        // SAFETY: the caller guarantees pointer validity, ownership, and lifetime.
        unsafe {
            check(
                cuMemsetD8Async(pointer, 0, bytes, self.raw),
                "cuMemsetD8Async",
            )
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let _ = self.device.make_current();
        // SAFETY: raw is exclusively owned by this object.
        unsafe {
            let _ = cuStreamSynchronize(self.raw);
            let _ = cuStreamDestroy_v2(self.raw);
        }
    }
}

#[derive(Debug)]
pub struct CudaModule {
    device: Rc<GpuDevice>,
    raw: CUmodule,
    _not_send_sync: PhantomData<*mut ()>,
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        let _ = self.device.make_current();
        // SAFETY: raw is exclusively owned by this object.
        unsafe {
            let _ = cuModuleUnload(self.raw);
        }
    }
}

impl CudaModule {
    /// Looks up a kernel entry point in this module.
    ///
    /// The returned function borrows the module, so it cannot outlive the
    /// loaded CUDA code that owns the native function handle.
    pub fn function(&self, name: &str) -> Result<CudaFunction<'_>> {
        self.device.make_current()?;
        let name = CString::new(name).map_err(|_| {
            Audio2xError::CudaUnavailable("CUDA function name contains a NUL byte".into())
        })?;
        let mut raw = ptr::null_mut();
        // SAFETY: the module is live, the name is NUL terminated, and raw is a
        // valid output pointer.
        unsafe {
            check(
                cuModuleGetFunction(&mut raw, self.raw, name.as_ptr()),
                "cuModuleGetFunction",
            )?
        };
        Ok(CudaFunction {
            module: self,
            raw,
            _not_send_sync: PhantomData,
        })
    }
}

/// A kernel entry point borrowed from a loaded [`CudaModule`].
#[derive(Debug)]
pub struct CudaFunction<'module> {
    module: &'module CudaModule,
    raw: CUfunction,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CudaFunction<'_> {
    pub fn device_id(&self) -> DeviceId {
        self.module.device.id()
    }

    /// Enqueues this kernel on `stream` using CUDA's raw parameter ABI.
    ///
    /// `grid` and `block` are `(x, y, z)` dimensions. CUDA reports invalid or
    /// unsupported launch dimensions through the returned error.
    ///
    /// # Safety
    ///
    /// Every entry in `kernel_params` must point to suitably aligned storage
    /// containing one kernel argument with the exact type and order declared by
    /// the PTX/CUDA entry point. Any referenced host argument storage must live
    /// until `cuLaunchKernel` returns. Device pointers encoded by those
    /// arguments must belong to this function's CUDA context and remain valid
    /// until all launched work completes. The caller must also prevent mutable
    /// aliasing and data races involving those allocations for that duration.
    pub unsafe fn launch_raw(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_memory_bytes: u32,
        stream: &CudaStream,
        kernel_params: &mut [*mut std::ffi::c_void],
    ) -> Result<()> {
        ensure_same_device(self.device_id(), stream.device_id())?;
        self.module.device.make_current()?;
        // SAFETY: argument ABI, allocation ownership, and asynchronous
        // lifetimes are delegated to the caller as documented above.
        unsafe {
            check(
                cuLaunchKernel(
                    self.raw,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_memory_bytes,
                    stream.raw,
                    kernel_params.as_mut_ptr(),
                    ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )
        }
    }
}

pub struct CublasHandle {
    device: Rc<GpuDevice>,
    raw: cudarc::cublas::sys::cublasHandle_t,
    _stream: PhantomData<*const CudaStream>,
    _not_send_sync: PhantomData<*mut ()>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CublasTranspose {
    None,
    Transpose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcaDimensions {
    pub shape_size: usize,
    pub shape_count: usize,
    pub batch_size: usize,
}

impl CublasHandle {
    pub fn new(stream: &CudaStream) -> Result<Self> {
        stream.device.make_current()?;
        let raw = cudarc::cublas::result::create_handle()
            .map_err(|error| Audio2xError::CudaUnavailable(format!("cuBLAS create: {error:?}")))?;
        // SAFETY: handle and stream are live and owned by the retained context.
        unsafe {
            cudarc::cublas::result::set_stream(raw, stream.raw.cast()).map_err(|error| {
                Audio2xError::CudaUnavailable(format!("cuBLAS set stream: {error:?}"))
            })?;
        }
        Ok(Self {
            device: Rc::clone(&stream.device),
            raw,
            _stream: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Enqueues column-major `y = alpha * op(A) * x + beta * y`.
    ///
    /// This low-level operation intentionally does not create a completion
    /// event, so several cuBLAS and kernel operations can form one pipeline.
    ///
    /// # Safety
    ///
    /// The caller must retain this handle, `matrix`, `input`, `output`, and
    /// `stream` until the queued operation completes. It must also prevent any
    /// conflicting access to the output during that interval.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn enqueue_matrix_vector(
        &self,
        matrix: DeviceView<'_, f32>,
        input: DeviceView<'_, f32>,
        output: &mut DeviceBuffer<f32>,
        rows: usize,
        columns: usize,
        transpose: CublasTranspose,
        alpha: f32,
        beta: f32,
        stream: &CudaStream,
    ) -> Result<()> {
        ensure_same_device(self.device.id(), stream.device_id())?;
        ensure_same_device(self.device.id(), matrix.device_id())?;
        ensure_same_device(self.device.id(), input.device_id())?;
        ensure_same_device(self.device.id(), output.device_id())?;
        let matrix_len = rows
            .checked_mul(columns)
            .ok_or(Audio2xError::IntegerOverflow {
                field: "matrix_vector_matrix",
                value: columns,
                target: "usize",
            })?;
        let (input_len, output_len, operation) = match transpose {
            CublasTranspose::None => (
                columns,
                rows,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
            ),
            CublasTranspose::Transpose => (
                rows,
                columns,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
            ),
        };
        if matrix.len() != matrix_len || input.len() != input_len || output.len() != output_len {
            return Err(Audio2xError::InvalidSchema(
                "cuBLAS matrix-vector dimensions do not match".into(),
            ));
        }
        let m = crate::audio2x_core_checked_i32(rows, "matrix_vector_rows")?;
        let n = crate::audio2x_core_checked_i32(columns, "matrix_vector_columns")?;
        self.device.make_current()?;
        // SAFETY: dimensions and device ownership were validated. The caller
        // provides the asynchronous resource lifetime and aliasing invariant.
        unsafe {
            cudarc::cublas::result::sgemv(
                self.raw,
                operation,
                m,
                n,
                &alpha,
                matrix.pointer as usize as *const f32,
                m,
                input.pointer as usize as *const f32,
                1,
                &beta,
                output.pointer as usize as *mut f32,
                1,
            )
            .map_err(|error| Audio2xError::CudaUnavailable(format!("cuBLAS SGEMV: {error:?}")))
        }
    }

    /// Enqueues column-major PCA reconstruction `Y = shapes * coefficients`.
    ///
    /// The returned fence borrows all device allocations, this handle, and the
    /// stream until synchronization, preventing asynchronous use-after-free.
    pub fn pca_reconstruct<'a>(
        &'a self,
        shapes: &'a DeviceBuffer<f32>,
        coefficients: &'a DeviceBuffer<f32>,
        output: &'a mut DeviceBuffer<f32>,
        dimensions: PcaDimensions,
        stream: &'a CudaStream,
    ) -> Result<CublasFence<'a>> {
        self.pca_reconstruct_views(
            shapes.view(),
            coefficients.view(),
            output,
            dimensions,
            stream,
        )
    }

    pub fn pca_reconstruct_views<'a>(
        &'a self,
        shapes: DeviceView<'a, f32>,
        coefficients: DeviceView<'a, f32>,
        output: &'a mut DeviceBuffer<f32>,
        dimensions: PcaDimensions,
        stream: &'a CudaStream,
    ) -> Result<CublasFence<'a>> {
        let PcaDimensions {
            shape_size,
            shape_count,
            batch_size,
        } = dimensions;
        ensure_same_device(self.device.id(), stream.device_id())?;
        ensure_same_device(self.device.id(), shapes.device_id())?;
        ensure_same_device(self.device.id(), coefficients.device_id())?;
        ensure_same_device(self.device.id(), output.device_id())?;
        let matrix_len =
            shape_size
                .checked_mul(shape_count)
                .ok_or(Audio2xError::IntegerOverflow {
                    field: "pca_matrix",
                    value: shape_count,
                    target: "usize",
                })?;
        let coefficients_len =
            shape_count
                .checked_mul(batch_size)
                .ok_or(Audio2xError::IntegerOverflow {
                    field: "pca_coefficients",
                    value: batch_size,
                    target: "usize",
                })?;
        let output_len =
            shape_size
                .checked_mul(batch_size)
                .ok_or(Audio2xError::IntegerOverflow {
                    field: "pca_output",
                    value: batch_size,
                    target: "usize",
                })?;
        if shapes.len() != matrix_len
            || coefficients.len() != coefficients_len
            || output.len() != output_len
        {
            return Err(Audio2xError::InvalidSchema(
                "PCA buffer dimensions do not match".into(),
            ));
        }
        let m = crate::audio2x_core_checked_i32(shape_size, "pca_shape_size")?;
        let n = crate::audio2x_core_checked_i32(batch_size, "pca_batch_size")?;
        let k = crate::audio2x_core_checked_i32(shape_count, "pca_shape_count")?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        self.device.make_current()?;
        // SAFETY: validated allocations cover column-major A(m*k), B(k*n), C(m*n),
        // and the returned fence holds every owner until the recorded event completes.
        unsafe {
            cudarc::cublas::result::sgemm(
                self.raw,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                shapes.pointer as usize as *const f32,
                m,
                coefficients.pointer as usize as *const f32,
                k,
                &beta,
                output.pointer as usize as *mut f32,
                m,
            )
            .map_err(|error| Audio2xError::CudaUnavailable(format!("cuBLAS SGEMM: {error:?}")))?;
        }
        let event = stream.create_event()?;
        event.record(stream)?;
        Ok(CublasFence {
            event,
            _resources: PhantomData,
        })
    }
}

pub struct CublasFence<'a> {
    event: CudaEvent,
    _resources: PhantomData<(
        &'a CublasHandle,
        &'a CudaStream,
        &'a DeviceBuffer<f32>,
        &'a mut DeviceBuffer<f32>,
    )>,
}

impl CublasFence<'_> {
    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        let _ = self.device.make_current();
        // SAFETY: raw is exclusively owned by this object.
        unsafe {
            let _ = cudarc::cublas::result::destroy_handle(self.raw);
        }
    }
}

pub struct CurandHandle {
    device: Rc<GpuDevice>,
    raw: cudarc::curand::sys::curandGenerator_t,
    _stream: PhantomData<*const CudaStream>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CurandHandle {
    pub fn new(stream: &CudaStream) -> Result<Self> {
        stream.device.make_current()?;
        let raw = cudarc::curand::result::create_generator_kind(
            cudarc::curand::sys::curandRngType_t::CURAND_RNG_PSEUDO_PHILOX4_32_10,
        )
        .map_err(|error| Audio2xError::CudaUnavailable(format!("cuRAND create: {error:?}")))?;
        // SAFETY: generator and stream are live and owned by the retained context.
        unsafe {
            cudarc::curand::result::set_stream(raw, stream.raw.cast()).map_err(|error| {
                Audio2xError::CudaUnavailable(format!("cuRAND set stream: {error:?}"))
            })?;
        }
        Ok(Self {
            device: Rc::clone(&stream.device),
            raw,
            _stream: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Resets the generator to an absolute element offset in its Philox stream.
    pub fn set_offset(&mut self, offset: u64) -> Result<()> {
        self.device.make_current()?;
        // SAFETY: `raw` is exclusively owned and remains allocated for this call.
        unsafe {
            cudarc::curand::result::set_offset(self.raw, offset).map_err(|error| {
                Audio2xError::CudaUnavailable(format!("cuRAND set offset: {error:?}"))
            })
        }
    }

    pub fn set_seed(&mut self, seed: u64) -> Result<()> {
        self.device.make_current()?;
        // SAFETY: `raw` is exclusively owned and is a pseudo-random generator.
        unsafe {
            cudarc::curand::result::set_seed(self.raw, seed).map_err(|error| {
                Audio2xError::CudaUnavailable(format!("cuRAND set seed: {error:?}"))
            })
        }
    }

    /// Enqueues standard-normal generation into a device allocation.
    ///
    /// cuRAND requires an even number of `f32` values. The returned fence
    /// retains the generator, stream and output allocation until completion.
    pub fn generate_normal<'a>(
        &'a mut self,
        output: &'a mut DeviceBuffer<f32>,
        stream: &'a CudaStream,
    ) -> Result<CurandFence<'a>> {
        ensure_same_device(self.device.id(), stream.device_id())?;
        ensure_same_device(self.device.id(), output.device_id())?;
        if output.is_empty() || output.len() % 2 != 0 {
            return Err(Audio2xError::InvalidSchema(
                "cuRAND normal output length must be non-zero and even".into(),
            ));
        }
        self.device.make_current()?;
        // SAFETY: output owns `len` writable f32 elements and the returned
        // fence prevents generator, stream, or allocation destruction.
        unsafe {
            cudarc::curand::result::generate::normal_f32(
                self.raw,
                output.pointer as usize as *mut f32,
                output.len,
                0.0,
                1.0,
            )
            .map_err(|error| {
                Audio2xError::CudaUnavailable(format!("cuRAND normal generation: {error:?}"))
            })?;
        }
        let event = stream.create_event()?;
        event.record(stream)?;
        Ok(CurandFence {
            event,
            _resources: PhantomData,
        })
    }
}

pub struct CurandFence<'a> {
    event: CudaEvent,
    _resources: PhantomData<(
        &'a mut CurandHandle,
        &'a CudaStream,
        &'a mut DeviceBuffer<f32>,
    )>,
}

impl CurandFence<'_> {
    pub fn synchronize(&self) -> Result<()> {
        self.event.synchronize()
    }
}

impl Drop for CurandHandle {
    fn drop(&mut self) {
        let _ = self.device.make_current();
        // SAFETY: raw is exclusively owned by this object.
        unsafe {
            let _ = cudarc::curand::result::destroy_generator(self.raw);
        }
    }
}

#[derive(Debug)]
pub struct CudaEvent {
    device: Rc<GpuDevice>,
    raw: CUevent,
    _not_send_sync: PhantomData<*mut ()>,
}

impl CudaEvent {
    /// Records this event after all work already queued on `stream`.
    ///
    /// Recording is asynchronous. The event and stream must remain alive until
    /// [`Self::synchronize`] completes or a waiting stream has completed its
    /// dependent work.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        ensure_same_device(self.device.id(), stream.device_id())?;
        self.device.make_current()?;
        // SAFETY: event and stream are live and belong to the same context.
        unsafe { check(cuEventRecord(self.raw, stream.raw), "cuEventRecord") }
    }

    /// Makes `stream` wait for this event without synchronizing the host.
    ///
    /// Both resources must remain alive until the waiting stream completes.
    pub fn wait_on(&self, stream: &CudaStream) -> Result<()> {
        ensure_same_device(self.device.id(), stream.device_id())?;
        self.device.make_current()?;
        // SAFETY: event and stream are live and belong to the same context.
        unsafe {
            check(
                cuStreamWaitEvent(stream.raw, self.raw, 0),
                "cuStreamWaitEvent",
            )
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        self.device.make_current()?;
        // SAFETY: raw is owned by this object and remains valid for the call.
        unsafe { check(cuEventSynchronize(self.raw), "cuEventSynchronize") }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let _ = self.device.make_current();
        // SAFETY: raw is exclusively owned by this object.
        unsafe {
            let _ = cuEventDestroy_v2(self.raw);
        }
    }
}

#[derive(Debug)]
pub struct DeviceBuffer<T> {
    device: Rc<GpuDevice>,
    pointer: CUdeviceptr,
    len: usize,
    _type: PhantomData<T>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<T> DeviceBuffer<T> {
    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device_id(&self) -> DeviceId {
        self.device.id()
    }

    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView {
            pointer: self.pointer,
            len: self.len,
            device: self.device.id(),
            _owner: PhantomData,
        }
    }

    /// Copies a complete host slice to this allocation.
    ///
    /// CUDA performs the transfer on `stream`, then this method synchronizes
    /// before returning. Consequently the host slice and allocation need no
    /// additional lifetime guard after a successful return.
    pub fn copy_from(&mut self, source: &[T], stream: &CudaStream) -> Result<()> {
        // SAFETY: synchronization below keeps source alive until the transfer completes.
        unsafe { self.copy_from_async(source, stream)? };
        stream.synchronize()
    }

    /// Enqueues a complete host-to-device copy without synchronization.
    ///
    /// # Safety
    ///
    /// `source` and this allocation must remain alive and must not be mutated
    /// until all preceding work on `stream` has completed.
    pub unsafe fn copy_from_async(&mut self, source: &[T], stream: &CudaStream) -> Result<()> {
        ensure_same_device(self.device.id(), stream.device_id())?;
        if source.len() != self.len {
            return Err(Audio2xError::InvalidSchema(format!(
                "copy length {} does not match allocation length {}",
                source.len(),
                self.len
            )));
        }
        self.device.make_current()?;
        // SAFETY: source and destination cover the validated byte count; their
        // asynchronous lifetime is delegated to the caller.
        unsafe {
            check(
                cuMemcpyHtoDAsync_v2(
                    self.pointer,
                    source.as_ptr().cast(),
                    std::mem::size_of_val(source),
                    stream.raw,
                ),
                "cuMemcpyHtoDAsync",
            )
        }
    }

    /// Copies this complete allocation to host memory and synchronizes before
    /// returning, so the destination is ready for immediate CPU access.
    pub fn copy_to(&self, destination: &mut [T], stream: &CudaStream) -> Result<()> {
        ensure_same_device(self.device.id(), stream.device_id())?;
        if destination.len() != self.len {
            return Err(Audio2xError::InvalidSchema(format!(
                "copy length {} does not match allocation length {}",
                destination.len(),
                self.len
            )));
        }
        self.device.make_current()?;
        // SAFETY: source and destination cover the validated byte count; synchronization
        // before return prevents the host slice from being accessed while CUDA writes it.
        unsafe {
            check(
                cuMemcpyDtoHAsync_v2(
                    destination.as_mut_ptr().cast(),
                    self.pointer,
                    std::mem::size_of_val(destination),
                    stream.raw,
                ),
                "cuMemcpyDtoHAsync",
            )?;
        }
        stream.synchronize()
    }

    /// Enqueues a byte-wise zero fill and synchronizes before returning.
    pub fn memset_zero(&mut self, stream: &CudaStream) -> Result<()> {
        ensure_same_device(self.device.id(), stream.device_id())?;
        self.device.make_current()?;
        let bytes = self
            .len
            .checked_mul(size_of::<T>())
            .ok_or(Audio2xError::IntegerOverflow {
                field: "device_memset_bytes",
                value: self.len,
                target: "usize",
            })?;
        // SAFETY: allocation owns at least `bytes`, and synchronization completes the write.
        unsafe {
            check(
                cuMemsetD8Async(self.pointer, 0, bytes, stream.raw),
                "cuMemsetD8Async",
            )?;
        }
        stream.synchronize()
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        let _ = self.device.make_current();
        // SAFETY: pointer is exclusively owned by this object.
        unsafe {
            let _ = cuMemFree_v2(self.pointer);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceView<'a, T> {
    pointer: CUdeviceptr,
    len: usize,
    device: DeviceId,
    _owner: PhantomData<&'a DeviceBuffer<T>>,
}

impl<T> DeviceView<'_, T> {
    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    pub const fn as_raw(&self) -> CUdeviceptr {
        self.pointer
    }

    pub fn slice(&self, offset: usize, len: usize) -> Result<DeviceView<'_, T>> {
        let end = offset
            .checked_add(len)
            .ok_or(Audio2xError::IntegerOverflow {
                field: "device_view_end",
                value: len,
                target: "usize",
            })?;
        if end > self.len {
            return Err(Audio2xError::InvalidSchema(format!(
                "device view range {offset}..{end} exceeds length {}",
                self.len
            )));
        }
        let byte_offset =
            offset
                .checked_mul(size_of::<T>())
                .ok_or(Audio2xError::IntegerOverflow {
                    field: "device_view_byte_offset",
                    value: offset,
                    target: "usize",
                })?;
        let pointer = self
            .pointer
            .checked_add(byte_offset as u64)
            .ok_or_else(|| Audio2xError::InvalidSchema("device view pointer overflow".into()))?;
        Ok(DeviceView {
            pointer,
            len,
            device: self.device,
            _owner: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADD_ONE_PTX: &str = r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry add_one(
    .param .u64 values,
    .param .u32 count
)
{
    .reg .pred %p;
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;

    ld.param.u64 %rd1, [values];
    ld.param.u32 %r1, [count];
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.s32 %r2, %r3, %r4, %r2;
    setp.ge.u32 %p, %r2, %r1;
    @%p bra DONE;
    mul.wide.u32 %rd2, %r2, 4;
    add.s64 %rd2, %rd1, %rd2;
    ld.global.u32 %r3, [%rd2];
    add.u32 %r3, %r3, 1;
    st.global.u32 [%rd2], %r3;
DONE:
    ret;
}
"#;

    #[test]
    fn stream_event_orders_device_memory() {
        let device = GpuDevice::new(0).unwrap();
        let producer = device.create_stream().unwrap();
        let consumer = device.create_stream().unwrap();
        let event = producer.create_event().unwrap();
        let mut buffer = device.allocate::<u32>(4).unwrap();

        buffer.copy_from(&[1, 2, 3, 4], &producer).unwrap();
        event.record(&producer).unwrap();
        event.wait_on(&consumer).unwrap();
        let mut output = [0; 4];
        buffer.copy_to(&mut output, &consumer).unwrap();
        assert_eq!(output, [1, 2, 3, 4]);

        buffer.memset_zero(&consumer).unwrap();
        buffer.copy_to(&mut output, &consumer).unwrap();
        assert_eq!(output, [0; 4]);
    }

    #[test]
    fn handles_bind_to_stream() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let _blas = CublasHandle::new(&stream).unwrap();
        let _rand = CurandHandle::new(&stream).unwrap();
    }

    #[test]
    fn philox_reset_replays_device_noise() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let mut generator = CurandHandle::new(&stream).unwrap();
        let mut values = device.allocate::<f32>(1024).unwrap();
        generator
            .generate_normal(&mut values, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let mut first = vec![0.0; values.len()];
        values.copy_to(&mut first, &stream).unwrap();
        generator.set_offset(0).unwrap();
        generator
            .generate_normal(&mut values, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let mut replay = vec![0.0; values.len()];
        values.copy_to(&mut replay, &stream).unwrap();
        assert_eq!(first, replay);
    }

    #[test]
    fn philox_device_noise_has_standard_normal_statistics() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let mut generator = CurandHandle::new(&stream).unwrap();
        let mut values = device.allocate::<f32>(100_000).unwrap();
        generator
            .generate_normal(&mut values, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
        let mut host = vec![0.0; values.len()];
        values.copy_to(&mut host, &stream).unwrap();
        let mean = host.iter().sum::<f32>() / host.len() as f32;
        let variance =
            host.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / host.len() as f32;
        assert!(mean.abs() < 0.02, "mean={mean}");
        assert!((variance - 1.0).abs() < 0.03, "variance={variance}");
    }

    #[test]
    fn pca_reconstruction_matches_column_major_product() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let handle = CublasHandle::new(&stream).unwrap();
        let mut shapes = device.allocate::<f32>(6).unwrap();
        let mut coefficients = device.allocate::<f32>(4).unwrap();
        let mut output = device.allocate::<f32>(6).unwrap();
        shapes
            .copy_from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &stream)
            .unwrap();
        coefficients
            .copy_from(&[1.0, 2.0, 3.0, 4.0], &stream)
            .unwrap();
        handle
            .pca_reconstruct(
                &shapes,
                &coefficients,
                &mut output,
                PcaDimensions {
                    shape_size: 3,
                    shape_count: 2,
                    batch_size: 2,
                },
                &stream,
            )
            .unwrap()
            .synchronize()
            .unwrap();
        let mut host = [0.0; 6];
        output.copy_to(&mut host, &stream).unwrap();
        assert_eq!(host, [9.0, 12.0, 15.0, 19.0, 26.0, 33.0]);
    }

    #[test]
    fn launches_named_ptx_function_on_stream() {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let module = device.load_module(ADD_ONE_PTX).unwrap();
        let function = module.function("add_one").unwrap();
        let mut values = device.allocate::<u32>(5).unwrap();
        values.copy_from(&[10, 20, 30, 40, 50], &stream).unwrap();

        let mut pointer = values.view().as_raw();
        let mut count = u32::try_from(values.len()).unwrap();
        let mut params = [
            (&mut pointer as *mut CUdeviceptr).cast(),
            (&mut count as *mut u32).cast(),
        ];
        // SAFETY: params exactly match add_one(u64, u32); values belongs to the
        // same device and copy_to synchronizes before values is released.
        unsafe {
            function
                .launch_raw((1, 1, 1), (32, 1, 1), 0, &stream, &mut params)
                .unwrap();
        }

        let mut output = [0; 5];
        values.copy_to(&mut output, &stream).unwrap();
        assert_eq!(output, [11, 21, 31, 41, 51]);
    }

    #[test]
    fn device_view_slice_checks_bounds_and_offsets_pointer() {
        let device = GpuDevice::new(0).unwrap();
        let values = device.allocate::<f32>(8).unwrap();
        let view = values.view();
        let slice = view.slice(3, 2).unwrap();
        assert_eq!(slice.len(), 2);
        assert_eq!(slice.as_raw(), view.as_raw() + 3 * size_of::<f32>() as u64);
        assert!(view.slice(7, 2).is_err());
    }
}
