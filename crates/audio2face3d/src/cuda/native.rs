use super::api::CudaApi;
use crate::common::{Error, Result};
use crate::cuda::{CudaStreamRef, DeviceId, DeviceView, ensure_same_device};
use cudarc::driver::sys::*;
use std::ffi::CString;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Mutex};

/// Restores the caller's thread-local CUDA context when the guard is dropped.
/// CUDA contexts are thread-local, so every native entry point must establish
/// the context it owns without leaking that change to an embedding process.
pub(crate) struct CurrentContextGuard {
    previous: CUcontext,
    api: Arc<CudaApi>,
}

impl CurrentContextGuard {
    fn enter(target: CUcontext, api: Arc<CudaApi>) -> Result<Self> {
        let mut previous = ptr::null_mut();
        // SAFETY: CUDA writes one context handle to a valid output pointer.
        unsafe {
            check(
                (api.driver.cuCtxGetCurrent)(&mut previous),
                "cuCtxGetCurrent",
            )?
        };
        if previous != target {
            // SAFETY: target is retained by the owning GpuDevice.
            unsafe { check((api.driver.cuCtxSetCurrent)(target), "cuCtxSetCurrent")? };
        }
        Ok(Self { previous, api })
    }
}

impl Drop for CurrentContextGuard {
    fn drop(&mut self) {
        // SAFETY: restoring the context is best-effort during unwinding/drop;
        // the previous handle was returned by CUDA on this same thread.
        unsafe {
            let _ = (self.api.driver.cuCtxSetCurrent)(self.previous);
        }
    }
}

fn check(code: CUresult, operation: &'static str) -> Result<()> {
    if code == cudaError_enum::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(Error::Cuda {
            operation,
            code: code as u32,
        })
    }
}

#[derive(Debug)]
pub struct GpuDevice {
    scope: crate::logging::integration::LogScope,
    id: DeviceId,
    api: Arc<CudaApi>,
    raw_device: CUdevice,
    context: CUcontext,
}

// SAFETY: the primary context retain is reference-counted by `Arc<GpuDevice>`;
// all driver calls install the context on the calling thread, and the final
// release occurs only after all child resources have been dropped.
unsafe impl Send for GpuDevice {}
// SAFETY: concurrent immutable access uses per-call context guards; child
// resource ownership retains the primary context through the final call.
unsafe impl Sync for GpuDevice {}

impl GpuDevice {
    pub fn new(ordinal: i32) -> Result<Arc<Self>> {
        let id = DeviceId::new(ordinal)?;
        let scope = crate::logging::integration::LogScope::capture();
        let api = CudaApi::initialize(scope.context())?;
        let mut raw_device = 0;
        let mut context = ptr::null_mut();
        // SAFETY: output pointers are valid, and CUDA initialization precedes device access.
        unsafe {
            check(
                (api.driver.cuDeviceGet)(&mut raw_device, ordinal),
                "cuDeviceGet",
            )?;
            check(
                (api.driver.cuDevicePrimaryCtxRetain)(&mut context, raw_device),
                "cuDevicePrimaryCtxRetain",
            )?;
        }
        scope.context().native_device_ready();
        crate::logging::integration::LogScope::capture().log(
            crate::logging::LogLevel::Debug,
            || {
                crate::logging::LogRecord::new("retained CUDA primary context")
                    .field("source", module_path!())
                    .field("device", (ordinal) as u64)
            },
        );
        Ok(Arc::new(Self {
            scope,
            api,
            id,
            raw_device,
            context,
        }))
    }

    #[cfg(feature = "tensorrt")]
    pub(crate) fn runtime_context(&self) -> &crate::Audio2Face3DContext {
        self.scope.context()
    }

    pub const fn id(&self) -> DeviceId {
        self.id
    }

    pub(crate) fn make_current(&self) -> Result<CurrentContextGuard> {
        let _scope = self.scope.activate();
        CurrentContextGuard::enter(self.context, Arc::clone(&self.api))
    }

    #[cfg(all(feature = "animation", feature = "tensorrt"))]
    pub(crate) fn synchronize_borrowed_stream(&self, stream: CudaStreamRef<'_>) -> Result<()> {
        let _scope = self.scope.activate();
        ensure_same_device(self.id(), stream.device_id())?;
        let _context = self.make_current()?;
        // SAFETY: the callback-scoped descriptor guarantees that the stream
        // owner remains alive for this call.
        unsafe {
            check(
                (self.api.driver.cuStreamSynchronize)(stream.as_raw().cast()),
                "cuStreamSynchronize",
            )
        }
    }

    pub fn create_stream(self: &Arc<Self>) -> Result<CudaStream> {
        let _context = self.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: current context is retained and the output pointer is valid.
        unsafe {
            check(
                (self.api.driver.cuStreamCreate)(&mut raw, 0),
                "cuStreamCreate",
            )?
        };
        crate::logging::integration::LogScope::capture().log(
            crate::logging::LogLevel::Debug,
            || {
                crate::logging::LogRecord::new("created CUDA stream")
                    .field("source", module_path!())
                    .field("device", (self.id().ordinal()) as u64)
            },
        );
        Ok(CudaStream {
            device: Arc::clone(self),
            raw,
        })
    }

    pub fn allocate<T>(self: &Arc<Self>, len: usize) -> Result<DeviceBuffer<T>> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(Error::IntegerOverflow {
                field: "device_allocation_bytes",
                value: len,
                target: "usize",
            })?;
        if bytes == 0 {
            return Err(Error::CudaUnavailable(
                "zero-sized device allocations are unsupported".into(),
            ));
        }
        let _context = self.make_current()?;
        let mut pointer = 0;
        // SAFETY: the current context is retained and pointer is a valid output.
        unsafe {
            check(
                (self.api.driver.cuMemAlloc_v2)(&mut pointer, bytes),
                "cuMemAlloc",
            )?
        };
        crate::logging::integration::LogScope::capture().log(
            crate::logging::LogLevel::Debug,
            || {
                crate::logging::LogRecord::new("allocated CUDA device memory")
                    .field("source", module_path!())
                    .field("device", (self.id().ordinal()) as u64)
                    .field("elements", (len) as u64)
                    .field("bytes", (bytes) as u64)
            },
        );
        Ok(DeviceBuffer {
            device: Arc::clone(self),
            pointer,
            len,
            _type: PhantomData,
        })
    }

    pub fn load_module(self: &Arc<Self>, ptx: &str) -> Result<CudaModule> {
        let _context = self.make_current()?;
        let ptx = CString::new(ptx)
            .map_err(|_| Error::CudaUnavailable("PTX contains a NUL byte".into()))?;
        let mut raw = ptr::null_mut();
        // SAFETY: PTX is NUL terminated and remains alive for the duration of the call.
        unsafe {
            check(
                (self.api.driver.cuModuleLoadData)(&mut raw, ptx.as_ptr().cast()),
                "cuModuleLoadData",
            )?;
        }
        Ok(CudaModule {
            device: Arc::clone(self),
            raw,
        })
    }
}

impl Drop for GpuDevice {
    fn drop(&mut self) {
        let _scope = self.scope.activate();
        let _context = self.make_current();
        // SAFETY: this object owns one primary-context retain count. All child
        // resources hold an Arc and therefore outlive this final release.
        unsafe {
            let _ = (self.api.driver.cuDevicePrimaryCtxRelease_v2)(self.raw_device);
        }
    }
}

#[derive(Debug)]
pub struct CudaStream {
    device: Arc<GpuDevice>,
    raw: CUstream,
}

// SAFETY: CUDA stream handles are designed for concurrent host submission;
// destruction is serialized by unique ownership and all calls establish the
// stream's retained context first.
unsafe impl Send for CudaStream {}
// SAFETY: CUDA permits concurrent host submission to a stream, while Rust
// ownership and the retained device context serialize destruction.
unsafe impl Sync for CudaStream {}

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

    /// Borrows this stream for a device-result callback.
    ///
    /// The returned reference has the same portable type regardless of whether
    /// downstream code is type-checked with the `cuda` feature enabled.
    pub fn as_ref(&self) -> CudaStreamRef<'_> {
        // SAFETY: the borrow prevents this owning stream from being dropped for
        // the returned lifetime, and the stream's device cannot change.
        unsafe {
            CudaStreamRef::from_raw(self.as_raw(), self.device.context.cast(), self.device_id())
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        let _context = self.device.make_current()?;
        // SAFETY: raw is owned by this object and remains valid for the call.
        unsafe {
            check(
                (self.device.api.driver.cuStreamSynchronize)(self.raw),
                "cuStreamSynchronize",
            )
        }
    }

    pub fn create_event(&self) -> Result<CudaEvent> {
        let _context = self.device.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: output pointer is valid and the current context is retained.
        unsafe {
            check(
                (self.device.api.driver.cuEventCreate)(&mut raw, 0),
                "cuEventCreate",
            )?
        };
        Ok(CudaEvent {
            device: Arc::clone(&self.device),
            raw,
            record_lock: Mutex::new(()),
        })
    }

    /// Enqueues a zero-fill for an arbitrary device allocation.
    ///
    /// # Safety
    ///
    /// `pointer..pointer + bytes` must be a writable allocation owned by this
    /// stream's CUDA context and must remain alive until the stream completes.
    pub unsafe fn memset_device_zero(&self, pointer: u64, bytes: usize) -> Result<()> {
        let _context = self.device.make_current()?;
        // SAFETY: the caller guarantees pointer validity, ownership, and lifetime.
        unsafe {
            check(
                (self.device.api.driver.cuMemsetD8Async)(pointer, 0, bytes, self.raw),
                "cuMemsetD8Async",
            )
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let _context = self.device.make_current();
        // SAFETY: raw is exclusively owned by this object.
        unsafe {
            let _ = (self.device.api.driver.cuStreamSynchronize)(self.raw);
            let _ = (self.device.api.driver.cuStreamDestroy_v2)(self.raw);
        }
    }
}

pub(crate) fn copy_device_view_to_host<T: Copy>(
    source: DeviceView<'_, T>,
    destination: &mut [T],
    stream: CudaStreamRef<'_>,
) -> Result<()> {
    ensure_same_device(source.device_id(), stream.device_id())?;
    if destination.len() != source.len() {
        return Err(Error::InvalidSchema(format!(
            "copy length {} does not match device view length {}",
            destination.len(),
            source.len()
        )));
    }
    let api = CudaApi::loaded()?;
    let _context = CurrentContextGuard::enter(stream.context_raw().cast(), Arc::clone(&api))?;
    // SAFETY: the callback-scoped view covers the checked byte count, the host
    // destination remains borrowed through synchronization, and the borrowed
    // stream/context remain alive for this call.
    unsafe {
        check(
            (api.driver.cuMemcpyDtoHAsync_v2)(
                destination.as_mut_ptr().cast(),
                source.as_raw(),
                std::mem::size_of_val(destination),
                stream.as_raw().cast(),
            ),
            "cuMemcpyDtoHAsync",
        )?;
        check(
            (api.driver.cuStreamSynchronize)(stream.as_raw().cast()),
            "cuStreamSynchronize",
        )
    }
}

#[derive(Debug)]
pub struct CudaModule {
    device: Arc<GpuDevice>,
    raw: CUmodule,
}

// SAFETY: module loading is immutable after construction; its retained device
// context and Drop implementation make moving/sharing the handle safe.
unsafe impl Send for CudaModule {}
// SAFETY: the loaded module is immutable, and borrowed functions prevent
// unloading while a safe function descriptor remains live.
unsafe impl Sync for CudaModule {}

impl Drop for CudaModule {
    fn drop(&mut self) {
        let _context = self.device.make_current();
        // SAFETY: draining the retained context ensures no queued kernel still
        // references module code; `raw` is exclusively owned here.
        unsafe {
            let _ = (self.device.api.driver.cuCtxSynchronize)();
            let _ = (self.device.api.driver.cuModuleUnload)(self.raw);
        }
    }
}

impl CudaModule {
    /// Looks up a kernel entry point in this module.
    ///
    /// The returned function borrows the module, so it cannot outlive the
    /// loaded CUDA code that owns the native function handle.
    pub fn function(&self, name: &str) -> Result<CudaFunction<'_>> {
        let _context = self.device.make_current()?;
        let name = CString::new(name)
            .map_err(|_| Error::CudaUnavailable("CUDA function name contains a NUL byte".into()))?;
        let mut raw = ptr::null_mut();
        // SAFETY: the module is live, the name is NUL terminated, and raw is a
        // valid output pointer.
        unsafe {
            check(
                (self.device.api.driver.cuModuleGetFunction)(&mut raw, self.raw, name.as_ptr()),
                "cuModuleGetFunction",
            )?
        };
        Ok(CudaFunction { module: self, raw })
    }
}

/// A kernel entry point borrowed from a loaded [`CudaModule`].
#[derive(Debug)]
pub struct CudaFunction<'module> {
    module: &'module CudaModule,
    raw: CUfunction,
}

// SAFETY: a function handle is immutable and borrows its live module.
unsafe impl Send for CudaFunction<'_> {}
// SAFETY: the function handle is immutable and its module borrow prevents
// concurrent module destruction.
unsafe impl Sync for CudaFunction<'_> {}

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
        let _context = self.module.device.make_current()?;
        // SAFETY: argument ABI, allocation ownership, and asynchronous
        // lifetimes are delegated to the caller as documented above.
        unsafe {
            check(
                (self.module.device.api.driver.cuLaunchKernel)(
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
    device: Arc<GpuDevice>,
    raw: cudarc::cublas::sys::cublasHandle_t,
    _stream: PhantomData<*const CudaStream>,
    call_lock: Mutex<()>,
}

// SAFETY: the handle's stream/configuration is fixed at construction and all
// host calls are serialized by `call_lock`; the retained device context is
// installed by every operation.
unsafe impl Send for CublasHandle {}
// SAFETY: all calls and mutable library state are serialized by `call_lock`;
// stream selection is fixed for the handle's lifetime.
unsafe impl Sync for CublasHandle {}

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
        let _context = stream.device.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: output storage is valid and the device retains the loaded API.
        unsafe { (stream.device.api.cublas.cublasCreate_v2)(&mut raw).result() }
            .map_err(|error| Error::CudaUnavailable(format!("cuBLAS create: {error:?}")))?;
        // SAFETY: raw and stream are valid; failed stream setup releases the new handle.
        unsafe {
            if let Err(error) =
                (stream.device.api.cublas.cublasSetStream_v2)(raw, stream.raw.cast()).result()
            {
                let _ = (stream.device.api.cublas.cublasDestroy_v2)(raw);
                return Err(Error::CudaUnavailable(format!(
                    "cuBLAS set stream: {error:?}"
                )));
            }
        }
        Ok(Self {
            device: Arc::clone(&stream.device),
            raw,
            _stream: PhantomData,
            call_lock: Mutex::new(()),
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
        let matrix_len = rows.checked_mul(columns).ok_or(Error::IntegerOverflow {
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
            return Err(Error::InvalidSchema(
                "cuBLAS matrix-vector dimensions do not match".into(),
            ));
        }
        let m = crate::cuda::checked_i32_for_cuda(rows, "matrix_vector_rows")?;
        let n = crate::cuda::checked_i32_for_cuda(columns, "matrix_vector_columns")?;
        let _call = self.call_lock.lock().map_err(|_| Error::Poisoned {
            resource: "cublas_handle",
        })?;
        let _context = self.device.make_current()?;
        // SAFETY: dimensions and device ownership were validated. The caller
        // provides the asynchronous resource lifetime and aliasing invariant.
        unsafe {
            (self.device.api.cublas.cublasSgemv_v2)(
                self.raw,
                operation,
                m,
                n,
                &alpha,
                matrix.as_raw() as usize as *const f32,
                m,
                input.as_raw() as usize as *const f32,
                1,
                &beta,
                output.pointer as usize as *mut f32,
                1,
            )
            .result()
            .map_err(|error| Error::CudaUnavailable(format!("cuBLAS SGEMV: {error:?}")))
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
        let matrix_len = shape_size
            .checked_mul(shape_count)
            .ok_or(Error::IntegerOverflow {
                field: "pca_matrix",
                value: shape_count,
                target: "usize",
            })?;
        let coefficients_len =
            shape_count
                .checked_mul(batch_size)
                .ok_or(Error::IntegerOverflow {
                    field: "pca_coefficients",
                    value: batch_size,
                    target: "usize",
                })?;
        let output_len = shape_size
            .checked_mul(batch_size)
            .ok_or(Error::IntegerOverflow {
                field: "pca_output",
                value: batch_size,
                target: "usize",
            })?;
        if shapes.len() != matrix_len
            || coefficients.len() != coefficients_len
            || output.len() != output_len
        {
            return Err(Error::InvalidSchema(
                "PCA buffer dimensions do not match".into(),
            ));
        }
        let m = crate::cuda::checked_i32_for_cuda(shape_size, "pca_shape_size")?;
        let n = crate::cuda::checked_i32_for_cuda(batch_size, "pca_batch_size")?;
        let k = crate::cuda::checked_i32_for_cuda(shape_count, "pca_shape_count")?;
        let alpha = 1.0_f32;
        let beta = 0.0_f32;
        let _call = self.call_lock.lock().map_err(|_| Error::Poisoned {
            resource: "cublas_handle",
        })?;
        let _context = self.device.make_current()?;
        // SAFETY: validated allocations cover column-major A(m*k), B(k*n), C(m*n),
        // and the returned fence holds every owner until the recorded event completes.
        unsafe {
            (self.device.api.cublas.cublasSgemm_v2)(
                self.raw,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                shapes.as_raw() as usize as *const f32,
                m,
                coefficients.as_raw() as usize as *const f32,
                k,
                &beta,
                output.pointer as usize as *mut f32,
                m,
            )
            .result()
            .map_err(|error| Error::CudaUnavailable(format!("cuBLAS SGEMM: {error:?}")))?;
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
        let _context = self.device.make_current();
        // SAFETY: the context is retained, queued work is drained first, and
        // this object exclusively owns the cuBLAS handle.
        unsafe {
            let _ = (self.device.api.driver.cuCtxSynchronize)();
            let _ = (self.device.api.cublas.cublasDestroy_v2)(self.raw).result();
        }
    }
}

pub struct CurandHandle {
    device: Arc<GpuDevice>,
    raw: cudarc::curand::sys::curandGenerator_t,
    _stream: PhantomData<*const CudaStream>,
}

// SAFETY: generator mutation is exposed only through `&mut self`, and the
// retained device/context is installed before every native operation.
unsafe impl Send for CurandHandle {}

impl CurandHandle {
    pub fn new(stream: &CudaStream) -> Result<Self> {
        let _context = stream.device.make_current()?;
        let mut raw = ptr::null_mut();
        // SAFETY: output storage is valid and the device retains the loaded API.
        unsafe {
            (stream.device.api.curand.curandCreateGenerator)(
                &mut raw,
                cudarc::curand::sys::curandRngType_t::CURAND_RNG_PSEUDO_PHILOX4_32_10,
            )
            .result()
        }
        .map_err(|error| Error::CudaUnavailable(format!("cuRAND create: {error:?}")))?;
        // SAFETY: raw and stream are valid; failed setup releases the new generator.
        unsafe {
            if let Err(error) =
                (stream.device.api.curand.curandSetStream)(raw, stream.raw.cast()).result()
            {
                let _ = (stream.device.api.curand.curandDestroyGenerator)(raw);
                return Err(Error::CudaUnavailable(format!(
                    "cuRAND set stream: {error:?}"
                )));
            }
        }
        Ok(Self {
            device: Arc::clone(&stream.device),
            raw,
            _stream: PhantomData,
        })
    }

    /// Resets the generator to an absolute element offset in its Philox stream.
    pub fn set_offset(&mut self, offset: u64) -> Result<()> {
        let _context = self.device.make_current()?;
        // SAFETY: `raw` is exclusively owned and remains allocated for this call.
        unsafe {
            (self.device.api.curand.curandSetGeneratorOffset)(self.raw, offset)
                .result()
                .map_err(|error| Error::CudaUnavailable(format!("cuRAND set offset: {error:?}")))
        }
    }

    pub fn set_seed(&mut self, seed: u64) -> Result<()> {
        let _context = self.device.make_current()?;
        // SAFETY: `raw` is exclusively owned and is a pseudo-random generator.
        unsafe {
            (self.device.api.curand.curandSetPseudoRandomGeneratorSeed)(self.raw, seed)
                .result()
                .map_err(|error| Error::CudaUnavailable(format!("cuRAND set seed: {error:?}")))
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
        if output.is_empty() || !output.len().is_multiple_of(2) {
            return Err(Error::InvalidSchema(
                "cuRAND normal output length must be non-zero and even".into(),
            ));
        }
        let _context = self.device.make_current()?;
        // SAFETY: output owns `len` writable f32 elements and the returned
        // fence prevents generator, stream, or allocation destruction.
        unsafe {
            (self.device.api.curand.curandGenerateNormal)(
                self.raw,
                output.pointer as usize as *mut f32,
                output.len,
                0.0,
                1.0,
            )
            .result()
            .map_err(|error| {
                Error::CudaUnavailable(format!("cuRAND normal generation: {error:?}"))
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
        let _context = self.device.make_current();
        // SAFETY: the context is retained, queued work is drained first, and
        // this object exclusively owns the cuRAND generator.
        unsafe {
            let _ = (self.device.api.driver.cuCtxSynchronize)();
            let _ = (self.device.api.curand.curandDestroyGenerator)(self.raw).result();
        }
    }
}

#[derive(Debug)]
pub struct CudaEvent {
    device: Arc<GpuDevice>,
    raw: CUevent,
    record_lock: Mutex<()>,
}

// SAFETY: event record/wait/synchronize operations are serialized by
// `record_lock`; destruction is unique and all calls establish the context.
unsafe impl Send for CudaEvent {}
// SAFETY: record/wait/synchronize calls are protected by `record_lock`, and
// the Arc-held device context outlives the event.
unsafe impl Sync for CudaEvent {}

impl CudaEvent {
    /// Records this event after all work already queued on `stream`.
    ///
    /// Recording is asynchronous. The event and stream must remain alive until
    /// [`Self::synchronize`] completes or a waiting stream has completed its
    /// dependent work.
    pub fn record(&self, stream: &CudaStream) -> Result<()> {
        let _lock = self.record_lock.lock().map_err(|_| Error::Poisoned {
            resource: "cuda_event",
        })?;
        ensure_same_device(self.device.id(), stream.device_id())?;
        let _context = self.device.make_current()?;
        // SAFETY: event and stream are live and belong to the same context.
        unsafe {
            check(
                (self.device.api.driver.cuEventRecord)(self.raw, stream.raw),
                "cuEventRecord",
            )
        }
    }

    /// Makes `stream` wait for this event without synchronizing the host.
    ///
    /// Both resources must remain alive until the waiting stream completes.
    pub fn wait_on(&self, stream: &CudaStream) -> Result<()> {
        let _lock = self.record_lock.lock().map_err(|_| Error::Poisoned {
            resource: "cuda_event",
        })?;
        ensure_same_device(self.device.id(), stream.device_id())?;
        let _context = self.device.make_current()?;
        // SAFETY: event and stream are live and belong to the same context.
        unsafe {
            check(
                (self.device.api.driver.cuStreamWaitEvent)(stream.raw, self.raw, 0),
                "cuStreamWaitEvent",
            )
        }
    }

    pub fn synchronize(&self) -> Result<()> {
        let _lock = self.record_lock.lock().map_err(|_| Error::Poisoned {
            resource: "cuda_event",
        })?;
        let _context = self.device.make_current()?;
        // SAFETY: raw is owned by this object and remains valid for the call.
        unsafe {
            check(
                (self.device.api.driver.cuEventSynchronize)(self.raw),
                "cuEventSynchronize",
            )
        }
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        let _context = self.device.make_current();
        // SAFETY: this event is uniquely owned; synchronizing before destroy
        // prevents pending stream dependencies from retaining it.
        unsafe {
            let _ = (self.device.api.driver.cuEventSynchronize)(self.raw);
            let _ = (self.device.api.driver.cuEventDestroy_v2)(self.raw);
        }
    }
}

#[derive(Debug)]
pub struct DeviceBuffer<T> {
    device: Arc<GpuDevice>,
    pointer: CUdeviceptr,
    len: usize,
    _type: PhantomData<T>,
}

// SAFETY: the allocation is uniquely owned; safe writes require `&mut self`,
// and async APIs document the fence/lifetime requirement. The device owner is
// shareable and every native call establishes its context.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
// SAFETY: shared access exposes only a read-only device descriptor when `T`
// is Sync; mutation still requires `&mut self` or an explicitly fenced API.
unsafe impl<T: Sync> Sync for DeviceBuffer<T> {}

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
        // SAFETY: the returned lifetime is borrowed from this allocation, which
        // owns `pointer` and keeps its device context retained.
        unsafe { DeviceView::from_raw_parts(self.pointer, self.len, self.device.id()) }
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
            return Err(Error::InvalidSchema(format!(
                "copy length {} does not match allocation length {}",
                source.len(),
                self.len
            )));
        }
        let _context = self.device.make_current()?;
        // SAFETY: source and destination cover the validated byte count; their
        // asynchronous lifetime is delegated to the caller.
        unsafe {
            check(
                (self.device.api.driver.cuMemcpyHtoDAsync_v2)(
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
            return Err(Error::InvalidSchema(format!(
                "copy length {} does not match allocation length {}",
                destination.len(),
                self.len
            )));
        }
        let _context = self.device.make_current()?;
        // SAFETY: source and destination cover the validated byte count; synchronization
        // before return prevents the host slice from being accessed while CUDA writes it.
        unsafe {
            check(
                (self.device.api.driver.cuMemcpyDtoHAsync_v2)(
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
        let _context = self.device.make_current()?;
        let bytes = self
            .len
            .checked_mul(size_of::<T>())
            .ok_or(Error::IntegerOverflow {
                field: "device_memset_bytes",
                value: self.len,
                target: "usize",
            })?;
        // SAFETY: allocation owns at least `bytes`, and synchronization completes the write.
        unsafe {
            check(
                (self.device.api.driver.cuMemsetD8Async)(self.pointer, 0, bytes, stream.raw),
                "cuMemsetD8Async",
            )?;
        }
        stream.synchronize()
    }

    /// Copies a range from another allocation on the same device and stream.
    pub fn copy_from_device_range(
        &mut self,
        target_offset: usize,
        source: &DeviceBuffer<T>,
        source_offset: usize,
        len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        ensure_same_device(self.device.id(), source.device.id())?;
        self.copy_from_device_view_range(target_offset, source.view(), source_offset, len, stream)
    }

    /// Copies a range from a borrowed allocation view on the same device.
    pub(crate) fn copy_from_device_view_range(
        &mut self,
        target_offset: usize,
        source: DeviceView<'_, T>,
        source_offset: usize,
        len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        ensure_same_device(self.device.id(), source.device_id())?;
        ensure_same_device(self.device.id(), stream.device_id())?;
        let target = self.view().slice(target_offset, len)?;
        let source = source.slice(source_offset, len)?;
        let _context = self.device.make_current()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(Error::IntegerOverflow {
                field: "device_copy_bytes",
                value: len,
                target: "usize",
            })?;
        // SAFETY: both validated views cover `bytes`, belong to the current
        // context, and synchronization completes the transfer before return.
        unsafe {
            check(
                (self.device.api.driver.cuMemcpyDtoDAsync_v2)(
                    target.as_raw(),
                    source.as_raw(),
                    bytes,
                    stream.raw,
                ),
                "cuMemcpyDtoDAsync",
            )?;
        }
        stream.synchronize()
    }

    /// Zero-fills a range of this allocation on `stream`.
    pub fn memset_zero_range(
        &mut self,
        offset: usize,
        len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        ensure_same_device(self.device.id(), stream.device_id())?;
        let target = self.view().slice(offset, len)?;
        let _context = self.device.make_current()?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or(Error::IntegerOverflow {
                field: "device_memset_bytes",
                value: len,
                target: "usize",
            })?;
        // SAFETY: the validated view covers `bytes`; synchronization completes
        // the write before this method returns.
        unsafe {
            check(
                (self.device.api.driver.cuMemsetD8Async)(target.as_raw(), 0, bytes, stream.raw),
                "cuMemsetD8Async",
            )?;
        }
        stream.synchronize()
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        let _context = self.device.make_current();
        // SAFETY: this allocation is uniquely owned and the retained context
        // is synchronized before its pointer is freed.
        unsafe {
            let _ = (self.device.api.driver.cuCtxSynchronize)();
            let _ = (self.device.api.driver.cuMemFree_v2)(self.pointer);
        }
    }
}

impl<'a, T> DeviceView<'a, T> {
    /// Copies this view to host memory and synchronizes before returning.
    ///
    /// This is primarily useful for diagnostics and reference capture inside
    /// a device-result callback. Production consumers should normally enqueue
    /// their dependent GPU work directly on the callback stream.
    pub fn copy_to(self, destination: &mut [T], stream: &CudaStream) -> Result<()> {
        ensure_same_device(self.device_id(), stream.device_id())?;
        if destination.len() != self.len() {
            return Err(Error::InvalidSchema(format!(
                "copy length {} does not match view length {}",
                destination.len(),
                self.len()
            )));
        }
        let _context = stream.device.make_current()?;
        // SAFETY: the borrowed view covers the validated byte count and its
        // owner remains live for this call. Synchronization keeps destination
        // unavailable until the transfer completes.
        unsafe {
            check(
                (stream.device.api.driver.cuMemcpyDtoHAsync_v2)(
                    destination.as_mut_ptr().cast(),
                    self.as_raw(),
                    std::mem::size_of_val(destination),
                    stream.raw,
                ),
                "cuMemcpyDtoHAsync",
            )?;
        }
        stream.synchronize()
    }
}

impl GpuDevice {
    pub fn new_with_context(
        ordinal: i32,
        context: crate::Audio2Face3DContext,
    ) -> Result<Arc<Self>> {
        crate::logging::integration::LogScope::new(context).in_scope(|| Self::new(ordinal))
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
    fn native_entry_points_restore_the_callers_current_context() {
        let device = GpuDevice::new(0).unwrap();
        let mut original = ptr::null_mut();
        // SAFETY: CUDA writes the calling thread's current context to a valid
        // output pointer. The same handle is restored before this test exits.
        unsafe {
            check(
                (device.api.driver.cuCtxGetCurrent)(&mut original),
                "cuCtxGetCurrent",
            )
            .unwrap()
        };

        // SAFETY: a null handle clears only this calling thread's current CUDA
        // context; `original` remains retained by its owner or the driver.
        unsafe {
            check(
                (device.api.driver.cuCtxSetCurrent)(ptr::null_mut()),
                "cuCtxSetCurrent",
            )
            .unwrap()
        };
        let stream = device.create_stream().unwrap();
        let mut after_create = original;
        // SAFETY: CUDA writes one context handle to the valid output pointer.
        unsafe {
            check(
                (device.api.driver.cuCtxGetCurrent)(&mut after_create),
                "cuCtxGetCurrent",
            )
            .unwrap()
        };
        assert!(after_create.is_null());

        drop(stream);
        let mut after_drop = original;
        // SAFETY: CUDA writes one context handle to the valid output pointer.
        unsafe {
            check(
                (device.api.driver.cuCtxGetCurrent)(&mut after_drop),
                "cuCtxGetCurrent",
            )
            .unwrap()
        };
        assert!(after_drop.is_null());

        // SAFETY: restore the context observed at test entry for subsequent
        // tests that may execute on this worker thread.
        unsafe {
            check(
                (device.api.driver.cuCtxSetCurrent)(original),
                "cuCtxSetCurrent",
            )
            .unwrap()
        };
    }

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
