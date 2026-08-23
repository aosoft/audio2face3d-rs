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

pub struct CublasHandle {
    device: Rc<GpuDevice>,
    raw: cudarc::cublas::sys::cublasHandle_t,
    _stream: PhantomData<*const CudaStream>,
    _not_send_sync: PhantomData<*mut ()>,
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
        let raw = cudarc::curand::result::create_generator()
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
        ensure_same_device(self.device.id(), stream.device_id())?;
        if source.len() != self.len {
            return Err(Audio2xError::InvalidSchema(format!(
                "copy length {} does not match allocation length {}",
                source.len(),
                self.len
            )));
        }
        self.device.make_current()?;
        // SAFETY: source and destination cover the validated byte count; synchronization
        // before return keeps the host slice alive for the complete asynchronous copy.
        unsafe {
            check(
                cuMemcpyHtoDAsync_v2(
                    self.pointer,
                    source.as_ptr().cast(),
                    std::mem::size_of_val(source),
                    stream.raw,
                ),
                "cuMemcpyHtoDAsync",
            )?;
        }
        stream.synchronize()
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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
