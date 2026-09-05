//! Portable borrowed CUDA contracts.
//!
//! These types describe callback-scoped device data without requiring the
//! CUDA toolkit or the `cuda` feature. They do not own the referenced device
//! allocation or stream, and safe code cannot construct either type from a raw
//! handle. The lifetime is supplied by the owning SDK resource when a view is
//! created or when a result callback is invoked.
//!
//! # Thread-safety
//!
//! Borrowed contracts follow the audited D-04 matrix:
//!
//! - [`DeviceView`] is `Send` when `T: Send` and `Sync` when `T: Sync`.
//! - [`CudaStreamRef`] is `Send + Sync` while its callback lifetime is live.
//!
//! The lifetime remains callback-scoped: neither type owns or destroys the
//! native resource, so safe code cannot retain a dangling handle.

use crate::common::{Error, Result};
use crate::cuda::DeviceId;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::mem::size_of;

/// A callback-scoped, read-only view of a CUDA device allocation.
///
/// This value does not own the allocation. Its lifetime is tied to the SDK
/// owner that produced it, so it cannot be retained after that owner (or the
/// callback borrow) ends. There is intentionally no public raw-parts
/// constructor. Use an owning `DeviceBuffer<T>` when the `cuda` feature is
/// enabled, or receive a view through a result callback.
///
/// Corresponds to `nva2x::DeviceTensorFloatConstView` in
/// `audio2x-common/include/audio2x/tensor_float.h`.
///
/// This type is `Send` when `T: Send` and `Sync` when `T: Sync`.
#[derive(Debug, Clone, Copy)]
pub struct DeviceView<'a, T> {
    pointer: u64,
    len: usize,
    device: DeviceId,
    _owner: PhantomData<&'a T>,
}

impl<'a, T> DeviceView<'a, T> {
    /// Creates a borrowed view from SDK-validated raw parts.
    ///
    /// # Safety
    ///
    /// `pointer` must name storage for `len` consecutive `T` values on `device`,
    /// and that allocation must remain valid for `'a`. This constructor does
    /// not assert that the allocation has been initialized; operations that
    /// read through the view must separately guarantee initialization and
    /// CUDA ordering. Only crate internals that retain the corresponding owner
    /// may call this.
    pub(crate) const unsafe fn from_raw_parts(pointer: u64, len: usize, device: DeviceId) -> Self {
        Self {
            pointer,
            len,
            device,
            _owner: PhantomData,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Returns the borrowed CUDA device pointer for FFI interoperability.
    ///
    /// The pointer remains owned by the producer of this view. It must not be
    /// freed, and asynchronous work using it must complete before `'a` ends.
    pub const fn as_raw(&self) -> u64 {
        self.pointer
    }

    pub fn slice(self, offset: usize, len: usize) -> Result<DeviceView<'a, T>> {
        let end = offset.checked_add(len).ok_or(Error::IntegerOverflow {
            field: "device_view_end",
            value: len,
            target: "usize",
        })?;
        if end > self.len {
            return Err(Error::InvalidSchema(format!(
                "device view range {offset}..{end} exceeds length {}",
                self.len
            )));
        }
        let byte_offset = offset
            .checked_mul(size_of::<T>())
            .ok_or(Error::IntegerOverflow {
                field: "device_view_byte_offset",
                value: offset,
                target: "usize",
            })?;
        let pointer = self
            .pointer
            .checked_add(byte_offset as u64)
            .ok_or_else(|| Error::InvalidSchema("device view pointer overflow".into()))?;
        // SAFETY: this is a checked subrange of the original borrowed view, so
        // it inherits the storage extent, device, and lifetime. Initialization
        // requirements remain the responsibility of an eventual read.
        Ok(unsafe { DeviceView::from_raw_parts(pointer, len, self.device) })
    }
}

// SAFETY: the descriptor does not own or free the allocation. Its lifetime is
// tied to the owner borrow, and raw-pointer use is restricted to explicit CUDA
// operations. Moving/sharing it therefore has the same bounds as `T`.
unsafe impl<T: Send> Send for DeviceView<'_, T> {}
// SAFETY: sharing the read-only descriptor is valid when the referenced
// element type itself can be shared between threads.
unsafe impl<T: Sync> Sync for DeviceView<'_, T> {}

/// A callback-scoped reference to a CUDA stream.
///
/// The stream remains owned by the SDK. The raw handle may be used to enqueue
/// dependent work during the callback, but it must not be destroyed or stored
/// beyond `'a`. There is intentionally no public raw-handle constructor.
///
/// Corresponds to `nva2x::ICudaStream` in
/// `audio2x-common/include/audio2x/cuda_stream.h`.
///
/// This type is `Send + Sync` while the owning stream remains alive.
#[derive(Debug, Clone, Copy)]
pub struct CudaStreamRef<'a> {
    raw: *mut c_void,
    device: DeviceId,
    _stream: PhantomData<&'a ()>,
}

impl<'a> CudaStreamRef<'a> {
    /// Creates a borrowed stream reference from an SDK-owned stream.
    ///
    /// # Safety
    ///
    /// `raw` must remain a valid CUDA stream on `device` for `'a`. Only crate
    /// internals that retain the corresponding stream owner may call this.
    #[cfg(feature = "cuda")]
    pub(crate) const unsafe fn from_raw(raw: *mut c_void, device: DeviceId) -> Self {
        Self {
            raw,
            device,
            _stream: PhantomData,
        }
    }

    pub const fn device_id(&self) -> DeviceId {
        self.device
    }

    /// Returns the borrowed native CUDA stream handle for FFI interoperability.
    ///
    /// The handle must not be destroyed and must not be used after `'a` ends.
    pub const fn as_raw(&self) -> *mut c_void {
        self.raw
    }
}

// SAFETY: this is a non-owning descriptor. The lifetime prevents use after
// the owning stream is dropped; the owner establishes the current CUDA context
// before every operation and serializes destruction by Rust ownership.
unsafe impl Send for CudaStreamRef<'_> {}
// SAFETY: concurrent copies of this borrowed descriptor do not own or destroy
// the stream; operations remain ordered by the CUDA stream itself.
unsafe impl Sync for CudaStreamRef<'_> {}
