use crate::{InferenceError, ffi};
use audio2x_core::{Binding, BindingSchema, Dimension, ElementType, IoMode, Shape};
use audio2x_cuda::{CudaEvent, CudaStream, DeviceId, DeviceView, GpuDevice};
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_char, c_void};
use std::marker::PhantomData;
use std::mem::size_of;
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;

const ERROR_CAPACITY: usize = 4096;
fn native_error(operation: &'static str, buffer: &[c_char]) -> InferenceError {
    // SAFETY: shim error buffers are initialized to zero and always NUL-terminated.
    let message = unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    InferenceError::Native { operation, message }
}

#[derive(Debug, Clone, Copy)]
pub struct BindingBuffer<'a> {
    ptr: u64,
    bytes: usize,
    device: DeviceId,
    element_type: ElementType,
    _owner: PhantomData<&'a ()>,
}
impl<'a> BindingBuffer<'a> {
    pub fn from_view<T>(view: DeviceView<'a, T>, element_type: ElementType) -> Self {
        Self {
            ptr: view.as_raw(),
            bytes: view.len().saturating_mul(size_of::<T>()),
            device: view.device_id(),
            element_type,
            _owner: PhantomData,
        }
    }
}

#[derive(Debug, Default)]
pub struct DeviceBindings<'a> {
    values: HashMap<String, BindingBuffer<'a>>,
    shapes: HashMap<String, Vec<i64>>,
}
impl<'a> DeviceBindings<'a> {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(
        &mut self,
        name: impl Into<String>,
        buffer: BindingBuffer<'a>,
    ) -> Result<(), InferenceError> {
        let name = name.into();
        if self.values.insert(name.clone(), buffer).is_some() {
            return Err(InferenceError::DuplicateBinding(name));
        }
        Ok(())
    }
    pub fn set_input_shape(
        &mut self,
        name: impl Into<String>,
        shape: Vec<i64>,
    ) -> Result<(), InferenceError> {
        if shape.iter().any(|v| *v <= 0) {
            return Err(InferenceError::InvalidBinding(
                "shape dimensions must be positive".into(),
            ));
        }
        self.shapes.insert(name.into(), shape);
        Ok(())
    }
}

pub struct TensorRtSession {
    handle: NonNull<ffi::TrtSessionHandle>,
    device: Rc<GpuDevice>,
    schema: BindingSchema,
    profile_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeTensorShape {
    pub name: String,
    pub dimensions: Vec<usize>,
}

impl TensorRtSession {
    pub fn load(device: Rc<GpuDevice>, engine: &Path) -> Result<Self, InferenceError> {
        let path = CString::new(engine.to_string_lossy().as_bytes())
            .map_err(|_| InferenceError::InvalidBinding("engine path contains NUL".into()))?;
        let mut error = [0; ERROR_CAPACITY];
        // SAFETY: path and output buffer remain live for the call.
        let raw = unsafe {
            ffi::trt_shim_create(
                path.as_ptr(),
                device.id().ordinal(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let handle = NonNull::new(raw).ok_or_else(|| native_error("create", &error))?;
        match read_metadata(handle) {
            Ok((schema, profile_count)) => Ok(Self {
                handle,
                device,
                schema,
                profile_count,
            }),
            Err(cause) => {
                // SAFETY: handle was returned by create and has not been destroyed.
                unsafe { ffi::trt_shim_destroy(handle.as_ptr()) };
                Err(cause)
            }
        }
    }
    pub fn metadata(&self) -> &BindingSchema {
        &self.schema
    }
    pub const fn profile_count(&self) -> usize {
        self.profile_count
    }

    /// Resolves every input and output dimension for one optimization profile.
    ///
    /// This mutates the execution context and therefore requires exclusive
    /// access to the session. No host/device tensor data is transferred.
    pub fn resolve_shapes(
        &mut self,
        profile: usize,
        bindings: &DeviceBindings<'_>,
        stream: &CudaStream,
    ) -> Result<Vec<RuntimeTensorShape>, InferenceError> {
        self.validate_profile_stream(profile, stream)?;
        // SAFETY: session and stream are live and belong to the validated device.
        call("set_profile", |e, n| unsafe {
            ffi::trt_shim_set_profile(self.handle.as_ptr(), profile as i32, stream.as_raw(), e, n)
        })?;
        self.apply_input_shapes(bindings)?;
        // SAFETY: every dynamic input shape was applied immediately above.
        call("infer_shapes", |e, n| unsafe {
            ffi::trt_shim_infer_shapes(self.handle.as_ptr(), e, n)
        })?;
        self.schema
            .bindings()
            .iter()
            .enumerate()
            .map(|(index, binding)| {
                let raw = context_dims(
                    self.handle,
                    i32::try_from(index).map_err(|_| {
                        InferenceError::InvalidBinding("tensor index exceeds i32".into())
                    })?,
                )?;
                let dimensions = raw
                    .into_iter()
                    .map(|value| {
                        usize::try_from(value).map_err(|_| {
                            InferenceError::InvalidBinding(format!(
                                "unresolved dimension for {}",
                                binding.name
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(RuntimeTensorShape {
                    name: binding.name.clone(),
                    dimensions,
                })
            })
            .collect()
    }

    fn validate_profile_stream(
        &self,
        profile: usize,
        stream: &CudaStream,
    ) -> Result<(), InferenceError> {
        if stream.device_id() != self.device.id() {
            return Err(InferenceError::DeviceMismatch {
                name: "stream".into(),
                expected: self.device.id().ordinal(),
                actual: stream.device_id().ordinal(),
            });
        }
        if profile >= self.profile_count {
            return Err(InferenceError::InvalidBinding(format!(
                "profile {profile} is out of range"
            )));
        }
        Ok(())
    }

    fn apply_input_shapes(&mut self, bindings: &DeviceBindings<'_>) -> Result<(), InferenceError> {
        for binding in self.schema.bindings().iter().filter(|binding| {
            binding.mode == IoMode::Input
                && binding
                    .shape
                    .dimensions()
                    .iter()
                    .any(|d| matches!(d, Dimension::Dynamic { .. }))
        }) {
            let shape = bindings
                .shapes
                .get(&binding.name)
                .ok_or_else(|| InferenceError::MissingShape(binding.name.clone()))?;
            let rank = i32::try_from(shape.len())
                .map_err(|_| InferenceError::InvalidBinding("shape rank exceeds i32".into()))?;
            let name = CString::new(binding.name.as_str())
                .map_err(|_| InferenceError::InvalidBinding(binding.name.clone()))?;
            // SAFETY: name and shape storage remain valid during the call.
            call("set_input_shape", |e, n| unsafe {
                ffi::trt_shim_set_input_shape(
                    self.handle.as_ptr(),
                    name.as_ptr(),
                    shape.as_ptr(),
                    rank,
                    e,
                    n,
                )
            })?;
        }
        Ok(())
    }

    /// Queues inference without a host synchronization. The returned fence
    /// keeps the mutable session, stream, and every binding buffer borrowed.
    pub fn enqueue<'s, 'b>(
        &'s mut self,
        profile: usize,
        bindings: &'b DeviceBindings<'b>,
        stream: &'b CudaStream,
    ) -> Result<InferenceFence<'s, 'b>, InferenceError> {
        self.validate_profile_stream(profile, stream)?;
        for name in bindings.values.keys() {
            if self.schema.get(name).is_none() {
                return Err(InferenceError::InvalidBinding(format!(
                    "unknown binding {name}"
                )));
            }
        }
        // SAFETY: session and stream are live and belong to the validated device.
        call("set_profile", |e, n| unsafe {
            ffi::trt_shim_set_profile(self.handle.as_ptr(), profile as i32, stream.as_raw(), e, n)
        })?;
        self.apply_input_shapes(bindings)?;
        for binding in self.schema.bindings() {
            let buffer = bindings
                .values
                .get(&binding.name)
                .ok_or_else(|| InferenceError::MissingBinding(binding.name.clone()))?;
            if buffer.device != self.device.id() {
                return Err(InferenceError::DeviceMismatch {
                    name: binding.name.clone(),
                    expected: self.device.id().ordinal(),
                    actual: buffer.device.ordinal(),
                });
            }
            if buffer.element_type != binding.element_type {
                return Err(InferenceError::InvalidBinding(format!(
                    "dtype mismatch for {}",
                    binding.name
                )));
            }
            let name = CString::new(binding.name.as_str())
                .map_err(|_| InferenceError::InvalidBinding(binding.name.clone()))?;
            let width = binding.element_type.byte_width().ok_or_else(|| {
                InferenceError::InvalidBinding(format!("unsupported dtype for {}", binding.name))
            })?;
            if buffer.bytes % width != 0 {
                return Err(InferenceError::InvalidBinding(format!(
                    "unaligned buffer for {}",
                    binding.name
                )));
            }
            // SAFETY: the buffer borrow is held by the returned fence.
            call("set_tensor_address", |e, n| unsafe {
                ffi::trt_shim_set_tensor_address(
                    self.handle.as_ptr(),
                    name.as_ptr(),
                    buffer.ptr as usize as *mut c_void,
                    e,
                    n,
                )
            })?;
        }
        // SAFETY: all required addresses are bound and the stream is live.
        call("enqueue", |e, n| unsafe {
            ffi::trt_shim_enqueue(self.handle.as_ptr(), stream.as_raw(), e, n)
        })?;
        let event = stream.create_event().map_err(InferenceError::Cuda)?;
        event.record(stream).map_err(InferenceError::Cuda)?;
        Ok(InferenceFence {
            event,
            _session: PhantomData,
            _bindings: PhantomData,
        })
    }
}
impl Drop for TensorRtSession {
    fn drop(&mut self) {
        // SAFETY: this object exclusively owns the live native handle.
        unsafe { ffi::trt_shim_destroy(self.handle.as_ptr()) }
    }
}
pub struct InferenceFence<'s, 'b> {
    event: CudaEvent,
    _session: PhantomData<&'s mut TensorRtSession>,
    _bindings: PhantomData<&'b DeviceBindings<'b>>,
}
impl InferenceFence<'_, '_> {
    pub fn synchronize(&self) -> Result<(), InferenceError> {
        self.event.synchronize().map_err(InferenceError::Cuda)
    }
}

fn call(
    operation: &'static str,
    f: impl FnOnce(*mut c_char, usize) -> i32,
) -> Result<(), InferenceError> {
    let mut error = [0; ERROR_CAPACITY];
    if f(error.as_mut_ptr(), error.len()) == 1 {
        Ok(())
    } else {
        Err(native_error(operation, &error))
    }
}
fn tensor_name(
    handle: NonNull<ffi::TrtSessionHandle>,
    index: i32,
) -> Result<String, InferenceError> {
    let mut required = 0;
    let mut error = [0; ERROR_CAPACITY];
    // SAFETY: handle and all output pointers are valid for the call.
    let ok = unsafe {
        ffi::trt_shim_tensor_name(
            handle.as_ptr(),
            index,
            std::ptr::null_mut(),
            0,
            &mut required,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if ok != 1 || required == 0 {
        return Err(native_error("tensor_name_size", &error));
    }
    let mut name = vec![0; required];
    // SAFETY: the allocated name buffer has the queried capacity.
    let ok = unsafe {
        ffi::trt_shim_tensor_name(
            handle.as_ptr(),
            index,
            name.as_mut_ptr(),
            name.len(),
            &mut required,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if ok != 1 {
        return Err(native_error("tensor_name", &error));
    }
    // SAFETY: a successful shim call wrote a NUL-terminated string.
    Ok(unsafe { CStr::from_ptr(name.as_ptr()) }
        .to_string_lossy()
        .into_owned())
}
fn dims_call(
    handle: NonNull<ffi::TrtSessionHandle>,
    index: i32,
    profile: Option<(i32, i32)>,
    info: *mut ffi::TensorInfo,
) -> Result<Vec<i64>, InferenceError> {
    let mut rank = 0;
    let mut error = [0; ERROR_CAPACITY];
    // SAFETY: handle is live; each supplied output buffer matches its capacity.
    let mut first = |dims, cap, rank| unsafe {
        match profile {
            Some((p, s)) => ffi::trt_shim_tensor_profile_dims(
                handle.as_ptr(),
                index,
                p,
                s,
                dims,
                cap,
                rank,
                error.as_mut_ptr(),
                error.len(),
            ),
            None => ffi::trt_shim_tensor_info_at(
                handle.as_ptr(),
                index,
                info,
                dims,
                cap,
                rank,
                error.as_mut_ptr(),
                error.len(),
            ),
        }
    };
    if first(std::ptr::null_mut(), 0, &mut rank) != 1 || rank < 0 {
        return Err(native_error("tensor_rank", &error));
    }
    let mut dims = vec![0; rank as usize];
    if first(dims.as_mut_ptr(), rank, &mut rank) != 1 {
        return Err(native_error("tensor_dimensions", &error));
    }
    Ok(dims)
}

fn context_dims(
    handle: NonNull<ffi::TrtSessionHandle>,
    index: i32,
) -> Result<Vec<i64>, InferenceError> {
    let mut rank = 0;
    let mut error = [0; ERROR_CAPACITY];
    // SAFETY: handle is live and the first call only queries required rank.
    let ok = unsafe {
        ffi::trt_shim_context_tensor_dims(
            handle.as_ptr(),
            index,
            std::ptr::null_mut(),
            0,
            &mut rank,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if ok != 1 || rank < 0 {
        return Err(native_error("context_tensor_rank", &error));
    }
    let mut dimensions = vec![0; rank as usize];
    // SAFETY: dimensions has exactly the capacity reported by the shim.
    let ok = unsafe {
        ffi::trt_shim_context_tensor_dims(
            handle.as_ptr(),
            index,
            dimensions.as_mut_ptr(),
            rank,
            &mut rank,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if ok != 1 {
        return Err(native_error("context_tensor_dimensions", &error));
    }
    Ok(dimensions)
}
fn read_metadata(
    handle: NonNull<ffi::TrtSessionHandle>,
) -> Result<(BindingSchema, usize), InferenceError> {
    // SAFETY: handle remains live throughout metadata collection.
    let count = unsafe { ffi::trt_shim_tensor_count(handle.as_ptr()) };
    // SAFETY: same live handle as above; this call does not mutate it.
    let profiles = unsafe { ffi::trt_shim_profile_count(handle.as_ptr()) };
    if count < 0 || profiles < 0 {
        return Err(InferenceError::Native {
            operation: "metadata",
            message: "invalid TensorRT handle".into(),
        });
    }
    let mut bindings = Vec::new();
    for index in 0..count {
        let name = tensor_name(handle, index)?;
        let mut info = ffi::TensorInfo::default();
        let raw = dims_call(handle, index, None, &mut info)?;
        let min = if info.io_mode == 1 && raw.iter().any(|v| *v < 0) {
            Some(dims_call(
                handle,
                index,
                Some((0, 0)),
                std::ptr::null_mut(),
            )?)
        } else {
            None
        };
        let max = if min.is_some() {
            Some(dims_call(
                handle,
                index,
                Some((0, 2)),
                std::ptr::null_mut(),
            )?)
        } else {
            None
        };
        let dimensions = raw
            .iter()
            .enumerate()
            .map(|(axis, value)| {
                if *value > 0 {
                    Ok(Dimension::Fixed(*value as usize))
                } else {
                    let low = min.as_ref().and_then(|v| v.get(axis)).copied().unwrap_or(1);
                    let high = max.as_ref().and_then(|v| v.get(axis)).copied().unwrap_or(1);
                    if low <= 0 || high < low {
                        Err(InferenceError::InvalidBinding(format!(
                            "invalid profile for {name}"
                        )))
                    } else {
                        Ok(Dimension::Dynamic {
                            min: low as usize,
                            max: high as usize,
                        })
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let element_type = match info.data_type {
            0 => ElementType::F32,
            1 => ElementType::F16,
            4 => ElementType::I64,
            5 => ElementType::Bool,
            _ => ElementType::Raw,
        };
        bindings.push(Binding {
            name,
            mode: if info.io_mode == 1 {
                IoMode::Input
            } else {
                IoMode::Output
            },
            element_type,
            shape: Shape::new(dimensions).map_err(InferenceError::Cuda)?,
        });
    }
    Ok((
        BindingSchema::new(bindings).map_err(InferenceError::Cuda)?,
        profiles as usize,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_real_engine_metadata_when_configured() {
        let Some(path) = std::env::var_os("AUDIO2X_TEST_ENGINE") else {
            eprintln!("skipped: AUDIO2X_TEST_ENGINE is not configured");
            return;
        };
        let device = GpuDevice::new(0).unwrap();
        let session = TensorRtSession::load(device, Path::new(&path)).unwrap();
        assert!(!session.metadata().bindings().is_empty());
        assert!(session.profile_count() > 0);
    }

    #[test]
    fn runs_real_engine_on_rust_stream_when_configured() {
        let Some(path) = std::env::var_os("AUDIO2X_TEST_ENGINE") else {
            eprintln!("skipped: AUDIO2X_TEST_ENGINE is not configured");
            return;
        };
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let mut session = TensorRtSession::load(Rc::clone(&device), Path::new(&path)).unwrap();
        let specs = session
            .metadata()
            .bindings()
            .iter()
            .map(|binding| {
                let shape = binding
                    .shape
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Fixed(value) => *value,
                        Dimension::Batch => 1,
                        Dimension::Dynamic { min, .. } => *min,
                    })
                    .collect::<Vec<_>>();
                let elements = shape.iter().product::<usize>();
                let bytes = elements * binding.element_type.byte_width().unwrap();
                (
                    binding.name.clone(),
                    binding.mode,
                    binding.element_type,
                    shape,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        let buffers = specs
            .iter()
            .map(|spec| device.allocate::<u8>(spec.4).unwrap())
            .collect::<Vec<_>>();
        let mut bindings = DeviceBindings::new();
        for (spec, buffer) in specs.iter().zip(&buffers) {
            bindings
                .insert(&spec.0, BindingBuffer::from_view(buffer.view(), spec.2))
                .unwrap();
            if spec.1 == IoMode::Input {
                bindings
                    .set_input_shape(&spec.0, spec.3.iter().map(|v| *v as i64).collect())
                    .unwrap();
            }
        }
        session
            .enqueue(0, &bindings, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
    }

    #[test]
    fn resolves_dynamic_output_shapes_when_configured() {
        let Some(path) = std::env::var_os("AUDIO2X_TEST_DYNAMIC_ENGINE") else {
            eprintln!("skipped: AUDIO2X_TEST_DYNAMIC_ENGINE is not configured");
            return;
        };
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let mut session = TensorRtSession::load(Rc::clone(&device), Path::new(&path)).unwrap();
        let mut bindings = DeviceBindings::new();
        for binding in session.metadata().bindings() {
            if binding.mode == IoMode::Input {
                let shape = binding
                    .shape
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Fixed(value) => *value as i64,
                        Dimension::Batch => 1,
                        Dimension::Dynamic { min, .. } => *min as i64,
                    })
                    .collect();
                bindings.set_input_shape(&binding.name, shape).unwrap();
            }
        }
        let resolved = session.resolve_shapes(0, &bindings, &stream).unwrap();
        assert_eq!(resolved.len(), session.metadata().bindings().len());
        assert!(
            resolved
                .iter()
                .all(|tensor| tensor.dimensions.iter().all(|value| *value > 0))
        );
        let specs = session
            .metadata()
            .bindings()
            .iter()
            .map(|binding| {
                let shape = resolved
                    .iter()
                    .find(|tensor| tensor.name == binding.name)
                    .unwrap();
                let bytes = shape.dimensions.iter().product::<usize>()
                    * binding.element_type.byte_width().unwrap();
                (binding.name.clone(), binding.element_type, bytes)
            })
            .collect::<Vec<_>>();
        let buffers = specs
            .iter()
            .map(|spec| device.allocate::<u8>(spec.2).unwrap())
            .collect::<Vec<_>>();
        for (spec, buffer) in specs.iter().zip(&buffers) {
            bindings
                .insert(&spec.0, BindingBuffer::from_view(buffer.view(), spec.1))
                .unwrap();
        }
        session
            .enqueue(0, &bindings, &stream)
            .unwrap()
            .synchronize()
            .unwrap();
    }
}
