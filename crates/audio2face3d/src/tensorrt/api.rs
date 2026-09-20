//! Process-wide TensorRT and CUDA Runtime initialization, before object creation.
use super::ffi;
use crate::{
    Audio2Face3DContext,
    cuda::api::CudaApi,
    logging::{LogLevel, Logger},
    runtime::{
        NativeLibraryInfo, NativeRuntimeError, NativeRuntimeErrorKind, NativeRuntimeInfo,
        NativeRuntimeState, NativeVersion,
        discovery::{self, Sdk},
        loader::{FileIdentity, LoadedLibrary},
        registry::Registry,
        version::verify,
    },
};
use std::{
    ffi::{CStr, c_char, c_void},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

pub(crate) struct NativeApi {
    pub(crate) info: NativeRuntimeInfo,
    directories: Vec<std::path::PathBuf>,
    _cuda: Arc<CudaApi>,
    _libraries: Vec<LoadedLibrary>,
}
static REGISTRY: Registry<Vec<FileIdentity>, NativeApi> = Registry::new();
struct Resolver<'a> {
    runtime: &'a LoadedLibrary,
    tensorrt: &'a LoadedLibrary,
    error: Option<NativeRuntimeError>,
}
unsafe extern "C" fn resolve(
    user: *mut c_void,
    component: i32,
    name: *const c_char,
) -> *mut c_void {
    // SAFETY: trt_shim_initialize invokes this synchronously with our live Resolver.
    let resolver = unsafe { &mut *user.cast::<Resolver<'_>>() };
    let result = catch_unwind(AssertUnwindSafe(|| {
        let library = match component {
            0 => resolver.runtime,
            1 => resolver.tensorrt,
            _ => {
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::AbiMismatch,
                    "unknown resolver component",
                ));
            }
        };
        // SAFETY: the shim provides a static NUL-terminated symbol name.
        let name = unsafe { CStr::from_ptr(name) };
        // SAFETY: the shim casts the export address to its exact header-declared function type.
        unsafe { library.symbol::<*mut c_void>(name.to_bytes_with_nul()) }
    }));
    match result {
        Ok(Ok(pointer)) => pointer,
        Ok(Err(error)) => {
            resolver.error = Some(error);
            std::ptr::null_mut()
        }
        Err(_) => {
            resolver.error = Some(NativeRuntimeError::new(
                NativeRuntimeErrorKind::InitializationFailed,
                "symbol resolver panicked",
            ));
            std::ptr::null_mut()
        }
    }
}
fn cuda_version(value: u32) -> NativeVersion {
    NativeVersion::new(value / 1000, (value % 1000) / 10, None, None)
}
impl NativeApi {
    pub(crate) fn validate(&self) -> Result<(), NativeRuntimeError> {
        crate::runtime::loader::validate_loaded(&self.directories, true)
    }
    pub(crate) fn initialize(
        context: &Audio2Face3DContext,
    ) -> Result<Arc<Self>, NativeRuntimeError> {
        if let Some(api) = context.cached_tensorrt() {
            return Ok(api);
        }
        if let Some(error) = REGISTRY.prior_failure() {
            return Err(error);
        }
        let cuda = CudaApi::initialize(context)?;
        let cuda_dirs = discovery::directories(context.native_runtime(), Sdk::Cuda)?;
        let trt_dirs = discovery::directories(context.native_runtime(), Sdk::TensorRt)?;
        #[cfg(windows)]
        let (runtime_name, trt_name, suffix) = ("cudart64_", "nvinfer_", ".dll");
        #[cfg(unix)]
        let (runtime_name, trt_name, suffix) = ("libcudart.so", "libnvinfer.so", "");
        let runtime = discovery::library(&cuda_dirs, runtime_name, suffix)?;
        let tensorrt = discovery::library(&trt_dirs, trt_name, suffix)?;
        let mut key = cuda
            .libraries
            .iter()
            .map(|library| library.file.identity.clone())
            .collect::<Vec<_>>();
        key.extend([runtime.identity.clone(), tensorrt.identity.clone()]);
        let api = REGISTRY.initialize(key, |attempt| {
            let dirs = cuda_dirs.into_iter().chain(trt_dirs).collect::<Vec<_>>();
            // SAFETY: selected SDK files are trusted and retained throughout process lifetime.
            let runtime = unsafe { LoadedLibrary::open(runtime, &dirs, attempt) }?;
            // SAFETY: selected SDK files are trusted and retained throughout process lifetime.
            let tensorrt = unsafe { LoadedLibrary::open(tensorrt, &dirs, attempt) }?;
            crate::runtime::loader::validate_loaded(&dirs, true)?;
            let mut resolver = Resolver {
                runtime: &runtime,
                tensorrt: &tensorrt,
                error: None,
            };
            let mut trt_version = 0;
            let mut runtime_version = 0;
            let mut error = [0 as c_char; 4096];
            // SAFETY: all pointers are live for this synchronous call. The callback never unwinds.
            let success = unsafe {
                ffi::trt_shim_initialize(
                    resolve,
                    (&mut resolver as *mut Resolver<'_>).cast(),
                    &mut trt_version,
                    &mut runtime_version,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            if let Some(error) = resolver.error {
                return Err(error);
            }
            if success == 0 || trt_version <= 0 || runtime_version <= 0 {
                // SAFETY: shim always NUL-terminates this initially zeroed error buffer.
                let message = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
                return Err(NativeRuntimeError::new(
                    NativeRuntimeErrorKind::InitializationFailed,
                    format!("native version initialization failed: {message}"),
                ));
            }
            let runtime_version = cuda_version(runtime_version as u32);
            let cuda_build = cuda_version(
                env!("AUDIO2FACE3D_BUILD_CUDA_VERSION")
                    .parse()
                    .expect("build CUDA version"),
            );
            let trt_version = NativeVersion::new(
                trt_version as u32 / 10000,
                (trt_version as u32 / 100) % 100,
                Some(trt_version as u32 % 100),
                None,
            );
            let trt_build = NativeVersion::new(
                env!("AUDIO2FACE3D_BUILD_TENSORRT_MAJOR")
                    .parse()
                    .expect("build TensorRT major"),
                env!("AUDIO2FACE3D_BUILD_TENSORRT_MINOR")
                    .parse()
                    .expect("build TensorRT minor"),
                Some(
                    env!("AUDIO2FACE3D_BUILD_TENSORRT_PATCH")
                        .parse()
                        .expect("build TensorRT patch"),
                ),
                Some(
                    env!("AUDIO2FACE3D_BUILD_TENSORRT_BUILD")
                        .parse()
                        .expect("build TensorRT build"),
                ),
            );
            verify(context, "CUDA Runtime", cuda_build, runtime_version)?;
            verify(context, "TensorRT", trt_build, trt_version)?;
            let mut libraries = cuda.info.libraries.clone();
            libraries.push(NativeLibraryInfo {
                name: "CUDA Runtime".into(),
                path: runtime.file.path.clone(),
                build_version: Some(cuda_build),
                runtime_version: Some(runtime_version),
            });
            libraries.push(NativeLibraryInfo {
                name: "TensorRT".into(),
                path: tensorrt.file.path.clone(),
                build_version: Some(trt_build),
                runtime_version: Some(trt_version),
            });
            let info = NativeRuntimeInfo {
                state: NativeRuntimeState::Loaded,
                libraries,
            };
            context.logger().log(LogLevel::Debug, || {
                format!("Native runtime initialized: {info:?}")
            });
            Ok(Self {
                info,
                directories: dirs,
                _cuda: cuda,
                _libraries: vec![runtime, tensorrt],
            })
        })?;
        context.retain_tensorrt(Arc::clone(&api));
        Ok(api)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    unsafe extern "C" fn missing(_: *mut c_void, _: i32, _: *const c_char) -> *mut c_void {
        std::ptr::null_mut()
    }
    #[test]
    fn shim_reports_missing_entry_point_without_publishing_api() {
        let (mut trt, mut cuda) = (0, 0);
        let mut error = [0 as c_char; 128];
        // SAFETY: synchronous callback, valid output buffers; missing callback never dereferences inputs.
        let result = unsafe {
            ffi::trt_shim_initialize(
                missing,
                std::ptr::null_mut(),
                &mut trt,
                &mut cuda,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        assert_eq!(result, 0);
        assert_eq!((trt, cuda), (0, 0));
        // SAFETY: shim NUL-terminates this buffer on failure.
        assert!(
            // SAFETY: shim NUL-terminates this buffer on failure.
            unsafe { CStr::from_ptr(error.as_ptr()) }
                .to_str()
                .unwrap()
                .contains("missing cudaSetDevice")
        );
    }
}

#[cfg(test)]
mod version_failure_tests {
    use super::*;
    unsafe extern "C" fn runtime_failure(_: *mut i32) -> i32 {
        999
    }
    unsafe extern "C" fn trt_version() -> i32 {
        101601
    }
    unsafe extern "C" fn unused() {}
    unsafe extern "C" fn resolver(user: *mut c_void, _: i32, name: *const c_char) -> *mut c_void {
        // SAFETY: shim supplies static NUL-terminated export names and a live bool.
        let name = unsafe { CStr::from_ptr(name) }.to_bytes();
        // SAFETY: test owns this boolean until synchronous initialization returns.
        let missing = unsafe { *user.cast::<bool>() };
        match name {
            b"cudaRuntimeGetVersion" if missing => std::ptr::null_mut(),
            b"cudaRuntimeGetVersion" => runtime_failure as *const () as *mut c_void,
            b"getInferLibVersion" => trt_version as *const () as *mut c_void,
            // These exports are checked for presence but never invoked on this failing path.
            _ => unused as *const () as *mut c_void,
        }
    }
    #[test]
    fn shim_version_symbol_and_api_failures_do_not_publish_function_table() {
        for mut missing in [true, false] {
            let (mut trt, mut cuda) = (0, 0);
            let mut error = [0 as c_char; 128];
            // SAFETY: live buffers and synchronous resolver; its only invoked exports have exact ABI.
            let result = unsafe {
                ffi::trt_shim_initialize(
                    resolver,
                    (&mut missing as *mut bool).cast(),
                    &mut trt,
                    &mut cuda,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            assert_eq!(result, 0);
            // SAFETY: shim NUL-terminates errors and never installs this failing table.
            let text = unsafe { CStr::from_ptr(error.as_ptr()) }.to_str().unwrap();
            assert_eq!(
                text,
                if missing {
                    "missing cudaRuntimeGetVersion"
                } else {
                    "cudaRuntimeGetVersion failed"
                }
            );
        }
    }
}
