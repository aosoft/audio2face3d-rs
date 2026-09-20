//! Owned CUDA API tables. cudarc supplies ABI types only; its loader is never invoked.
use crate::logging::{LogLevel, Logger};
use crate::{
    Audio2Face3DContext,
    runtime::{
        NativeRuntimeError, NativeRuntimeErrorKind,
        discovery::{self, Sdk},
        loader::{FileIdentity, LoadedLibrary},
        registry::Registry,
    },
};
use std::sync::{Arc, OnceLock};
const _: () = assert!(
    cudarc::driver::sys::CUDA_VERSION == 12090,
    "incompatible cudarc binding feature; audio2face3d requires cuda-12090"
);
mod driver {
    use super::*;
    use cudarc::driver::sys::*;
    #[derive(Debug)]
    #[allow(non_snake_case)]
    pub(crate) struct Driver {
        pub(crate) cuCtxGetCurrent: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
        pub(crate) cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
        pub(crate) cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
        pub(crate) cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, ::core::ffi::c_int) -> CUresult,
        pub(crate) cuDevicePrimaryCtxRelease_v2: unsafe extern "C" fn(CUdevice) -> CUresult,
        pub(crate) cuDevicePrimaryCtxRetain:
            unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
        pub(crate) cuEventCreate:
            unsafe extern "C" fn(*mut CUevent, ::core::ffi::c_uint) -> CUresult,
        pub(crate) cuEventDestroy_v2: unsafe extern "C" fn(CUevent) -> CUresult,
        pub(crate) cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
        pub(crate) cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
        pub(crate) cuInit: unsafe extern "C" fn(::core::ffi::c_uint) -> CUresult,
        pub(crate) cuLaunchKernel: unsafe extern "C" fn(
            CUfunction,
            ::core::ffi::c_uint,
            ::core::ffi::c_uint,
            ::core::ffi::c_uint,
            ::core::ffi::c_uint,
            ::core::ffi::c_uint,
            ::core::ffi::c_uint,
            ::core::ffi::c_uint,
            CUstream,
            *mut *mut ::core::ffi::c_void,
            *mut *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub(crate) cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
        pub(crate) cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
        pub(crate) cuMemcpyDtoDAsync_v2:
            unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
        pub(crate) cuMemcpyDtoHAsync_v2: unsafe extern "C" fn(
            *mut ::core::ffi::c_void,
            CUdeviceptr,
            usize,
            CUstream,
        ) -> CUresult,
        pub(crate) cuMemcpyHtoDAsync_v2: unsafe extern "C" fn(
            CUdeviceptr,
            *const ::core::ffi::c_void,
            usize,
            CUstream,
        ) -> CUresult,
        pub(crate) cuMemsetD8Async:
            unsafe extern "C" fn(CUdeviceptr, ::core::ffi::c_uchar, usize, CUstream) -> CUresult,
        pub(crate) cuModuleGetFunction:
            unsafe extern "C" fn(*mut CUfunction, CUmodule, *const ::core::ffi::c_char) -> CUresult,
        pub(crate) cuModuleLoadData:
            unsafe extern "C" fn(*mut CUmodule, *const ::core::ffi::c_void) -> CUresult,
        pub(crate) cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUresult,
        pub(crate) cuStreamCreate:
            unsafe extern "C" fn(*mut CUstream, ::core::ffi::c_uint) -> CUresult,
        pub(crate) cuStreamDestroy_v2: unsafe extern "C" fn(CUstream) -> CUresult,
        pub(crate) cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,
        pub(crate) cuStreamWaitEvent:
            unsafe extern "C" fn(CUstream, CUevent, ::core::ffi::c_uint) -> CUresult,
        pub(crate) cuDriverGetVersion: unsafe extern "C" fn(*mut ::core::ffi::c_int) -> CUresult,
    }
    impl Driver {
        pub(crate) unsafe fn load(library: &LoadedLibrary) -> Result<Self, NativeRuntimeError> {
            // SAFETY: signatures match the pinned CUDA bindings and modules remain loaded.
            unsafe {
                Ok(Self {
                    cuCtxGetCurrent: library.symbol(b"cuCtxGetCurrent\0")?,
                    cuCtxSetCurrent: library.symbol(b"cuCtxSetCurrent\0")?,
                    cuCtxSynchronize: library.symbol(b"cuCtxSynchronize\0")?,
                    cuDeviceGet: library.symbol(b"cuDeviceGet\0")?,
                    cuDevicePrimaryCtxRelease_v2: library
                        .symbol(b"cuDevicePrimaryCtxRelease_v2\0")?,
                    cuDevicePrimaryCtxRetain: library.symbol(b"cuDevicePrimaryCtxRetain\0")?,
                    cuEventCreate: library.symbol(b"cuEventCreate\0")?,
                    cuEventDestroy_v2: library.symbol(b"cuEventDestroy_v2\0")?,
                    cuEventRecord: library.symbol(b"cuEventRecord\0")?,
                    cuEventSynchronize: library.symbol(b"cuEventSynchronize\0")?,
                    cuInit: library.symbol(b"cuInit\0")?,
                    cuLaunchKernel: library.symbol(b"cuLaunchKernel\0")?,
                    cuMemAlloc_v2: library.symbol(b"cuMemAlloc_v2\0")?,
                    cuMemFree_v2: library.symbol(b"cuMemFree_v2\0")?,
                    cuMemcpyDtoDAsync_v2: library.symbol(b"cuMemcpyDtoDAsync_v2\0")?,
                    cuMemcpyDtoHAsync_v2: library.symbol(b"cuMemcpyDtoHAsync_v2\0")?,
                    cuMemcpyHtoDAsync_v2: library.symbol(b"cuMemcpyHtoDAsync_v2\0")?,
                    cuMemsetD8Async: library.symbol(b"cuMemsetD8Async\0")?,
                    cuModuleGetFunction: library.symbol(b"cuModuleGetFunction\0")?,
                    cuModuleLoadData: library.symbol(b"cuModuleLoadData\0")?,
                    cuModuleUnload: library.symbol(b"cuModuleUnload\0")?,
                    cuStreamCreate: library.symbol(b"cuStreamCreate\0")?,
                    cuStreamDestroy_v2: library.symbol(b"cuStreamDestroy_v2\0")?,
                    cuStreamSynchronize: library.symbol(b"cuStreamSynchronize\0")?,
                    cuStreamWaitEvent: library.symbol(b"cuStreamWaitEvent\0")?,
                    cuDriverGetVersion: library.symbol(b"cuDriverGetVersion\0")?,
                })
            }
        }
    }
}
mod cublas {
    use super::*;
    use cudarc::cublas::sys::*;
    #[derive(Debug)]
    #[allow(non_snake_case)]
    pub(crate) struct Cublas {
        pub(crate) cublasCreate_v2: unsafe extern "C" fn(*mut cublasHandle_t) -> cublasStatus_t,
        pub(crate) cublasDestroy_v2: unsafe extern "C" fn(cublasHandle_t) -> cublasStatus_t,
        pub(crate) cublasSetStream_v2:
            unsafe extern "C" fn(cublasHandle_t, cudaStream_t) -> cublasStatus_t,
        pub(crate) cublasSgemv_v2: unsafe extern "C" fn(
            cublasHandle_t,
            cublasOperation_t,
            ::core::ffi::c_int,
            ::core::ffi::c_int,
            *const f32,
            *const f32,
            ::core::ffi::c_int,
            *const f32,
            ::core::ffi::c_int,
            *const f32,
            *mut f32,
            ::core::ffi::c_int,
        ) -> cublasStatus_t,
        pub(crate) cublasSgemm_v2: unsafe extern "C" fn(
            cublasHandle_t,
            cublasOperation_t,
            cublasOperation_t,
            ::core::ffi::c_int,
            ::core::ffi::c_int,
            ::core::ffi::c_int,
            *const f32,
            *const f32,
            ::core::ffi::c_int,
            *const f32,
            ::core::ffi::c_int,
            *const f32,
            *mut f32,
            ::core::ffi::c_int,
        ) -> cublasStatus_t,
    }
    impl Cublas {
        pub(crate) unsafe fn load(library: &LoadedLibrary) -> Result<Self, NativeRuntimeError> {
            // SAFETY: signatures match the pinned CUDA bindings and modules remain loaded.
            unsafe {
                Ok(Self {
                    cublasCreate_v2: library.symbol(b"cublasCreate_v2\0")?,
                    cublasDestroy_v2: library.symbol(b"cublasDestroy_v2\0")?,
                    cublasSetStream_v2: library.symbol(b"cublasSetStream_v2\0")?,
                    cublasSgemv_v2: library.symbol(b"cublasSgemv_v2\0")?,
                    cublasSgemm_v2: library.symbol(b"cublasSgemm_v2\0")?,
                })
            }
        }
    }
}
mod curand {
    use super::*;
    use cudarc::curand::sys::*;
    #[derive(Debug)]
    #[allow(non_snake_case)]
    pub(crate) struct Curand {
        pub(crate) curandCreateGenerator:
            unsafe extern "C" fn(*mut curandGenerator_t, curandRngType_t) -> curandStatus_t,
        pub(crate) curandDestroyGenerator:
            unsafe extern "C" fn(curandGenerator_t) -> curandStatus_t,
        pub(crate) curandSetStream:
            unsafe extern "C" fn(curandGenerator_t, cudaStream_t) -> curandStatus_t,
        pub(crate) curandSetGeneratorOffset:
            unsafe extern "C" fn(curandGenerator_t, ::core::ffi::c_ulonglong) -> curandStatus_t,
        pub(crate) curandSetPseudoRandomGeneratorSeed:
            unsafe extern "C" fn(curandGenerator_t, ::core::ffi::c_ulonglong) -> curandStatus_t,
        pub(crate) curandGenerateNormal:
            unsafe extern "C" fn(curandGenerator_t, *mut f32, usize, f32, f32) -> curandStatus_t,
    }
    impl Curand {
        pub(crate) unsafe fn load(library: &LoadedLibrary) -> Result<Self, NativeRuntimeError> {
            // SAFETY: signatures match the pinned CUDA bindings and modules remain loaded.
            unsafe {
                Ok(Self {
                    curandCreateGenerator: library.symbol(b"curandCreateGenerator\0")?,
                    curandDestroyGenerator: library.symbol(b"curandDestroyGenerator\0")?,
                    curandSetStream: library.symbol(b"curandSetStream\0")?,
                    curandSetGeneratorOffset: library.symbol(b"curandSetGeneratorOffset\0")?,
                    curandSetPseudoRandomGeneratorSeed: library
                        .symbol(b"curandSetPseudoRandomGeneratorSeed\0")?,
                    curandGenerateNormal: library.symbol(b"curandGenerateNormal\0")?,
                })
            }
        }
    }
}
#[derive(Debug)]
pub(crate) struct CudaApi {
    pub(crate) info: crate::runtime::NativeRuntimeInfo,
    pub(crate) driver: driver::Driver,
    pub(crate) cublas: cublas::Cublas,
    pub(crate) curand: curand::Curand,
    pub(crate) libraries: Vec<LoadedLibrary>,
}
static REGISTRY: Registry<Vec<FileIdentity>, CudaApi> = Registry::new();
static READY: OnceLock<Arc<CudaApi>> = OnceLock::new();
impl CudaApi {
    pub(crate) fn loaded() -> Result<Arc<Self>, NativeRuntimeError> {
        READY.get().cloned().ok_or_else(|| {
            NativeRuntimeError::new(
                NativeRuntimeErrorKind::InitializationFailed,
                "borrowed CUDA operation requires an initialized runtime",
            )
        })
    }
    pub(crate) fn initialize(
        context: &Audio2Face3DContext,
    ) -> Result<Arc<Self>, NativeRuntimeError> {
        if let Some(api) = context.cached_cuda() {
            return Ok(api);
        }
        if let Some(error) = REGISTRY.prior_failure() {
            return Err(error);
        }
        let dirs = discovery::directories(context.native_runtime(), Sdk::Cuda)?;
        let driver = discovery::driver()?;
        #[cfg(windows)]
        let names = [
            ("cublasLt64_", ".dll"),
            ("cublas64_", ".dll"),
            ("curand64_", ".dll"),
        ];
        #[cfg(unix)]
        let names = [
            ("libcublasLt.so", ""),
            ("libcublas.so", ""),
            ("libcurand.so", ""),
        ];
        let mut files = vec![driver];
        for (prefix, suffix) in names {
            files.push(discovery::library(&dirs, prefix, suffix)?);
        }
        let key = files.iter().map(|f| f.identity.clone()).collect();
        let api = REGISTRY.initialize(key, |attempt| {
            let mut libraries = Vec::new();
            for file in files {
                // SAFETY: configuration explicitly selects these SDK binaries; each handle is retained.
                libraries.push(unsafe { LoadedLibrary::open(file, &dirs, attempt) }?);
            }
            crate::runtime::loader::validate_loaded(&dirs, false)?;
            // SAFETY: table signatures match pinned CUDA binding ABI; missing exports return errors.
            let mut api = unsafe {
                Self {
                    info: crate::runtime::NativeRuntimeInfo {
                        state: crate::runtime::NativeRuntimeState::Loaded,
                        libraries: Vec::new(),
                    },
                    driver: driver::Driver::load(&libraries[0])?,
                    cublas: cublas::Cublas::load(&libraries[2])?,
                    curand: curand::Curand::load(&libraries[3])?,
                    libraries,
                }
            };
            let mut version = 0;
            // SAFETY: the driver table is loaded and the version pointer is valid.
            unsafe {
                let status = (api.driver.cuInit)(0);
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(NativeRuntimeError::new(
                        NativeRuntimeErrorKind::DriverUnavailable,
                        format!("cuInit failed: {status:?}"),
                    ));
                }
                let status = (api.driver.cuDriverGetVersion)(&mut version);
                if status != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    return Err(NativeRuntimeError::new(
                        NativeRuntimeErrorKind::DriverUnavailable,
                        format!("cuDriverGetVersion failed: {status:?}"),
                    ));
                }
            }
            api.info.libraries = api
                .libraries
                .iter()
                .enumerate()
                .map(|(index, library)| crate::runtime::NativeLibraryInfo {
                    name: library
                        .file
                        .path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                    path: library.file.path.clone(),
                    build_version: None,
                    runtime_version: (index == 0).then(|| {
                        crate::runtime::NativeVersion::new(
                            version as u32 / 1000,
                            (version as u32 % 1000) / 10,
                            None,
                            None,
                        )
                    }),
                })
                .collect();
            context.logger().log(LogLevel::Debug, || {
                format!("CUDA Driver API version: {version}")
            });
            for library in &api.libraries {
                context.logger().log(LogLevel::Debug, || {
                    format!("Loaded native library: {}", library.file.path.display())
                });
            }
            Ok(api)
        })?;
        let _ = READY.set(Arc::clone(&api));
        context.retain_cuda(Arc::clone(&api));
        Ok(api)
    }
}
