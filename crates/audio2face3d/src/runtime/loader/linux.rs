//! Experimental Linux loader; the selected SDK must have resolvable ELF dependencies.
use super::*;
use std::{
    ffi::CString,
    os::unix::{ffi::OsStrExt, fs::MetadataExt},
};
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileIdentity(u64, u64);
pub(crate) fn identity(path: &Path) -> Result<FileIdentity, NativeRuntimeError> {
    let metadata = std::fs::metadata(path).map_err(|e| {
        NativeRuntimeError::new(NativeRuntimeErrorKind::LibraryNotFound, e.to_string())
            .with_path(path)
    })?;
    Ok(FileIdentity(metadata.dev(), metadata.ino()))
}
pub(super) unsafe fn open(
    file: &LibraryFile,
    _directories: &[PathBuf],
    attempt: &mut LoadAttempt,
) -> Result<&'static libloading::Library, NativeRuntimeError> {
    #[cfg(target_os = "linux")]
    check_existing(file)?;
    let name = CString::new(file.path.as_os_str().as_bytes()).map_err(|e| {
        NativeRuntimeError::new(NativeRuntimeErrorKind::InvalidConfig, e.to_string())
    })?;
    attempt.begin();
    // SAFETY: caller authorizes loading this absolute executable path. RTLD_NOW detects missing dependencies immediately.
    let library = unsafe {
        libloading::os::unix::Library::open(
            Some(std::ffi::OsStr::from_bytes(name.as_bytes())),
            libc::RTLD_NOW | libc::RTLD_LOCAL,
        )
    }
    .map_err(|e| {
        NativeRuntimeError::new(NativeRuntimeErrorKind::DependencyLoadFailed, e.to_string())
            .with_path(&file.path)
    })?;
    Ok(Box::leak(Box::new(library.into())))
}
#[cfg(target_os = "linux")]
fn check_existing(file: &LibraryFile) -> Result<(), NativeRuntimeError> {
    struct Scan<'a> {
        expected: &'a LibraryFile,
        conflict: Option<PathBuf>,
    }
    unsafe extern "C" fn visit(
        info: *mut libc::dl_phdr_info,
        _size: usize,
        data: *mut std::ffi::c_void,
    ) -> i32 {
        // SAFETY: dl_iterate_phdr invokes this callback synchronously with our Scan pointer.
        let scan = unsafe { &mut *data.cast::<Scan<'_>>() };
        // SAFETY: info and its NUL-terminated name are supplied by the dynamic loader.
        let name = unsafe { std::ffi::CStr::from_ptr((*info).dlpi_name) };
        let path = Path::new(std::ffi::OsStr::from_bytes(name.to_bytes()));
        if path.file_name() == scan.expected.path.file_name()
            && identity(path).ok().as_ref() != Some(&scan.expected.identity)
        {
            scan.conflict = Some(path.to_owned());
            return 1;
        }
        0
    }
    let mut scan = Scan {
        expected: file,
        conflict: None,
    };
    // SAFETY: scan lives through this synchronous traversal and visit uses its exact type.
    unsafe {
        libc::dl_iterate_phdr(Some(visit), (&mut scan as *mut Scan<'_>).cast());
    }
    if let Some(path) = scan.conflict {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::RuntimeConflict,
            "another file with this module name is already loaded",
        )
        .with_path(path)
        .with_path(&file.path)
        .after_load());
    }
    Ok(())
}
