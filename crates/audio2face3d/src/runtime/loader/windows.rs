use super::*;
use std::{
    ffi::OsString,
    fs::File,
    os::windows::{
        ffi::{OsStrExt, OsStringExt},
        io::AsRawHandle,
    },
};
use windows_sys::Win32::{
    Storage::FileSystem::{BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle},
    System::LibraryLoader::*,
};
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FileIdentity(u32, u32, u32);
pub(crate) fn identity(path: &Path) -> Result<FileIdentity, NativeRuntimeError> {
    let file = File::open(path).map_err(|e| failure(path, e))?;
    let mut info: BY_HANDLE_FILE_INFORMATION = Default::default();
    // SAFETY: file owns a valid handle and info is writable for this call.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
        return Err(failure(path, std::io::Error::last_os_error()));
    }
    Ok(FileIdentity(
        info.dwVolumeSerialNumber,
        info.nFileIndexHigh,
        info.nFileIndexLow,
    ))
}
fn failure(path: &Path, error: impl std::fmt::Display) -> NativeRuntimeError {
    NativeRuntimeError::new(
        NativeRuntimeErrorKind::DependencyLoadFailed,
        error.to_string(),
    )
    .with_path(path)
}
fn wide(value: &std::ffi::OsStr) -> Vec<u16> {
    value.encode_wide().chain(Some(0)).collect()
}
unsafe fn module_path(
    handle: windows_sys::Win32::Foundation::HMODULE,
) -> Result<PathBuf, NativeRuntimeError> {
    let mut buffer = vec![0u16; 32768];
    // SAFETY: handle is retained by the caller and buffer is writable.
    let count = unsafe { GetModuleFileNameW(handle, buffer.as_mut_ptr(), buffer.len() as u32) };
    if count == 0 || count as usize >= buffer.len() {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::DependencyLoadFailed,
            "cannot read loaded module path",
        ));
    }
    Ok(PathBuf::from(OsString::from_wide(
        &buffer[..count as usize],
    )))
}
pub(super) unsafe fn open(
    file: &LibraryFile,
    directories: &[PathBuf],
    attempt: &mut LoadAttempt,
) -> Result<&'static libloading::Library, NativeRuntimeError> {
    let name = wide(
        file.path
            .file_name()
            .ok_or_else(|| failure(&file.path, "missing library name"))?,
    );
    let mut handle = std::ptr::null_mut();
    // SAFETY: name is NUL terminated; acquiring a reference protects the handle.
    if unsafe { GetModuleHandleExW(0, name.as_ptr(), &mut handle) } != 0 {
        attempt.begin();
        // SAFETY: GetModuleHandleExW supplied one owned reference, retained for process lifetime.
        let library: &'static libloading::Library = Box::leak(Box::new(
            unsafe { libloading::os::windows::Library::from_raw(handle as isize) }.into(),
        ));
        // SAFETY: library keeps handle alive.
        let actual = LibraryFile::resolve(&unsafe { module_path(handle) }?)?;
        if actual.identity != file.identity {
            return Err(NativeRuntimeError::new(
                NativeRuntimeErrorKind::RuntimeConflict,
                "module with the same name is already loaded from another file",
            )
            .with_path(&actual.path)
            .with_path(&file.path));
        }
        return Ok(library);
    }
    attempt.begin();
    // Cookies intentionally remain installed for delayed dependencies throughout process lifetime.
    for directory in directories {
        let directory_w = wide(directory.as_os_str());
        // SAFETY: directory_w is a NUL-terminated absolute directory path.
        if unsafe { AddDllDirectory(directory_w.as_ptr()) }.is_null() {
            return Err(failure(directory, std::io::Error::last_os_error()));
        }
    }
    // SAFETY: caller selected this executable file; dependency search excludes cwd and PATH.
    let library = unsafe {
        libloading::os::windows::Library::load_with_flags(
            &file.path,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
                | LOAD_LIBRARY_SEARCH_SYSTEM32
                | LOAD_LIBRARY_SEARCH_USER_DIRS,
        )
    }
    .map_err(|e| failure(&file.path, e))?;
    let library: &'static libloading::Library = Box::leak(Box::new(library.into()));
    let mut actual_handle = std::ptr::null_mut();
    let full = wide(file.path.as_os_str());
    // SAFETY: full is NUL terminated and the permanent library owns the module reference.
    if unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            full.as_ptr(),
            &mut actual_handle,
        )
    } == 0
    {
        return Err(failure(&file.path, "cannot verify loaded module"));
    }
    // SAFETY: the permanent library owns the module.
    let actual = LibraryFile::resolve(&unsafe { module_path(actual_handle) }?)?;
    if actual.identity != file.identity {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::RuntimeConflict,
            "loaded module differs from selected file",
        )
        .with_path(&actual.path)
        .with_path(&file.path));
    }
    Ok(library)
}

pub(super) fn loaded_paths() -> Result<Vec<PathBuf>, NativeRuntimeError> {
    use windows_sys::Win32::{
        Foundation::{CloseHandle, INVALID_HANDLE_VALUE},
        System::Diagnostics::ToolHelp::*,
    };
    // SAFETY: enumerate the current process; no foreign memory is accessed directly.
    let snapshot = unsafe {
        CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, std::process::id())
    };
    if snapshot == INVALID_HANDLE_VALUE {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::DependencyLoadFailed,
            std::io::Error::last_os_error().to_string(),
        ));
    }
    let mut entry = MODULEENTRY32W {
        dwSize: std::mem::size_of::<MODULEENTRY32W>() as u32,
        ..Default::default()
    };
    let mut paths = Vec::new();
    // SAFETY: snapshot is valid and entry has the documented structure size.
    let mut found = unsafe { Module32FirstW(snapshot, &mut entry) };
    while found != 0 {
        let len = entry
            .szExePath
            .iter()
            .position(|&value| value == 0)
            .unwrap_or(entry.szExePath.len());
        paths.push(PathBuf::from(OsString::from_wide(&entry.szExePath[..len])));
        // SAFETY: same live snapshot and writable entry.
        found = unsafe { Module32NextW(snapshot, &mut entry) };
    }
    let error = std::io::Error::last_os_error();
    // SAFETY: this function owns the snapshot and closes it exactly once.
    unsafe {
        CloseHandle(snapshot);
    }
    if error.raw_os_error() != Some(18) {
        return Err(NativeRuntimeError::new(
            NativeRuntimeErrorKind::DependencyLoadFailed,
            error.to_string(),
        ));
    }
    Ok(paths)
}
