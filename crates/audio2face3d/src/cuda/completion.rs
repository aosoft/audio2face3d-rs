//! Completion of native work that borrows Rust-owned storage.

use crate::common::Result;

/// Armed immediately before native submission, including calls that may
/// partially submit work before failing. Both return and unwinding must drain
/// that work before releasing the caller's host/device borrow.
pub(crate) struct CompletionGuard<F: FnOnce() -> Result<()>> {
    synchronize: Option<F>,
}

impl<F: FnOnce() -> Result<()>> CompletionGuard<F> {
    pub(crate) fn new(synchronize: F) -> Self {
        Self {
            synchronize: Some(synchronize),
        }
    }

    pub(crate) fn finish(mut self) {
        self.complete();
    }

    fn complete(&mut self) {
        if let Some(synchronize) = self.synchronize.take() {
            // No state is reused after an unsuccessful or panicking wait, so
            // catching an unwind here is solely a boundary for terminating.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(synchronize));
            // Returning or unwinding would release borrowed storage while
            // native work might still access it. A failed wait does not prove
            // completion, so this boundary is deliberately fail-stop.
            if !matches!(result, Ok(Ok(()))) {
                std::process::abort();
            }
        }
    }
}

impl<F: FnOnce() -> Result<()>> Drop for CompletionGuard<F> {
    fn drop(&mut self) {
        self.complete();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Error;
    use std::cell::Cell;

    #[test]
    fn success_completes_once_before_releasing_the_borrow() {
        let calls = Cell::new(0);
        let completion = CompletionGuard::new(|| {
            calls.set(calls.get() + 1);
            Ok(())
        });
        completion.finish();
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn submission_error_still_completes() {
        fn submit(completed: &Cell<bool>) -> Result<()> {
            let _completion = CompletionGuard::new(|| {
                completed.set(true);
                Ok(())
            });
            Err(Error::InvalidSchema("injected submission error".into()))
        }
        let completed = Cell::new(false);
        assert!(submit(&completed).is_err());
        assert!(completed.get());
    }

    #[test]
    fn unwinding_still_completes() {
        let completed = Cell::new(false);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _completion = CompletionGuard::new(|| {
                completed.set(true);
                Ok(())
            });
            panic!("injected postprocess panic");
        }));
        assert!(result.is_err());
        assert!(completed.get());
    }

    #[test]
    fn failed_wait_does_not_return_to_the_caller() {
        const CHILD: &str = "AUDIO2FACE3D_COMPLETION_ABORT_TEST";
        if let Some(mode) = std::env::var_os(CHILD) {
            if mode == "error" {
                CompletionGuard::new(|| {
                    Err(Error::CudaUnavailable("injected wait failure".into()))
                })
                .finish();
            } else {
                // If a panicking wait unwinds instead of aborting, this outer
                // catcher releases the borrow and makes the child exit zero.
                let _ = std::panic::catch_unwind(|| {
                    CompletionGuard::new(|| panic!("injected wait panic")).finish();
                });
            }
            std::process::exit(0);
        }
        for mode in ["error", "panic"] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "cuda::completion::tests::failed_wait_does_not_return_to_the_caller",
                ])
                .env(CHILD, mode)
                .status()
                .unwrap();
            assert!(!status.success(), "{mode} wait returned to the caller");
        }
    }
}
