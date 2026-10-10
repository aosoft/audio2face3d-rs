# Native scheduler ownership

R5 makes host adapters and native adapters run the same model-specific driver.

- Regression uses `pump_with` / `pump_inner` for readiness, fixed track slots, callback suppression, frame advancement and accumulator disposal. Its adapter prepares a batch and delivers one track's result without borrowing scheduler state during callbacks.
- Diffusion uses `execute_with` / `execute_inner` for fixed slots, noise preparation, latent commits, frame-major callbacks, negative/end-of-audio timestamp filtering and inference advancement. Adapters own output validation, latent access and host or device result delivery.
- Classifier uses `execute_with` for compact batches. Batch slot indices stay separate from track indices; adapters own preferred-emotion postprocessing and callback borrows.
- The Regression and Diffusion interactive facades share `interactive_driver` for yielding, interrupt generations, frame iteration and emitted-frame totals. Log scopes still wrap each future's poll and drop.

Callback suppression is local to one call. Successful inference/state updates and input disposal retain their existing order. Backend or postprocess errors do not advance the frame/inference index; Regression and Diffusion running guards also release on errors and callback panics. Mathematical processing, seeds, public signatures and output contracts are unchanged.

Validation includes 174 portable library tests with `animation,emotion`, all-feature/all-target compilation and Clippy with warnings denied. Added tests exercise adapter inference/delivery failures, a compact batch with a finished first track, and the shared facade's interrupt/yield/count behavior. Existing traces cover fixed inactive slots, callback resumption, recurrent state, noise and input windows. The `cuda-lifetime` tier also checks native buffers and callback lifetime contracts.

Final R5–R8 integration checks also passed the `tensorrt-model` tier, including acquired Regression, Diffusion and Emotion models. C++ SDK comparisons passed standard execution with two tracks and interactive-all execution with one track for all three pipelines, plus the existing emotion and standalone-teeth cases. Metadata matched exactly and values satisfied the configured component tolerances. See [the validation record](refactoring-r5-r8-validation.md).

These tests do not establish a performance improvement or bitwise CPU/GPU equality.
