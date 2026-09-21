# Low-level runtime APIs

[Project overview](../README.md) · [Documentation index](../README.md#documentation)

Use the SDK-shaped executors when you need direct control over geometry, device buffers, or interactive execution.

## Inference-free Audio2Emotion

`PostProcessEmotionExecutorFactory` creates the original post-process-only
executor without loading a TensorRT engine. Its `load` method returns a
runtime-independent Future. Audio samples define the duration but are not
read; each frame feeds zero classifier output plus the configured or
accumulated preferred emotion into the post-processor.

```rust,no_run
use audio2face3d::audio2emotion::EmotionExecutor;
use audio2face3d::audio2emotion::post_process::{
    PostProcessEmotionExecutorCreationParameters, PostProcessEmotionExecutorFactory,
};

async fn run(parameters: PostProcessEmotionExecutorCreationParameters) -> audio2face3d::Result<()> {
    let mut executor = PostProcessEmotionExecutorFactory::load(parameters).await?;
    let mut callback = |_result| std::ops::ControlFlow::Continue(());
    executor.execute(&mut callback)?.await?;
    Ok(())
}
```

## Component composition

`GeometryExecutorBundleFactory` wraps the model-specific Regression or
Diffusion factory in a closed, non-generic owning bundle. `load` returns a
runtime-independent Future and retains model kind, track resources, CUDA
stream, and output validation in the completed executor:

```rust,no_run
use audio2face3d::audio2face::{
    GeometryExecutorBundle, GeometryExecutorBundleCreationParameters,
    GeometryExecutorBundleFactory,
};

async fn load(
    parameters: GeometryExecutorBundleCreationParameters,
) -> audio2face3d::Result<GeometryExecutorBundle> {
    GeometryExecutorBundleFactory::load(parameters).await
}
```

After loading, accumulate and close the shared track accumulators. Match the
closed bundle once to call `GeometryExecutor::execute`, copy any required
`DeviceComponentResults` with `copy_to` inside the callback, and await the
returned `Execution`.

The closed bundle exposes `executor()`, `cuda_stream()`,
`audio_accumulator()`, and `emotion_accumulator()` accessors. Geometry can be
consumed exactly once by `try_into_host_blendshape` or
`try_into_device_blendshape`; failed transfers return the original bundle in a
`TransferError`.

## Interactive and GPU callback quick start

Interactive execution keeps inference, post-processing, and BlendShape caches
inside one owner. Callback device views are valid only for the callback and
cannot be retained by safe Rust:

```rust,no_run
use audio2face3d::audio2face::GeometryInteractiveExecutor;
use audio2face3d::audio2face::regression::{
    RegressionGeometryInteractiveExecutorCreationParameters,
    RegressionGeometryInteractiveExecutorFactory,
};

async fn run(
    parameters: RegressionGeometryInteractiveExecutorCreationParameters,
) -> audio2face3d::Result<()> {
    let mut executor = RegressionGeometryInteractiveExecutorFactory::load(parameters).await?;
    let mut callback = |_result| std::ops::ControlFlow::Continue(());
    executor.compute_frame(0, &mut callback).await?;
    Ok(())
}
```

Use `compute_all_frames` for an ordered temporal pass. Use `compute_frame` for
random access and replay; invalidating geometry automatically invalidates the
dependent cache. Host/device BlendShape conversion is exposed by the owning
geometry bundle and all device views require an explicit copy on their stream.

Diffusion derives its frame cadence from the model's frame count, audio window,
and sample rate. Its standard creation parameter `frame_rate` is retained for
source compatibility; it does not override that cadence. With `constant_noise`
enabled, the supplied seed generates one cuRAND noise tensor shared across all
tracks and inference calls, including interactive replay.

## Standalone teeth animation

`AnimatorTeeth` is the host-side semantic name for the teeth transform and
matches the original `IAnimatorTeeth` parameter and column-major transform
contract. Create it through `create_animator_teeth`; the lower-level
`JawTransform` helper remains an implementation detail. GPU solver work is
returned through the owning device executor and its stream-scoped results.

## Safety, errors, and versioning

Completed executors and TensorRT sessions are `Send` but not `Sync`;
operations and destruction restore the appropriate CUDA context. Shared device,
stream, buffer, and borrowed-view capabilities follow the type-specific
auto-trait assertions in the API contract tests. Device views borrow their allocations, and
asynchronous fences retain all borrowed resources until queued CUDA work is
observed complete. Device identity is checked before composition. Interactive
cache eviction, invalidation, and destruction synchronize before releasing
referenced buffers. TensorRT execution contexts require mutable access and
cannot be enqueued concurrently.

Every TensorRT C++ shim entry catches C++ exceptions before returning through
the C ABI. Public operations report malformed input, schema, device, CUDA, and
TensorRT failures through `audio2face3d::Result`; panics are not part of the
public error contract. Compile-fail tests fix the view/fence lifetime and
auto-trait rules, while `release_stress` covers long input, maximum track
limits, reset/replay, queued work, and drop behavior.

This project follows SemVer 2.0.0. Before 1.0, minor releases may change the
public API; patch releases remain compatible. Removing a public feature is a
breaking change, and every release requires a public-API and feature review.
