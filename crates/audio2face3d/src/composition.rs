//! Ownership-preserving composition for geometry executor components.

use crate::audio2face::GeometryExecutor;
use crate::audio2face::bundle::{
    GeometryExecutorBundle as FacadeGeometryExecutorBundle, GeometryExecutorRef,
};
use crate::audio2x::Executor;
use crate::{
    CallbackMetadata, GeometryExecutorBundle, GeometryFrame, GeometryPipelineComponents, Model,
    ModelKind, PipelineOptions, PipelineStatus, TrackParameters,
};
use crate::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

/// Stable shape contract for one geometry track.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeometryOutputShape {
    pub skin: usize,
    pub tongue: usize,
    pub jaw_transform: usize,
    pub eyes_rotation: usize,
}

impl GeometryOutputShape {
    pub fn validate(self) -> Result<()> {
        if self.skin == 0 || self.tongue == 0 || self.jaw_transform != 16 || self.eyes_rotation != 6
        {
            return Err(invalid("geometry output shape is invalid"));
        }
        Ok(())
    }

    pub fn validate_frame(self, frame: &GeometryFrame) -> Result<()> {
        if frame.skin.len() != self.skin
            || frame.tongue.len() != self.tongue
            || frame.jaw_transform.len() != self.jaw_transform
            || frame.eyes_rotation.right.len() + frame.eyes_rotation.left.len()
                != self.eyes_rotation
        {
            return Err(invalid("geometry component returned an incompatible shape"));
        }
        Ok(())
    }
}

/// An owned geometry execution component accepted by the public composition builder.
///
/// Implementations may contain a TensorRT backend, a fake backend, custom
/// accumulators/post-processing, external typed buffers, and their CUDA stream.
/// All borrowed output remains callback-scoped.
pub trait GeometryExecutorComponent {
    fn kind(&self) -> ModelKind;
    fn track_count(&self) -> usize;
    fn device_ordinal(&self) -> Option<i32>;
    fn output_shape(&self, track: usize) -> Result<GeometryOutputShape>;
    fn accumulate_audio(&self, track: usize, samples: &[f32]) -> Result<()>;
    fn close_audio(&self, track: usize) -> Result<()>;
    fn set_track_parameters(&mut self, track: usize, value: TrackParameters) -> Result<()>;
    fn execute(
        &mut self,
        callback: &mut dyn FnMut(CallbackMetadata, &GeometryFrame) -> bool,
    ) -> Result<PipelineStatus>;
    fn reset(&mut self, track: usize) -> Result<()>;
    fn wait(&self) -> Result<()> {
        Ok(())
    }
}

impl GeometryExecutorComponent for GeometryExecutorBundle {
    fn kind(&self) -> ModelKind {
        self.pipeline().kind()
    }

    fn track_count(&self) -> usize {
        self.track_count()
    }

    fn device_ordinal(&self) -> Option<i32> {
        self.pipeline()
            .geometry_components()
            .ok()
            .map(|components| match components {
                GeometryPipelineComponents::Regression { backend, .. } => {
                    backend.device().id().ordinal()
                }
                GeometryPipelineComponents::Diffusion { backend, .. } => {
                    backend.device().id().ordinal()
                }
            })
    }

    fn output_shape(&self, track: usize) -> Result<GeometryOutputShape> {
        let data = self.model_data(track)?;
        Ok(GeometryOutputShape {
            skin: data.skin_neutral_pose.len(),
            tongue: data.tongue_neutral_pose.len(),
            jaw_transform: 16,
            eyes_rotation: 6,
        })
    }

    fn accumulate_audio(&self, track: usize, samples: &[f32]) -> Result<()> {
        self.accumulate_audio(track, samples)
    }

    fn close_audio(&self, track: usize) -> Result<()> {
        self.close_audio(track)
    }

    fn set_track_parameters(&mut self, track: usize, value: TrackParameters) -> Result<()> {
        self.set_track_parameters(track, value)
    }

    fn execute(
        &mut self,
        callback: &mut dyn FnMut(CallbackMetadata, &GeometryFrame) -> bool,
    ) -> Result<PipelineStatus> {
        self.execute(callback)
    }

    fn reset(&mut self, track: usize) -> Result<()> {
        self.reset_postprocess(track)
    }

    fn wait(&self) -> Result<()> {
        match self.pipeline().geometry_components()? {
            GeometryPipelineComponents::Regression { backend, .. } => {
                backend.stream().synchronize()
            }
            GeometryPipelineComponents::Diffusion { backend, .. } => backend.stream().synchronize(),
        }
    }
}

/// Adapts the completed SDK-facing geometry bundle to the composition layer.
///
/// This adapter is deliberately kept here, rather than adding composition's
/// legacy `PipelineStatus`/`GeometryFrame` types to the facade bundle API.
impl GeometryExecutorComponent for FacadeGeometryExecutorBundle {
    fn kind(&self) -> ModelKind {
        self.kind()
    }

    fn track_count(&self) -> usize {
        self.track_count()
    }

    fn device_ordinal(&self) -> Option<i32> {
        match self.executor() {
            GeometryExecutorRef::Regression(executor) => Some(executor.device_arc().id().ordinal()),
            GeometryExecutorRef::Diffusion(executor) => Some(executor.device_arc().id().ordinal()),
        }
    }

    fn output_shape(&self, track: usize) -> Result<GeometryOutputShape> {
        if track >= self.track_count() {
            return Err(invalid("geometry bundle track is out of range"));
        }
        let shape = match self.executor() {
            GeometryExecutorRef::Regression(executor) => GeometryOutputShape {
                skin: executor.skin_geometry_size(),
                tongue: executor.tongue_geometry_size(),
                jaw_transform: executor.jaw_transform_size(),
                eyes_rotation: executor.eyes_rotation_size(),
            },
            GeometryExecutorRef::Diffusion(executor) => GeometryOutputShape {
                skin: executor.skin_geometry_size(),
                tongue: executor.tongue_geometry_size(),
                jaw_transform: executor.jaw_transform_size(),
                eyes_rotation: executor.eyes_rotation_size(),
            },
        };
        shape.validate()?;
        Ok(shape)
    }

    fn accumulate_audio(&self, track: usize, samples: &[f32]) -> Result<()> {
        self.audio_accumulator(track)?.accumulate(samples)
    }

    fn close_audio(&self, track: usize) -> Result<()> {
        self.audio_accumulator(track)?.close()
    }

    fn set_track_parameters(&mut self, track: usize, value: TrackParameters) -> Result<()> {
        match (self, value) {
            (
                FacadeGeometryExecutorBundle::Regression(executor),
                TrackParameters::Regression {
                    input_strength,
                    implicit_emotion,
                },
            ) => {
                executor.set_input_strength(input_strength)?;
                executor.set_implicit_emotion(track, &implicit_emotion)
            }
            (
                FacadeGeometryExecutorBundle::Diffusion(executor),
                TrackParameters::Diffusion {
                    input_strength,
                    identity_index,
                },
            ) => {
                executor.set_input_strength(input_strength)?;
                executor.set_identity_index(identity_index)
            }
            _ => Err(invalid(
                "geometry track parameters do not match bundle model",
            )),
        }
    }

    fn execute(
        &mut self,
        callback: &mut dyn FnMut(CallbackMetadata, &GeometryFrame) -> bool,
    ) -> Result<PipelineStatus> {
        match self {
            FacadeGeometryExecutorBundle::Regression(executor) => {
                let status = executor.execute_host(|metadata, geometry| {
                    let keep_going = callback(
                        CallbackMetadata {
                            kind: ModelKind::Regression,
                            track: metadata.track,
                            inference: None,
                            frame: metadata.frame,
                            timestamp: metadata.timestamp,
                            next_timestamp: metadata.next_timestamp,
                        },
                        &GeometryFrame::from(geometry.clone()),
                    );
                    if keep_going {
                        std::ops::ControlFlow::Continue(())
                    } else {
                        std::ops::ControlFlow::Break(())
                    }
                })?;
                Ok(match status {
                    crate::animation::PumpStatus::AwaitingInput => PipelineStatus::AwaitingInput,
                    crate::animation::PumpStatus::Complete => PipelineStatus::Complete,
                    crate::animation::PumpStatus::Interrupted => PipelineStatus::Interrupted,
                })
            }
            FacadeGeometryExecutorBundle::Diffusion(executor) => {
                let mut frames = 0;
                let status = executor.execute_host(|metadata, geometry| {
                    frames += 1;
                    let keep_going = callback(
                        CallbackMetadata {
                            kind: ModelKind::Diffusion,
                            track: metadata.track,
                            inference: Some(metadata.inference),
                            frame: metadata.frame,
                            timestamp: metadata.timestamp,
                            next_timestamp: metadata.next_timestamp,
                        },
                        &GeometryFrame::from(geometry.clone()),
                    );
                    if keep_going {
                        std::ops::ControlFlow::Continue(())
                    } else {
                        std::ops::ControlFlow::Break(())
                    }
                })?;
                Ok(match status {
                    crate::animation::DiffusionExecutionStatus::AwaitingInput => {
                        PipelineStatus::AwaitingInput
                    }
                    crate::animation::DiffusionExecutionStatus::Executed { tracks } => {
                        PipelineStatus::Executed { tracks, frames }
                    }
                    crate::animation::DiffusionExecutionStatus::Complete => {
                        PipelineStatus::Complete
                    }
                })
            }
        }
    }

    fn reset(&mut self, track: usize) -> Result<()> {
        match self {
            FacadeGeometryExecutorBundle::Regression(executor) => {
                Executor::reset_track(executor, track)
            }
            FacadeGeometryExecutorBundle::Diffusion(executor) => {
                Executor::reset_track(executor, track)
            }
        }
    }

    fn wait(&self) -> Result<()> {
        self.cuda_stream().synchronize()
    }
}

/// Receives validated geometry frames before the user callback.
pub trait GeometryObserver {
    fn observe(&mut self, metadata: CallbackMetadata, frame: &GeometryFrame) -> Result<()>;

    fn reset(&mut self, _track: usize) -> Result<()> {
        Ok(())
    }
}

impl<F> GeometryObserver for F
where
    F: FnMut(CallbackMetadata, &GeometryFrame) -> Result<()>,
{
    fn observe(&mut self, metadata: CallbackMetadata, frame: &GeometryFrame) -> Result<()> {
        self(metadata, frame)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GeometryComponentContract {
    pub kind: Option<ModelKind>,
    pub track_count: Option<usize>,
    pub device_ordinal: Option<i32>,
}

/// Builder that moves a user-owned component into a validated high-level bundle.
pub struct GeometryExecutorBundleBuilder<C> {
    component: C,
    contract: GeometryComponentContract,
    observer: Option<Box<dyn GeometryObserver>>,
}

impl GeometryExecutorBundleBuilder<GeometryExecutorBundle> {
    pub fn from_model(model: &Model, options: PipelineOptions) -> Result<Self> {
        let expected = GeometryComponentContract {
            kind: Some(model.kind()),
            track_count: Some(options.track_count),
            device_ordinal: Some(options.device_ordinal),
        };
        Ok(Self::new(GeometryExecutorBundle::load(model, options)?).contract(expected))
    }
}

impl<C> GeometryExecutorBundleBuilder<C>
where
    C: GeometryExecutorComponent,
{
    pub fn new(component: C) -> Self {
        Self {
            component,
            contract: GeometryComponentContract::default(),
            observer: None,
        }
    }

    pub fn contract(mut self, contract: GeometryComponentContract) -> Self {
        self.contract = contract;
        self
    }

    pub fn observer(mut self, observer: impl GeometryObserver + 'static) -> Self {
        self.observer = Some(Box::new(observer));
        self
    }

    pub fn build(self) -> Result<ComposedGeometryExecutorBundle<C>> {
        let kind = self.component.kind();
        if kind == ModelKind::Emotion {
            return Err(invalid("geometry component cannot use an emotion model"));
        }
        let track_count = self.component.track_count();
        if track_count == 0 {
            return Err(invalid("geometry component track count must be positive"));
        }
        if self.contract.kind.is_some_and(|expected| expected != kind) {
            return Err(invalid(
                "geometry component model kind differs from its contract",
            ));
        }
        if self
            .contract
            .track_count
            .is_some_and(|expected| expected != track_count)
        {
            return Err(invalid(
                "geometry component track count differs from its contract",
            ));
        }
        if self.contract.device_ordinal.is_some()
            && self.contract.device_ordinal != self.component.device_ordinal()
        {
            return Err(invalid(
                "geometry component device differs from its contract",
            ));
        }
        let shapes = (0..track_count)
            .map(|track| {
                let shape = self.component.output_shape(track)?;
                shape.validate()?;
                Ok(shape)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ComposedGeometryExecutorBundle {
            component: self.component,
            observer: self.observer,
            shapes,
        })
    }
}

/// A validated high-level geometry bundle with an optional injected observer.
pub struct ComposedGeometryExecutorBundle<C> {
    // Components drop before observers, preserving dependency order.
    component: C,
    observer: Option<Box<dyn GeometryObserver>>,
    shapes: Vec<GeometryOutputShape>,
}

impl<C> ComposedGeometryExecutorBundle<C>
where
    C: GeometryExecutorComponent,
{
    pub fn component(&self) -> &C {
        &self.component
    }

    pub fn component_mut(&mut self) -> &mut C {
        &mut self.component
    }

    pub fn kind(&self) -> ModelKind {
        self.component.kind()
    }

    pub fn track_count(&self) -> usize {
        self.component.track_count()
    }

    pub fn output_shape(&self, track: usize) -> Result<GeometryOutputShape> {
        self.shapes
            .get(track)
            .copied()
            .ok_or_else(|| invalid("geometry bundle track is out of range"))
    }

    pub fn accumulate_audio(&self, track: usize, samples: &[f32]) -> Result<()> {
        self.component.accumulate_audio(track, samples)
    }

    pub fn close_audio(&self, track: usize) -> Result<()> {
        self.component.close_audio(track)
    }

    pub fn set_track_parameters(&mut self, track: usize, value: TrackParameters) -> Result<()> {
        self.component.set_track_parameters(track, value)
    }

    pub fn execute<Cb>(&mut self, mut callback: Cb) -> Result<PipelineStatus>
    where
        Cb: for<'frame> FnMut(CallbackMetadata, &'frame GeometryFrame) -> bool,
    {
        let observer = &mut self.observer;
        let shapes = &self.shapes;
        let mut callback_error = None;
        let status = self.component.execute(&mut |metadata, frame| {
            let result = shapes
                .get(metadata.track)
                .ok_or_else(|| invalid("geometry callback track is out of range"))
                .and_then(|shape| shape.validate_frame(frame))
                .and_then(|()| match observer {
                    Some(observer) => observer.observe(metadata, frame),
                    None => Ok(()),
                });
            match result {
                Ok(()) => callback(metadata, frame),
                Err(error) => {
                    callback_error = Some(error);
                    false
                }
            }
        })?;
        callback_error.map_or(Ok(status), Err)
    }

    pub fn reset(&mut self, track: usize) -> Result<()> {
        self.output_shape(track)?;
        self.component.reset(track)?;
        if let Some(observer) = &mut self.observer {
            observer.reset(track)?;
        }
        Ok(())
    }

    pub fn wait(&self) -> Result<()> {
        self.component.wait()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::EyesRotation;
    use std::cell::RefCell;
    use std::rc::Rc;

    struct FakeGeometry {
        kind: ModelKind,
        tracks: usize,
        device: Option<i32>,
        shape: GeometryOutputShape,
        frame: GeometryFrame,
        events: Rc<RefCell<Vec<&'static str>>>,
    }

    impl Drop for FakeGeometry {
        fn drop(&mut self) {
            self.events.borrow_mut().push("component-drop");
        }
    }

    impl GeometryExecutorComponent for FakeGeometry {
        fn kind(&self) -> ModelKind {
            self.kind
        }
        fn track_count(&self) -> usize {
            self.tracks
        }
        fn device_ordinal(&self) -> Option<i32> {
            self.device
        }
        fn output_shape(&self, track: usize) -> Result<GeometryOutputShape> {
            (track < self.tracks)
                .then_some(self.shape)
                .ok_or_else(|| invalid("track"))
        }
        fn accumulate_audio(&self, _: usize, _: &[f32]) -> Result<()> {
            Ok(())
        }
        fn close_audio(&self, _: usize) -> Result<()> {
            Ok(())
        }
        fn set_track_parameters(&mut self, _: usize, _: TrackParameters) -> Result<()> {
            Ok(())
        }
        fn execute(
            &mut self,
            callback: &mut dyn FnMut(CallbackMetadata, &GeometryFrame) -> bool,
        ) -> Result<PipelineStatus> {
            self.events.borrow_mut().push("execute");
            let keep_running = callback(
                CallbackMetadata {
                    kind: self.kind,
                    track: 0,
                    inference: None,
                    frame: 0,
                    timestamp: 0,
                    next_timestamp: 1,
                },
                &self.frame,
            );
            Ok(if keep_running {
                PipelineStatus::Complete
            } else {
                PipelineStatus::Interrupted
            })
        }
        fn reset(&mut self, _: usize) -> Result<()> {
            self.events.borrow_mut().push("component-reset");
            Ok(())
        }
    }

    struct RecordingObserver(Rc<RefCell<Vec<&'static str>>>);

    impl GeometryObserver for RecordingObserver {
        fn observe(&mut self, _: CallbackMetadata, _: &GeometryFrame) -> Result<()> {
            self.0.borrow_mut().push("observe");
            Ok(())
        }
        fn reset(&mut self, _: usize) -> Result<()> {
            self.0.borrow_mut().push("observer-reset");
            Ok(())
        }
    }

    impl Drop for RecordingObserver {
        fn drop(&mut self) {
            self.0.borrow_mut().push("observer-drop");
        }
    }

    fn fake(events: Rc<RefCell<Vec<&'static str>>>) -> FakeGeometry {
        FakeGeometry {
            kind: ModelKind::Regression,
            tracks: 1,
            device: Some(0),
            shape: GeometryOutputShape {
                skin: 3,
                tongue: 3,
                jaw_transform: 16,
                eyes_rotation: 6,
            },
            frame: GeometryFrame {
                skin: vec![0.0; 3],
                tongue: vec![0.0; 3],
                jaw_transform: [0.0; 16],
                eyes_rotation: EyesRotation {
                    right: [0.0; 3],
                    left: [0.0; 3],
                },
            },
            events,
        }
    }

    #[test]
    fn validates_contract_before_execution() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let result = GeometryExecutorBundleBuilder::new(fake(events))
            .contract(GeometryComponentContract {
                kind: Some(ModelKind::Diffusion),
                track_count: Some(1),
                device_ordinal: Some(0),
            })
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn rejects_track_device_and_shape_mismatches() {
        let events = Rc::new(RefCell::new(Vec::new()));
        assert!(
            GeometryExecutorBundleBuilder::new(fake(Rc::clone(&events)))
                .contract(GeometryComponentContract {
                    kind: Some(ModelKind::Regression),
                    track_count: Some(2),
                    device_ordinal: Some(0),
                })
                .build()
                .is_err()
        );
        assert!(
            GeometryExecutorBundleBuilder::new(fake(Rc::clone(&events)))
                .contract(GeometryComponentContract {
                    kind: Some(ModelKind::Regression),
                    track_count: Some(1),
                    device_ordinal: Some(1),
                })
                .build()
                .is_err()
        );
        let mut invalid_shape = fake(events);
        invalid_shape.shape.jaw_transform = 15;
        assert!(
            GeometryExecutorBundleBuilder::new(invalid_shape)
                .build()
                .is_err()
        );
    }

    #[test]
    fn custom_component_observer_reset_and_drop_order_are_stable() {
        let events = Rc::new(RefCell::new(Vec::new()));
        {
            let mut bundle = GeometryExecutorBundleBuilder::new(fake(Rc::clone(&events)))
                .observer(RecordingObserver(Rc::clone(&events)))
                .build()
                .unwrap();
            bundle
                .execute(|_, _| {
                    events.borrow_mut().push("callback");
                    true
                })
                .unwrap();
            bundle.reset(0).unwrap();
        }
        assert_eq!(
            *events.borrow(),
            [
                "execute",
                "observe",
                "callback",
                "component-reset",
                "observer-reset",
                "component-drop",
                "observer-drop",
            ]
        );
    }

    #[test]
    fn observer_errors_stop_before_the_user_callback() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut bundle = GeometryExecutorBundleBuilder::new(fake(events))
            .observer(|_: CallbackMetadata, _: &GeometryFrame| Err(invalid("observer failed")))
            .build()
            .unwrap();
        let mut callbacks = 0;
        assert!(
            bundle
                .execute(|_, _| {
                    callbacks += 1;
                    true
                })
                .is_err()
        );
        assert_eq!(callbacks, 0);
    }
}
