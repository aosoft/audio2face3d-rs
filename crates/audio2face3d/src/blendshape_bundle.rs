//! High-level geometry-to-blendshape execution bundles.

use crate::animation::{
    BlendshapeData, BlendshapeSolverKind, CpuBlendshapeSolver, GpuBlendshapeSolver,
};
use crate::common::{
    AudioAccumulator, EmotionAccumulator, Error, ModelDataPaths, Result, load_blendshape_config,
};
use crate::cuda::{CudaStream, DeviceBuffer, DeviceView, GpuDevice, ensure_same_device};
use crate::{
    CallbackMetadata, GeometryExecutorBundle, GeometryFrame, Model, PipelineOptions,
    PipelineStatus, TrackParameters,
};
use std::sync::Arc;

/// Blendshape weights produced for one geometry frame.
///
/// The host variant borrows ordinary slices. The device variant borrows CUDA
/// allocations and the stream on which both solves were enqueued. Consumers
/// may enqueue dependent device work on that stream from inside the callback;
/// host reads must synchronize first. Skin weights
/// always precede tongue weights conceptually, matching the original SDK's
/// packed result layout while keeping the two Rust slices dimension-safe.
#[derive(Clone, Copy, Debug)]
pub enum BlendshapeOutput<'a> {
    Host {
        skin_weights: &'a [f32],
        tongue_weights: &'a [f32],
    },
    Device {
        skin_weights: Option<DeviceView<'a, f32>>,
        tongue_weights: Option<DeviceView<'a, f32>>,
        stream: &'a CudaStream,
    },
}

impl BlendshapeOutput<'_> {
    pub fn skin_weight_count(&self) -> usize {
        match self {
            Self::Host { skin_weights, .. } => skin_weights.len(),
            Self::Device { skin_weights, .. } => skin_weights.map_or(0, |weights| weights.len()),
        }
    }

    pub fn tongue_weight_count(&self) -> usize {
        match self {
            Self::Host { tongue_weights, .. } => tongue_weights.len(),
            Self::Device { tongue_weights, .. } => {
                tongue_weights.map_or(0, |weights| weights.len())
            }
        }
    }
}

/// Borrowed solver components for a single bundle track.
pub enum BlendshapeSolverComponents<'a> {
    Cpu {
        skin: Option<&'a CpuBlendshapeSolver>,
        tongue: Option<&'a CpuBlendshapeSolver>,
    },
    Gpu {
        skin: Option<&'a GpuBlendshapeSolver>,
        tongue: Option<&'a GpuBlendshapeSolver>,
        stream: &'a CudaStream,
    },
}

/// Mutably borrowed solver components for a single bundle track.
pub enum BlendshapeSolverComponentsMut<'a> {
    Cpu {
        skin: Option<&'a mut CpuBlendshapeSolver>,
        tongue: Option<&'a mut CpuBlendshapeSolver>,
    },
    Gpu {
        skin: Option<&'a mut GpuBlendshapeSolver>,
        tongue: Option<&'a mut GpuBlendshapeSolver>,
        stream: &'a CudaStream,
    },
}

pub struct CpuBlendshapeTrackComponents {
    pub skin: Option<CpuBlendshapeSolver>,
    pub tongue: Option<CpuBlendshapeSolver>,
}

pub struct GpuBlendshapeComponent {
    solver: GpuBlendshapeSolver,
    target: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

impl GpuBlendshapeComponent {
    pub fn new(
        solver: GpuBlendshapeSolver,
        target: DeviceBuffer<f32>,
        output: DeviceBuffer<f32>,
    ) -> Result<Self> {
        ensure_same_device(solver.device_id(), target.device_id())?;
        ensure_same_device(solver.device_id(), output.device_id())?;
        if target.len() != solver.target_len() || output.len() != solver.pose_count() {
            return Err(invalid(
                "GPU BlendShape component buffer dimensions do not match its solver",
            ));
        }
        Ok(Self {
            solver,
            target,
            output,
        })
    }

    pub fn solver(&self) -> &GpuBlendshapeSolver {
        &self.solver
    }

    pub fn target(&self) -> &DeviceBuffer<f32> {
        &self.target
    }

    pub fn output(&self) -> &DeviceBuffer<f32> {
        &self.output
    }
}

pub struct GpuBlendshapeTrackComponents {
    pub skin: Option<GpuBlendshapeComponent>,
    pub tongue: Option<GpuBlendshapeComponent>,
}

pub struct GpuBlendshapeComponents {
    // Drop resources before their stream and retained device context.
    tracks: Vec<GpuBlendshapeTrackComponents>,
    stream: CudaStream,
    _device: Arc<GpuDevice>,
}

impl GpuBlendshapeComponents {
    pub fn new(
        device: Arc<GpuDevice>,
        stream: CudaStream,
        tracks: Vec<GpuBlendshapeTrackComponents>,
    ) -> Result<Self> {
        ensure_same_device(device.id(), stream.device_id())?;
        for track in &tracks {
            for component in [track.skin.as_ref(), track.tongue.as_ref()]
                .into_iter()
                .flatten()
            {
                ensure_same_device(device.id(), component.solver.device_id())?;
            }
        }
        Ok(Self {
            tracks,
            stream,
            _device: device,
        })
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn tracks(&self) -> &[GpuBlendshapeTrackComponents] {
        &self.tracks
    }
}

enum BundleSolvers {
    Cpu(Vec<CpuBlendshapeTrackComponents>),
    Gpu(GpuBlendshapeComponents),
}

/// Moves a geometry bundle and user-owned solver resources into one bundle.
pub struct BlendshapeExecutorBundleBuilder {
    geometry: GeometryExecutorBundle,
    solvers: BundleSolvers,
}

impl BlendshapeExecutorBundleBuilder {
    pub fn cpu(
        geometry: GeometryExecutorBundle,
        tracks: Vec<CpuBlendshapeTrackComponents>,
    ) -> Self {
        Self {
            geometry,
            solvers: BundleSolvers::Cpu(tracks),
        }
    }

    pub fn gpu(geometry: GeometryExecutorBundle, components: GpuBlendshapeComponents) -> Self {
        Self {
            geometry,
            solvers: BundleSolvers::Gpu(components),
        }
    }

    pub fn build(self) -> Result<BlendshapeExecutorBundle> {
        let track_count = self.geometry.track_count();
        let solver_track_count = match &self.solvers {
            BundleSolvers::Cpu(tracks) => tracks.len(),
            BundleSolvers::Gpu(components) => components.tracks.len(),
        };
        if track_count == 0 || solver_track_count != track_count {
            return Err(invalid(
                "BlendShape solver track count differs from geometry",
            ));
        }
        for track_index in 0..track_count {
            let data = self.geometry.model_data(track_index)?;
            match &self.solvers {
                BundleSolvers::Cpu(tracks) => {
                    validate_cpu_component(
                        tracks[track_index].skin.as_ref(),
                        data.skin_neutral_pose.len(),
                        "skin",
                    )?;
                    validate_cpu_component(
                        tracks[track_index].tongue.as_ref(),
                        data.tongue_neutral_pose.len(),
                        "tongue",
                    )?;
                }
                BundleSolvers::Gpu(components) => {
                    validate_gpu_component(
                        components.tracks[track_index].skin.as_ref(),
                        data.skin_neutral_pose.len(),
                        &components.stream,
                        "skin",
                    )?;
                    validate_gpu_component(
                        components.tracks[track_index].tongue.as_ref(),
                        data.tongue_neutral_pose.len(),
                        &components.stream,
                        "tongue",
                    )?;
                }
            }
        }
        Ok(BlendshapeExecutorBundle {
            geometry: self.geometry,
            solvers: self.solvers,
        })
    }
}

/// Owns geometry execution and per-track skin/tongue blendshape solvers.
pub struct BlendshapeExecutorBundle {
    geometry: GeometryExecutorBundle,
    solvers: BundleSolvers,
}

impl BlendshapeExecutorBundle {
    pub fn load(
        model: &Model,
        options: PipelineOptions,
        kind: BlendshapeSolverKind,
    ) -> Result<Self> {
        let solvers = match kind {
            BlendshapeSolverKind::Cpu => {
                let mut tracks = Vec::with_capacity(options.track_count);
                for track in 0..options.track_count {
                    let index = track.min(model.parameter_count() - 1);
                    tracks.push(CpuBlendshapeTrackComponents {
                        skin: load_cpu_component(model, index, "skin")?,
                        tongue: load_cpu_component(model, index, "tongue")?,
                    });
                }
                BundleSolvers::Cpu(tracks)
            }
            BlendshapeSolverKind::Gpu => {
                let device = GpuDevice::new(options.device_ordinal)?;
                let stream = device.create_stream()?;
                let mut tracks = Vec::with_capacity(options.track_count);
                for track in 0..options.track_count {
                    let index = track.min(model.parameter_count() - 1);
                    tracks.push(GpuBlendshapeTrackComponents {
                        skin: load_gpu_component(model, index, "skin", &device, &stream)?,
                        tongue: load_gpu_component(model, index, "tongue", &device, &stream)?,
                    });
                }
                stream.synchronize()?;
                BundleSolvers::Gpu(GpuBlendshapeComponents::new(device, stream, tracks)?)
            }
        };
        let geometry = GeometryExecutorBundle::load(model, options)?;
        BlendshapeExecutorBundleBuilder { geometry, solvers }.build()
    }

    pub fn geometry(&self) -> &GeometryExecutorBundle {
        &self.geometry
    }

    /// Formal owning geometry executor accessor.
    pub fn executor(&self) -> &GeometryExecutorBundle {
        &self.geometry
    }

    /// Returns the optional CUDA stream owned by the BlendShape bundle.
    ///
    /// CPU bundles do not own a CUDA stream. GPU bundles return their
    /// solver stream, which is the stream used for dependent BlendShape work.
    pub fn cuda_stream(&self) -> Option<&CudaStream> {
        match &self.solvers {
            BundleSolvers::Cpu(_) => None,
            BundleSolvers::Gpu(components) => Some(components.stream()),
        }
    }

    /// Borrows a shared audio accumulator from the underlying geometry
    /// executor.
    pub fn audio_accumulator(&self, track: usize) -> Result<&AudioAccumulator> {
        self.geometry.audio_accumulator(track)
    }

    /// Borrows a shared emotion accumulator from the underlying geometry
    /// executor.
    pub fn emotion_accumulator(&self, track: usize) -> Result<&EmotionAccumulator> {
        self.geometry.emotion_accumulator(track)
    }

    pub fn geometry_mut(&mut self) -> &mut GeometryExecutorBundle {
        &mut self.geometry
    }

    pub fn track_count(&self) -> usize {
        self.geometry.track_count()
    }

    pub fn solver_kind(&self) -> BlendshapeSolverKind {
        match self.solvers {
            BundleSolvers::Cpu(_) => BlendshapeSolverKind::Cpu,
            BundleSolvers::Gpu(_) => BlendshapeSolverKind::Gpu,
        }
    }

    /// Borrows the skin/tongue solver set for one track.
    pub fn solver_components(&self, track: usize) -> Result<BlendshapeSolverComponents<'_>> {
        match &self.solvers {
            BundleSolvers::Cpu(tracks) => {
                let track = tracks
                    .get(track)
                    .ok_or_else(|| invalid("blendshape bundle track is out of range"))?;
                Ok(BlendshapeSolverComponents::Cpu {
                    skin: track.skin.as_ref(),
                    tongue: track.tongue.as_ref(),
                })
            }
            BundleSolvers::Gpu(gpu) => {
                let track = gpu
                    .tracks
                    .get(track)
                    .ok_or_else(|| invalid("blendshape bundle track is out of range"))?;
                Ok(BlendshapeSolverComponents::Gpu {
                    skin: track.skin.as_ref().map(|component| &component.solver),
                    tongue: track.tongue.as_ref().map(|component| &component.solver),
                    stream: &gpu.stream,
                })
            }
        }
    }

    /// Mutably borrows one track's solvers while retaining their owning bundle.
    pub fn solver_components_mut(
        &mut self,
        track: usize,
    ) -> Result<BlendshapeSolverComponentsMut<'_>> {
        match &mut self.solvers {
            BundleSolvers::Cpu(tracks) => {
                let track = tracks
                    .get_mut(track)
                    .ok_or_else(|| invalid("blendshape bundle track is out of range"))?;
                Ok(BlendshapeSolverComponentsMut::Cpu {
                    skin: track.skin.as_mut(),
                    tongue: track.tongue.as_mut(),
                })
            }
            BundleSolvers::Gpu(gpu) => {
                let track = gpu
                    .tracks
                    .get_mut(track)
                    .ok_or_else(|| invalid("blendshape bundle track is out of range"))?;
                Ok(BlendshapeSolverComponentsMut::Gpu {
                    skin: track.skin.as_mut().map(|component| &mut component.solver),
                    tongue: track.tongue.as_mut().map(|component| &mut component.solver),
                    stream: &gpu.stream,
                })
            }
        }
    }

    pub fn accumulate_audio(&self, track: usize, samples: &[f32]) -> Result<()> {
        self.geometry.accumulate_audio(track, samples)
    }

    pub fn close_audio(&self, track: usize) -> Result<()> {
        self.geometry.close_audio(track)
    }

    pub fn set_track_parameters(&mut self, track: usize, value: TrackParameters) -> Result<()> {
        self.geometry.set_track_parameters(track, value)
    }

    /// Executes geometry post-processing and blendshape solving as one path.
    pub fn execute<C>(&mut self, mut callback: C) -> Result<PipelineStatus>
    where
        C: for<'weights> FnMut(CallbackMetadata, BlendshapeOutput<'weights>) -> bool,
    {
        let solvers = &mut self.solvers;
        let mut solve_error = None;
        let status = self.geometry.execute(|metadata, frame| {
            match solve_and_callback(solvers, metadata, frame, &mut callback) {
                Ok(keep_going) => keep_going,
                Err(error) => {
                    solve_error = Some(error);
                    false
                }
            }
        })?;
        if let Some(error) = solve_error {
            Err(error)
        } else {
            Ok(status)
        }
    }

    /// Clears temporal post-process and solver history for one track.
    /// Accumulated audio and recurrent inference state remain owned by the
    /// geometry pipeline and are intentionally not rewound.
    pub fn reset_frame_state(&mut self, track: usize) -> Result<()> {
        self.geometry.reset_postprocess(track)?;
        match &mut self.solvers {
            BundleSolvers::Cpu(tracks) => {
                let track = tracks
                    .get_mut(track)
                    .ok_or_else(|| invalid("blendshape bundle track is out of range"))?;
                if let Some(solver) = &mut track.skin {
                    solver.reset();
                }
                if let Some(solver) = &mut track.tongue {
                    solver.reset();
                }
            }
            BundleSolvers::Gpu(gpu) => {
                let track = gpu
                    .tracks
                    .get_mut(track)
                    .ok_or_else(|| invalid("blendshape bundle track is out of range"))?;
                if let Some(component) = &mut track.skin {
                    component.solver.reset(&gpu.stream)?;
                }
                if let Some(component) = &mut track.tongue {
                    component.solver.reset(&gpu.stream)?;
                }
                gpu.stream.synchronize()?;
            }
        }
        Ok(())
    }

    /// Waits for outstanding device solver work. CPU execution is synchronous.
    pub fn wait(&self) -> Result<()> {
        match &self.solvers {
            BundleSolvers::Cpu(_) => Ok(()),
            BundleSolvers::Gpu(gpu) => gpu.stream.synchronize(),
        }
    }
}

fn solve_and_callback<C>(
    solvers: &mut BundleSolvers,
    metadata: CallbackMetadata,
    frame: &GeometryFrame,
    callback: &mut C,
) -> Result<bool>
where
    C: for<'weights> FnMut(CallbackMetadata, BlendshapeOutput<'weights>) -> bool,
{
    match solvers {
        BundleSolvers::Cpu(tracks) => {
            let track = tracks
                .get_mut(metadata.track)
                .ok_or_else(|| invalid("blendshape bundle track is out of range"))?;
            let skin_weights = track
                .skin
                .as_mut()
                .map(|solver| solver.solve(&frame.skin))
                .transpose()?
                .unwrap_or_default();
            let tongue_weights = track
                .tongue
                .as_mut()
                .map(|solver| solver.solve(&frame.tongue))
                .transpose()?
                .unwrap_or_default();
            Ok(callback(
                metadata,
                BlendshapeOutput::Host {
                    skin_weights: &skin_weights,
                    tongue_weights: &tongue_weights,
                },
            ))
        }
        BundleSolvers::Gpu(gpu) => {
            let track = gpu
                .tracks
                .get_mut(metadata.track)
                .ok_or_else(|| invalid("blendshape bundle track is out of range"))?;
            let GpuBlendshapeTrackComponents { skin, tongue } = track;
            let skin_fence = enqueue_gpu_component(skin.as_mut(), &frame.skin, &gpu.stream)?;
            let tongue_fence = enqueue_gpu_component(tongue.as_mut(), &frame.tongue, &gpu.stream)?;
            Ok(callback(
                metadata,
                BlendshapeOutput::Device {
                    skin_weights: skin_fence.as_ref().map(|fence| fence.output()),
                    tongue_weights: tongue_fence.as_ref().map(|fence| fence.output()),
                    stream: &gpu.stream,
                },
            ))
        }
    }
}

fn enqueue_gpu_component<'a>(
    component: Option<&'a mut GpuBlendshapeComponent>,
    geometry: &[f32],
    stream: &'a CudaStream,
) -> Result<Option<crate::animation::GpuBlendshapeSolveFence<'a>>> {
    let Some(component) = component else {
        return Ok(None);
    };
    component.target.copy_from(geometry, stream)?;
    let fence = component
        .solver
        .solve_async(&component.target, &mut component.output, stream)?;
    Ok(Some(fence))
}

fn load_cpu_component(
    model: &Model,
    index: usize,
    name: &str,
) -> Result<Option<CpuBlendshapeSolver>> {
    let Some(paths) = component_paths(model, index, name)? else {
        return Ok(None);
    };
    let data = BlendshapeData::load_npz(&paths.data)?;
    let config = load_blendshape_config(&paths.config)?.blendshape_params;
    let mut solver = CpuBlendshapeSolver::from_config(data, &config)?;
    solver.prepare()?;
    Ok(Some(solver))
}

fn load_gpu_component(
    model: &Model,
    index: usize,
    name: &str,
    device: &Arc<GpuDevice>,
    stream: &CudaStream,
) -> Result<Option<GpuBlendshapeComponent>> {
    let Some(paths) = component_paths(model, index, name)? else {
        return Ok(None);
    };
    let data = BlendshapeData::load_npz(&paths.data)?;
    let target_count = data.neutral_pose.len();
    let output_count = data.pose_count();
    let config = load_blendshape_config(&paths.config)?.blendshape_params;
    GpuBlendshapeComponent::new(
        GpuBlendshapeSolver::new(device, stream, data, &config)?,
        device.allocate(target_count)?,
        device.allocate(output_count)?,
    )
    .map(Some)
}

fn validate_cpu_component(
    component: Option<&CpuBlendshapeSolver>,
    geometry_len: usize,
    name: &str,
) -> Result<()> {
    if let Some(component) = component {
        if !component.is_prepared() {
            return Err(invalid(format!(
                "{name} CPU BlendShape solver is not prepared"
            )));
        }
        if component.data().neutral_pose.len() != geometry_len {
            return Err(invalid(format!(
                "{name} CPU BlendShape solver shape differs from geometry"
            )));
        }
    }
    Ok(())
}

fn validate_gpu_component(
    component: Option<&GpuBlendshapeComponent>,
    geometry_len: usize,
    stream: &CudaStream,
    name: &str,
) -> Result<()> {
    if let Some(component) = component {
        ensure_same_device(stream.device_id(), component.solver.device_id())?;
        if component.solver.target_len() != geometry_len {
            return Err(invalid(format!(
                "{name} GPU BlendShape solver shape differs from geometry"
            )));
        }
    }
    Ok(())
}

fn component_paths<'a>(
    model: &'a Model,
    index: usize,
    name: &str,
) -> Result<Option<&'a ModelDataPaths>> {
    let paths = model
        .blendshape_paths(index)
        .or_else(|_| model.blendshape_paths(0))?;
    Ok(paths.get(name))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{BlendshapeSolverParameters, EyesRotation};
    use crate::common::{GeometryParameters, NetworkDocument};
    use crate::{ModelKind, ModelParameters};

    fn solver(name: &str) -> CpuBlendshapeSolver {
        let data = BlendshapeData {
            neutral_pose: vec![0.0, 0.0, 0.0, 2.0, 3.0, 4.0],
            delta_poses: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            pose_names: vec![name.into()],
            pose_mask: None,
        };
        let mut solver = CpuBlendshapeSolver::new(data).unwrap();
        solver
            .set_parameters(BlendshapeSolverParameters {
                l1_regularization: 0.0,
                l2_regularization: 0.0,
                symmetry_regularization: 0.0,
                temporal_regularization: 0.0,
                ..BlendshapeSolverParameters::default()
            })
            .unwrap();
        solver.prepare().unwrap();
        solver
    }

    #[test]
    fn output_reports_component_counts() {
        let skin = [0.1, 0.2];
        let tongue = [0.3];
        let output = BlendshapeOutput::Host {
            skin_weights: &skin,
            tongue_weights: &tongue,
        };
        assert_eq!(output.skin_weight_count(), 2);
        assert_eq!(output.tongue_weight_count(), 1);
    }

    #[test]
    fn cpu_component_validation_rejects_unprepared_and_wrong_shapes() {
        let unprepared = CpuBlendshapeSolver::new(BlendshapeData {
            neutral_pose: vec![0.0; 6],
            delta_poses: vec![1.0; 6],
            pose_names: vec!["shape".into()],
            pose_mask: None,
        })
        .unwrap();
        assert!(validate_cpu_component(Some(&unprepared), 6, "skin").is_err());
        let prepared = solver("skin");
        assert!(validate_cpu_component(Some(&prepared), 3, "skin").is_err());
        assert!(validate_cpu_component(Some(&prepared), 6, "skin").is_ok());
        assert!(validate_cpu_component(None, 6, "skin").is_ok());
    }

    #[test]
    fn cpu_track_solves_skin_and_tongue_and_propagates_callback_stop() {
        let mut solvers = BundleSolvers::Cpu(vec![CpuBlendshapeTrackComponents {
            skin: Some(solver("skin")),
            tongue: Some(solver("tongue")),
        }]);
        let frame = GeometryFrame {
            skin: vec![0.25, 0.0, 0.0, 2.0, 3.0, 4.0],
            tongue: vec![0.75, 0.0, 0.0, 2.0, 3.0, 4.0],
            jaw_transform: [0.0; 16],
            eyes_rotation: EyesRotation {
                right: [0.0; 3],
                left: [0.0; 3],
            },
        };
        let metadata = CallbackMetadata {
            kind: ModelKind::Regression,
            track: 0,
            inference: None,
            frame: 4,
            timestamp: 100,
            next_timestamp: 200,
        };
        let mut observed = None;
        let keep_going = solve_and_callback(
            &mut solvers,
            metadata,
            &frame,
            &mut |actual_metadata, output| {
                assert_eq!(actual_metadata, metadata);
                let BlendshapeOutput::Host {
                    skin_weights,
                    tongue_weights,
                } = output
                else {
                    panic!("expected host result")
                };
                observed = Some((skin_weights[0], tongue_weights[0]));
                false
            },
        )
        .unwrap();
        assert!(!keep_going);
        let (skin, tongue) = observed.unwrap();
        assert!((skin - 0.25).abs() < 1.0e-6);
        assert!((tongue - 0.75).abs() < 1.0e-6);
    }

    #[test]
    fn runs_installed_geometry_models_through_bundles_when_configured() {
        let Some(paths) = std::env::var_os("AUDIO2FACE3D_TEST_FACADE_MODELS") else {
            return;
        };
        for root in std::env::split_paths(&paths) {
            let model = Model::load(root.join("model.json")).unwrap();
            if model.kind() == ModelKind::Emotion {
                continue;
            }
            let NetworkDocument::Geometry(network) = model.network() else {
                unreachable!()
            };
            let ModelParameters::Geometry(_) = model.parameters(0).unwrap() else {
                unreachable!()
            };
            for kind in [BlendshapeSolverKind::Cpu, BlendshapeSolverKind::Gpu] {
                let mut bundle =
                    BlendshapeExecutorBundle::load(&model, PipelineOptions::default(), kind)
                        .unwrap();
                let parameters = match &network.params {
                    GeometryParameters::Regression(value) => TrackParameters::Regression {
                        input_strength: 0.5,
                        implicit_emotion: vec![0.0; value.implicit_emotion_len],
                    },
                    GeometryParameters::Diffusion(_) => TrackParameters::Diffusion {
                        input_strength: 0.5,
                        identity_index: 0,
                    },
                };
                bundle.set_track_parameters(0, parameters).unwrap();
                bundle.accumulate_audio(0, &vec![0.0; 1_600]).unwrap();
                bundle.close_audio(0).unwrap();
                let mut callbacks = 0;
                loop {
                    match bundle
                        .execute(|metadata, output| {
                            assert_eq!(metadata.track, 0);
                            assert!(output.skin_weight_count() > 0);
                            assert!(output.tongue_weight_count() > 0);
                            callbacks += 1;
                            true
                        })
                        .unwrap()
                    {
                        PipelineStatus::Executed { .. } => {}
                        PipelineStatus::Complete => break,
                        other => panic!("unexpected bundle status: {other:?}"),
                    }
                }
                bundle.wait().unwrap();
                assert!(callbacks > 0);
            }
        }
    }
}
