//! High-level geometry-to-blendshape execution bundles.

use crate::animation::{
    BlendshapeData, BlendshapeSolverKind, CpuBlendshapeSolver, GpuBlendshapeSolver,
};
use crate::common::{Error, ModelDataPaths, Result, load_blendshape_config};
use crate::cuda::{CudaStream, DeviceBuffer, DeviceView, GpuDevice};
use crate::{
    CallbackMetadata, GeometryExecutorBundle, GeometryFrame, Model, PipelineOptions,
    PipelineStatus, TrackParameters,
};
use std::rc::Rc;

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

struct CpuTrackSolvers {
    skin: Option<CpuBlendshapeSolver>,
    tongue: Option<CpuBlendshapeSolver>,
}

struct GpuComponentSolver {
    solver: GpuBlendshapeSolver,
    target: DeviceBuffer<f32>,
    output: DeviceBuffer<f32>,
}

struct GpuTrackSolvers {
    skin: Option<GpuComponentSolver>,
    tongue: Option<GpuComponentSolver>,
}

struct GpuSolvers {
    // Drop resources before their stream and retained device context.
    tracks: Vec<GpuTrackSolvers>,
    stream: CudaStream,
    _device: Rc<GpuDevice>,
}

enum BundleSolvers {
    Cpu(Vec<CpuTrackSolvers>),
    Gpu(GpuSolvers),
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
                    tracks.push(CpuTrackSolvers {
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
                    tracks.push(GpuTrackSolvers {
                        skin: load_gpu_component(model, index, "skin", &device, &stream)?,
                        tongue: load_gpu_component(model, index, "tongue", &device, &stream)?,
                    });
                }
                stream.synchronize()?;
                BundleSolvers::Gpu(GpuSolvers {
                    tracks,
                    stream,
                    _device: device,
                })
            }
        };
        let geometry = GeometryExecutorBundle::load(model, options)?;
        Ok(Self { geometry, solvers })
    }

    pub fn geometry(&self) -> &GeometryExecutorBundle {
        &self.geometry
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
            let GpuTrackSolvers { skin, tongue } = track;
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
    component: Option<&'a mut GpuComponentSolver>,
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
    device: &Rc<GpuDevice>,
    stream: &CudaStream,
) -> Result<Option<GpuComponentSolver>> {
    let Some(paths) = component_paths(model, index, name)? else {
        return Ok(None);
    };
    let data = BlendshapeData::load_npz(&paths.data)?;
    let target_count = data.neutral_pose.len();
    let output_count = data.pose_count();
    let config = load_blendshape_config(&paths.config)?.blendshape_params;
    Ok(Some(GpuComponentSolver {
        solver: GpuBlendshapeSolver::new(device, stream, data, &config)?,
        target: device.allocate(target_count)?,
        output: device.allocate(output_count)?,
    }))
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
    fn cpu_track_solves_skin_and_tongue_and_propagates_callback_stop() {
        let mut solvers = BundleSolvers::Cpu(vec![CpuTrackSolvers {
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
