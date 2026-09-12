//! Bounded device-resident BlendShape cache for interactive authoring.

use crate::animation::{
    BlendshapeInvalidationLayer, GpuBlendshapeSolveFence, GpuBlendshapeSolver,
    InteractiveBlendshapeWeights, RegressionGeometry,
};
use crate::common::{Error, Result};
use crate::cuda::{CudaStream, DeviceBuffer, DeviceView, GpuDevice, ensure_same_device};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

pub const DEFAULT_INTERACTIVE_GPU_CACHE_FRAMES: usize = 64;

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

/// Device views produced for one interactive frame.
///
/// The views are valid only for the callback invocation. Consumers may enqueue
/// dependent work on [`Self::stream`]; copying to host is deliberately an
/// explicit operation on [`InteractiveGpuBlendshapeLayer`].
#[derive(Clone, Copy, Debug)]
pub struct InteractiveGpuBlendshapeOutput<'a> {
    pub skin_weights: Option<DeviceView<'a, f32>>,
    pub tongue_weights: Option<DeviceView<'a, f32>>,
    pub stream: &'a CudaStream,
}

impl InteractiveGpuBlendshapeOutput<'_> {
    pub fn skin_weight_count(&self) -> usize {
        self.skin_weights.map_or(0, |weights| weights.len())
    }

    pub fn tongue_weight_count(&self) -> usize {
        self.tongue_weights.map_or(0, |weights| weights.len())
    }
}

struct InteractiveGpuComponent {
    solver: GpuBlendshapeSolver,
    target: DeviceBuffer<f32>,
}

impl InteractiveGpuComponent {
    fn new(
        device: &Arc<GpuDevice>,
        stream: &CudaStream,
        solver: GpuBlendshapeSolver,
    ) -> Result<Self> {
        ensure_same_device(device.id(), stream.device_id())?;
        ensure_same_device(device.id(), solver.device_id())?;
        let target = device.allocate(solver.target_len())?;
        Ok(Self { solver, target })
    }
}

struct CachedGpuFrame {
    skin: Option<DeviceBuffer<f32>>,
    tongue: Option<DeviceBuffer<f32>>,
}

/// Single-track interactive GPU BlendShape layer.
///
/// Random frames are solved with temporal regularization disabled. Ordered
/// all-frame execution resets each solver once and retains its temporal state.
/// At most `cache_capacity` frames own device output buffers; least-recently
/// used frames are synchronized and evicted, and are deterministically
/// recomputed when requested again.
pub struct InteractiveGpuBlendshapeLayer {
    skin: Option<InteractiveGpuComponent>,
    tongue: Option<InteractiveGpuComponent>,
    frames: HashMap<usize, CachedGpuFrame>,
    lru: VecDeque<usize>,
    valid_frames: HashSet<usize>,
    total_frames: Option<usize>,
    next_all_frame: Option<usize>,
    skin_prepare_valid: bool,
    tongue_prepare_valid: bool,
    cache_capacity: usize,
    // Drop the stream after solvers and buffers that may have queued work.
    stream: CudaStream,
    _device: Arc<GpuDevice>,
}

impl InteractiveGpuBlendshapeLayer {
    pub fn new(
        device: Arc<GpuDevice>,
        stream: CudaStream,
        skin: Option<GpuBlendshapeSolver>,
        tongue: Option<GpuBlendshapeSolver>,
        cache_capacity: usize,
    ) -> Result<Self> {
        if cache_capacity == 0 {
            return Err(invalid(
                "interactive GPU BlendShape cache capacity must be positive",
            ));
        }
        ensure_same_device(device.id(), stream.device_id())?;
        let skin = skin
            .map(|solver| InteractiveGpuComponent::new(&device, &stream, solver))
            .transpose()?;
        let tongue = tongue
            .map(|solver| InteractiveGpuComponent::new(&device, &stream, solver))
            .transpose()?;
        Ok(Self {
            skin,
            tongue,
            frames: HashMap::new(),
            lru: VecDeque::new(),
            valid_frames: HashSet::new(),
            total_frames: None,
            next_all_frame: None,
            skin_prepare_valid: true,
            tongue_prepare_valid: true,
            cache_capacity,
            stream,
            _device: device,
        })
    }

    pub fn with_default_cache(
        device: Arc<GpuDevice>,
        stream: CudaStream,
        skin: Option<GpuBlendshapeSolver>,
        tongue: Option<GpuBlendshapeSolver>,
    ) -> Result<Self> {
        Self::new(
            device,
            stream,
            skin,
            tongue,
            DEFAULT_INTERACTIVE_GPU_CACHE_FRAMES,
        )
    }

    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }

    pub fn skin_solver(&self) -> Option<&GpuBlendshapeSolver> {
        self.skin.as_ref().map(|component| &component.solver)
    }

    pub fn tongue_solver(&self) -> Option<&GpuBlendshapeSolver> {
        self.tongue.as_ref().map(|component| &component.solver)
    }

    /// Invalidates Skin Prepare and all cached weights before returning it.
    pub fn skin_solver_mut(&mut self) -> Option<&mut GpuBlendshapeSolver> {
        self.invalidate(BlendshapeInvalidationLayer::SkinSolverPrepare);
        self.skin.as_mut().map(|component| &mut component.solver)
    }

    /// Invalidates Tongue Prepare and all cached weights before returning it.
    pub fn tongue_solver_mut(&mut self) -> Option<&mut GpuBlendshapeSolver> {
        self.invalidate(BlendshapeInvalidationLayer::TongueSolverPrepare);
        self.tongue.as_mut().map(|component| &mut component.solver)
    }

    pub fn cache_capacity(&self) -> usize {
        self.cache_capacity
    }

    pub fn cached_frame_count(&self) -> usize {
        self.frames.len()
    }

    pub fn is_frame_cached(&self, frame: usize) -> bool {
        self.frames.contains_key(&frame)
    }

    pub fn is_frame_valid(&self, frame: usize) -> bool {
        self.valid_frames.contains(&frame)
    }

    pub fn invalidate(&mut self, layer: BlendshapeInvalidationLayer) {
        match layer {
            BlendshapeInvalidationLayer::None => {}
            BlendshapeInvalidationLayer::All => {
                self.skin_prepare_valid = self.skin.is_none();
                self.tongue_prepare_valid = self.tongue.is_none();
                self.clear_weights();
            }
            BlendshapeInvalidationLayer::SkinSolverPrepare => {
                self.skin_prepare_valid = self.skin.is_none();
                self.clear_weights();
            }
            BlendshapeInvalidationLayer::TongueSolverPrepare => {
                self.tongue_prepare_valid = self.tongue.is_none();
                self.clear_weights();
            }
            BlendshapeInvalidationLayer::Weights => self.clear_weights(),
        }
    }

    pub fn invalidate_geometry(&mut self) {
        self.invalidate(BlendshapeInvalidationLayer::Weights);
    }

    pub fn invalidate_geometry_frame(&mut self, frame: usize) {
        if self.frames.contains_key(&frame) {
            let _ = self.stream.synchronize();
            self.frames.remove(&frame);
            self.lru.retain(|cached| *cached != frame);
        }
        self.valid_frames.remove(&frame);
        self.next_all_frame = None;
    }

    pub fn is_valid(&self, layer: BlendshapeInvalidationLayer) -> bool {
        let weights_valid = self
            .total_frames
            .is_some_and(|total| total != 0 && self.valid_frames.len() == total);
        match layer {
            BlendshapeInvalidationLayer::None => true,
            BlendshapeInvalidationLayer::All => {
                self.skin_prepare_valid && self.tongue_prepare_valid && weights_valid
            }
            BlendshapeInvalidationLayer::SkinSolverPrepare => self.skin_prepare_valid,
            BlendshapeInvalidationLayer::TongueSolverPrepare => self.tongue_prepare_valid,
            BlendshapeInvalidationLayer::Weights => weights_valid,
        }
    }

    /// Computes or replays one stateless random-access frame.
    pub fn compute_frame<C>(
        &mut self,
        frame: usize,
        total_frames: usize,
        geometry: &RegressionGeometry,
        mut callback: C,
    ) -> Result<bool>
    where
        C: for<'output> FnMut(InteractiveGpuBlendshapeOutput<'output>) -> bool,
    {
        if frame >= total_frames {
            return Err(invalid("interactive GPU BlendShape frame is out of range"));
        }
        self.set_total_frames(total_frames);
        if self.frames.contains_key(&frame) {
            self.touch(frame);
            let cached = self.frames.get(&frame).expect("checked GPU cache entry");
            return Ok(callback(InteractiveGpuBlendshapeOutput {
                skin_weights: cached.skin.as_ref().map(DeviceBuffer::view),
                tongue_weights: cached.tongue.as_ref().map(DeviceBuffer::view),
                stream: &self.stream,
            }));
        }
        self.solve_frame(frame, geometry, true, callback)
    }

    /// Clears the weight cache and resets temporal state for an ordered pass.
    pub fn begin_all_frames(&mut self, total_frames: usize) -> Result<()> {
        self.clear_weights();
        self.total_frames = Some(total_frames);
        self.next_all_frame = Some(0);
        if let Some(component) = &mut self.skin {
            component.solver.reset(&self.stream)?;
            self.skin_prepare_valid = true;
        }
        if let Some(component) = &mut self.tongue {
            component.solver.reset(&self.stream)?;
            self.tongue_prepare_valid = true;
        }
        Ok(())
    }

    /// Computes the next frame of an ordered all-frame pass.
    pub fn compute_next_frame<C>(
        &mut self,
        frame: usize,
        geometry: &RegressionGeometry,
        callback: C,
    ) -> Result<bool>
    where
        C: for<'output> FnMut(InteractiveGpuBlendshapeOutput<'output>) -> bool,
    {
        let total_frames = self
            .total_frames
            .ok_or_else(|| invalid("interactive GPU all-frame pass has not begun"))?;
        if frame >= total_frames || self.next_all_frame != Some(frame) {
            return Err(invalid(
                "interactive GPU all-frame execution must use increasing frame order",
            ));
        }
        let keep_going = self.solve_frame(frame, geometry, false, callback)?;
        self.next_all_frame = keep_going.then_some(frame + 1);
        Ok(keep_going)
    }

    /// Copies a cached frame to host and synchronizes the layer stream.
    ///
    /// An evicted or never-computed frame returns `None` and can be recomputed
    /// through [`Self::compute_frame`].
    pub fn copy_cached_frame_to_host(
        &self,
        frame: usize,
    ) -> Result<Option<InteractiveBlendshapeWeights>> {
        let Some(cached) = self.frames.get(&frame) else {
            return Ok(None);
        };
        let mut skin = vec![0.0; cached.skin.as_ref().map_or(0, DeviceBuffer::len)];
        let mut tongue = vec![0.0; cached.tongue.as_ref().map_or(0, DeviceBuffer::len)];
        if let Some(buffer) = &cached.skin {
            buffer.copy_to(&mut skin, &self.stream)?;
        }
        if let Some(buffer) = &cached.tongue {
            buffer.copy_to(&mut tongue, &self.stream)?;
        }
        Ok(Some(InteractiveBlendshapeWeights { skin, tongue }))
    }

    pub fn wait(&self) -> Result<()> {
        self.stream.synchronize()
    }

    fn solve_frame<C>(
        &mut self,
        frame: usize,
        geometry: &RegressionGeometry,
        stateless: bool,
        mut callback: C,
    ) -> Result<bool>
    where
        C: for<'output> FnMut(InteractiveGpuBlendshapeOutput<'output>) -> bool,
    {
        self.evict_if_full()?;
        let mut cached = CachedGpuFrame {
            skin: self
                .skin
                .as_ref()
                .map(|component| self._device.allocate(component.solver.pose_count()))
                .transpose()?,
            tongue: self
                .tongue
                .as_ref()
                .map(|component| self._device.allocate(component.solver.pose_count()))
                .transpose()?,
        };
        let skin_fence = enqueue_component(
            self.skin.as_mut(),
            cached.skin.as_mut(),
            &geometry.skin,
            &self.stream,
            stateless,
        )?;
        let tongue_fence = enqueue_component(
            self.tongue.as_mut(),
            cached.tongue.as_mut(),
            &geometry.tongue,
            &self.stream,
            stateless,
        )?;
        self.skin_prepare_valid = true;
        self.tongue_prepare_valid = true;
        let keep_going = callback(InteractiveGpuBlendshapeOutput {
            skin_weights: skin_fence.as_ref().map(GpuBlendshapeSolveFence::output),
            tongue_weights: tongue_fence.as_ref().map(GpuBlendshapeSolveFence::output),
            stream: &self.stream,
        });
        drop(tongue_fence);
        drop(skin_fence);
        self.frames.insert(frame, cached);
        self.valid_frames.insert(frame);
        self.touch(frame);
        Ok(keep_going)
    }

    fn set_total_frames(&mut self, total_frames: usize) {
        if self.total_frames != Some(total_frames) {
            self.clear_weights();
            self.total_frames = Some(total_frames);
        }
        self.next_all_frame = None;
    }

    fn evict_if_full(&mut self) -> Result<()> {
        if self.frames.len() < self.cache_capacity {
            return Ok(());
        }
        self.stream.synchronize()?;
        if let Some(frame) = self.lru.pop_front() {
            self.frames.remove(&frame);
        }
        Ok(())
    }

    fn touch(&mut self, frame: usize) {
        self.lru.retain(|cached| *cached != frame);
        self.lru.push_back(frame);
    }

    fn clear_weights(&mut self) {
        if !self.frames.is_empty() {
            let _ = self.stream.synchronize();
        }
        self.frames.clear();
        self.lru.clear();
        self.valid_frames.clear();
        self.next_all_frame = None;
    }
}

impl Drop for InteractiveGpuBlendshapeLayer {
    fn drop(&mut self) {
        // Callback consumers may have enqueued work after the solve event. Keep
        // every cached output and solver alive until that stream work finishes.
        let _ = self.stream.synchronize();
    }
}

fn enqueue_component<'a>(
    component: Option<&'a mut InteractiveGpuComponent>,
    output: Option<&'a mut DeviceBuffer<f32>>,
    geometry: &[f32],
    stream: &'a CudaStream,
    stateless: bool,
) -> Result<Option<GpuBlendshapeSolveFence<'a>>> {
    let (Some(component), Some(output)) = (component, output) else {
        return Ok(None);
    };
    if geometry.len() != component.solver.target_len() {
        return Err(invalid(
            "interactive GPU BlendShape geometry dimension does not match solver",
        ));
    }
    if stateless {
        component.solver.reset(stream)?;
    }
    component.target.copy_from(geometry, stream)?;
    if stateless {
        component
            .solver
            .solve_stateless_async(&component.target, output, stream)
            .map(Some)
    } else {
        component
            .solver
            .solve_async(&component.target, output, stream)
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{BlendshapeData, CpuBlendshapeSolver, EyesRotation};
    use crate::common::BlendshapeConfig;

    fn data() -> BlendshapeData {
        BlendshapeData {
            neutral_pose: vec![0.0, 0.0, 0.0, 2.0, 3.0, 4.0],
            delta_poses: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0,
            ],
            pose_names: vec!["x".into(), "y".into(), "z".into()],
            pose_mask: None,
        }
    }

    fn config(temporal_regularization: f32) -> BlendshapeConfig {
        BlendshapeConfig {
            l2_regularization: 0.01,
            temporal_regularization,
            l1_regularization: 0.0,
            symmetry_regularization: 0.0,
            num_poses: 3,
            active_poses: vec![1, 1, 1],
            cancel_poses: vec![-1, -1, -1],
            symmetry_poses: vec![-1, -1, -1],
            multipliers: vec![1.0; 3],
            offsets: vec![0.0; 3],
            template_bb_size: 54.7,
            tolerance: 1.0e-10,
        }
    }

    fn geometry(weights: [f32; 3]) -> RegressionGeometry {
        let pose = data().evaluate_pose(&weights).unwrap();
        RegressionGeometry {
            skin: pose.clone(),
            tongue: pose,
            jaw_transform: [0.0; 16],
            eyes_rotation: EyesRotation {
                right: [0.0; 3],
                left: [0.0; 3],
            },
        }
    }

    fn gpu_layer(capacity: usize, temporal: f32) -> InteractiveGpuBlendshapeLayer {
        let device = GpuDevice::new(0).unwrap();
        let stream = device.create_stream().unwrap();
        let skin = GpuBlendshapeSolver::new(&device, &stream, data(), &config(temporal)).unwrap();
        let tongue = GpuBlendshapeSolver::new(&device, &stream, data(), &config(temporal)).unwrap();
        InteractiveGpuBlendshapeLayer::new(device, stream, Some(skin), Some(tongue), capacity)
            .unwrap()
    }

    fn capture(
        layer: &mut InteractiveGpuBlendshapeLayer,
        frame: usize,
        total: usize,
        geometry: &RegressionGeometry,
    ) -> InteractiveBlendshapeWeights {
        let mut captured = None;
        layer
            .compute_frame(frame, total, geometry, |output| {
                let mut skin = vec![0.0; output.skin_weight_count()];
                let mut tongue = vec![0.0; output.tongue_weight_count()];
                output
                    .skin_weights
                    .unwrap()
                    .copy_to(&mut skin, output.stream)
                    .unwrap();
                output
                    .tongue_weights
                    .unwrap()
                    .copy_to(&mut tongue, output.stream)
                    .unwrap();
                captured = Some(InteractiveBlendshapeWeights { skin, tongue });
                true
            })
            .unwrap();
        captured.unwrap()
    }

    #[test]
    fn random_frames_are_stateless_and_evicted_frames_recompute() {
        let mut layer = gpu_layer(2, 100.0);
        let first_geometry = geometry([0.2, 0.4, 0.6]);
        let first = capture(&mut layer, 0, 3, &first_geometry);
        // A random solve must match a solver prepared with TemporalReg=0,
        // not merely clear the previous-weight contribution on the RHS.
        let mut zero_temporal = gpu_layer(2, 0.0);
        let expected = capture(&mut zero_temporal, 0, 3, &first_geometry);
        for (actual, expected) in first
            .skin
            .iter()
            .chain(&first.tongue)
            .zip(expected.skin.iter().chain(&expected.tongue))
        {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
        capture(&mut layer, 1, 3, &geometry([0.3, 0.5, 0.7]));
        capture(&mut layer, 2, 3, &geometry([0.4, 0.6, 0.8]));
        assert_eq!(layer.cached_frame_count(), 2);
        assert!(!layer.is_frame_cached(0));
        assert!(layer.is_valid(BlendshapeInvalidationLayer::Weights));

        let replay = capture(&mut layer, 0, 3, &first_geometry);
        assert_eq!(first, replay);
        assert_eq!(layer.cached_frame_count(), 2);
    }

    #[test]
    fn invalidation_matches_cpu_layer_dependencies() {
        let mut layer = gpu_layer(2, 1.0);
        capture(&mut layer, 0, 1, &geometry([0.2, 0.4, 0.6]));
        assert!(layer.is_valid(BlendshapeInvalidationLayer::All));

        layer.invalidate(BlendshapeInvalidationLayer::Weights);
        assert!(layer.is_valid(BlendshapeInvalidationLayer::SkinSolverPrepare));
        assert!(layer.is_valid(BlendshapeInvalidationLayer::TongueSolverPrepare));
        assert!(!layer.is_valid(BlendshapeInvalidationLayer::Weights));

        layer.invalidate(BlendshapeInvalidationLayer::SkinSolverPrepare);
        assert!(!layer.is_valid(BlendshapeInvalidationLayer::SkinSolverPrepare));
        assert!(layer.is_valid(BlendshapeInvalidationLayer::TongueSolverPrepare));
        capture(&mut layer, 0, 1, &geometry([0.2, 0.4, 0.6]));
        assert!(layer.is_valid(BlendshapeInvalidationLayer::All));
    }

    #[test]
    fn ordered_gpu_results_match_cpu_within_solver_tolerance() {
        let temporal = 2.0;
        let config = config(temporal);
        let mut cpu_solver = CpuBlendshapeSolver::from_config(data(), &config).unwrap();
        cpu_solver.prepare().unwrap();
        let mut cpu = crate::animation::InteractiveBlendshapeLayer::new(
            Some(cpu_solver.clone()),
            Some(cpu_solver),
        );
        let geometry = [
            geometry([0.2, 0.4, 0.6]),
            geometry([0.7, 0.5, 0.3]),
            geometry([0.4, 0.6, 0.8]),
        ];
        let expected = cpu.compute_all_frames(&geometry).unwrap();

        let mut gpu = gpu_layer(2, temporal);
        gpu.begin_all_frames(geometry.len()).unwrap();
        let mut actual = Vec::new();
        for (frame, geometry) in geometry.iter().enumerate() {
            gpu.compute_next_frame(frame, geometry, |output| {
                let mut skin = vec![0.0; output.skin_weight_count()];
                let mut tongue = vec![0.0; output.tongue_weight_count()];
                output
                    .skin_weights
                    .unwrap()
                    .copy_to(&mut skin, output.stream)
                    .unwrap();
                output
                    .tongue_weights
                    .unwrap()
                    .copy_to(&mut tongue, output.stream)
                    .unwrap();
                actual.push(InteractiveBlendshapeWeights { skin, tongue });
                true
            })
            .unwrap();
        }
        assert!(gpu.is_valid(BlendshapeInvalidationLayer::All));
        assert_eq!(gpu.cached_frame_count(), 2);
        for (expected, actual) in expected.iter().zip(actual) {
            for (expected, actual) in expected.skin.iter().zip(actual.skin) {
                assert!((expected - actual).abs() < 0.03);
            }
            for (expected, actual) in expected.tongue.iter().zip(actual.tongue) {
                assert!((expected - actual).abs() < 0.03);
            }
        }
    }
}
