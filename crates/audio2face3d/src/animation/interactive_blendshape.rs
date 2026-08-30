//! Random-access CPU BlendShape layer for interactive geometry output.

use crate::animation::{CpuBlendshapeSolver, RegressionGeometry};
use crate::common::{Error, Result};

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

/// Original SDK-compatible BlendShape invalidation identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum BlendshapeInvalidationLayer {
    None = 0,
    All = 1,
    SkinSolverPrepare = 101,
    TongueSolverPrepare = 102,
    Weights = 103,
}

/// Skin and tongue weights for one interactive frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InteractiveBlendshapeWeights {
    pub skin: Vec<f32>,
    pub tongue: Vec<f32>,
}

/// Single-track interactive BlendShape cache.
///
/// [`Self::compute_frame`] uses solver clones with temporal regularization set
/// to zero, so random access does not depend on the previously requested
/// frame. [`Self::compute_all_frames`] resets and advances the owned solvers in
/// order, preserving their configured temporal regularization.
pub struct InteractiveBlendshapeLayer {
    skin: Option<CpuBlendshapeSolver>,
    tongue: Option<CpuBlendshapeSolver>,
    skin_prepare_valid: bool,
    tongue_prepare_valid: bool,
    frames: Vec<Option<InteractiveBlendshapeWeights>>,
}

impl InteractiveBlendshapeLayer {
    pub fn new(skin: Option<CpuBlendshapeSolver>, tongue: Option<CpuBlendshapeSolver>) -> Self {
        let skin_prepare_valid = skin.as_ref().is_none_or(CpuBlendshapeSolver::is_prepared);
        let tongue_prepare_valid = tongue.as_ref().is_none_or(CpuBlendshapeSolver::is_prepared);
        Self {
            skin,
            tongue,
            skin_prepare_valid,
            tongue_prepare_valid,
            frames: Vec::new(),
        }
    }

    pub fn skin_solver(&self) -> Option<&CpuBlendshapeSolver> {
        self.skin.as_ref()
    }

    pub fn tongue_solver(&self) -> Option<&CpuBlendshapeSolver> {
        self.tongue.as_ref()
    }

    /// Invalidates Skin Prepare and all cached weights before returning the solver.
    pub fn skin_solver_mut(&mut self) -> Option<&mut CpuBlendshapeSolver> {
        self.invalidate(BlendshapeInvalidationLayer::SkinSolverPrepare);
        self.skin.as_mut()
    }

    /// Invalidates Tongue Prepare and all cached weights before returning the solver.
    pub fn tongue_solver_mut(&mut self) -> Option<&mut CpuBlendshapeSolver> {
        self.invalidate(BlendshapeInvalidationLayer::TongueSolverPrepare);
        self.tongue.as_mut()
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

    /// Invalidates the dependent weight layer after any geometry-layer change.
    pub fn invalidate_geometry(&mut self) {
        self.invalidate(BlendshapeInvalidationLayer::Weights);
    }

    /// Invalidates only one frame after its geometry changes.
    pub fn invalidate_geometry_frame(&mut self, frame: usize) {
        if let Some(value) = self.frames.get_mut(frame) {
            *value = None;
        }
    }

    pub fn is_valid(&self, layer: BlendshapeInvalidationLayer) -> bool {
        let weights_valid = !self.frames.is_empty() && self.frames.iter().all(Option::is_some);
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

    pub fn cached_frame(&self, frame: usize) -> Option<&InteractiveBlendshapeWeights> {
        self.frames.get(frame).and_then(Option::as_ref)
    }

    pub fn compute_frame(
        &mut self,
        frame: usize,
        total_frames: usize,
        geometry: &RegressionGeometry,
    ) -> Result<InteractiveBlendshapeWeights> {
        if frame >= total_frames {
            return Err(invalid("interactive blendshape frame is out of range"));
        }
        self.resize_frames(total_frames);
        if let Some(weights) = &self.frames[frame] {
            return Ok(weights.clone());
        }
        let skin = solve_stateless(self.skin.as_ref(), &geometry.skin)?;
        let tongue = solve_stateless(self.tongue.as_ref(), &geometry.tongue)?;
        let weights = InteractiveBlendshapeWeights { skin, tongue };
        self.frames[frame] = Some(weights.clone());
        Ok(weights)
    }

    pub fn compute_all_frames(
        &mut self,
        geometry: &[RegressionGeometry],
    ) -> Result<Vec<InteractiveBlendshapeWeights>> {
        self.begin_all_frames(geometry.len())?;
        let mut output = Vec::with_capacity(geometry.len());
        for (frame, geometry) in geometry.iter().enumerate() {
            output.push(self.compute_next_frame(frame, geometry)?);
        }
        Ok(output)
    }

    /// Resets temporal solver state before ordered full-frame computation.
    pub fn begin_all_frames(&mut self, total_frames: usize) -> Result<()> {
        self.frames.clear();
        self.frames.resize_with(total_frames, || None);
        prepare_owned(&mut self.skin, &mut self.skin_prepare_valid)?;
        prepare_owned(&mut self.tongue, &mut self.tongue_prepare_valid)?;
        if let Some(solver) = &mut self.skin {
            solver.reset();
        }
        if let Some(solver) = &mut self.tongue {
            solver.reset();
        }
        Ok(())
    }

    /// Computes one frame in increasing order after [`Self::begin_all_frames`].
    pub fn compute_next_frame(
        &mut self,
        frame: usize,
        geometry: &RegressionGeometry,
    ) -> Result<InteractiveBlendshapeWeights> {
        if frame >= self.frames.len() {
            return Err(invalid("interactive blendshape frame is out of range"));
        }
        let weights = InteractiveBlendshapeWeights {
            skin: solve_owned(self.skin.as_mut(), &geometry.skin)?,
            tongue: solve_owned(self.tongue.as_mut(), &geometry.tongue)?,
        };
        self.frames[frame] = Some(weights.clone());
        Ok(weights)
    }

    fn resize_frames(&mut self, total_frames: usize) {
        if self.frames.len() != total_frames {
            self.frames.resize_with(total_frames, || None);
        }
    }

    fn clear_weights(&mut self) {
        for frame in &mut self.frames {
            *frame = None;
        }
    }
}

fn prepare_owned(solver: &mut Option<CpuBlendshapeSolver>, valid: &mut bool) -> Result<()> {
    if let Some(solver) = solver
        && !*valid
    {
        solver.prepare()?;
        *valid = true;
    }
    Ok(())
}

fn solve_owned(solver: Option<&mut CpuBlendshapeSolver>, target: &[f32]) -> Result<Vec<f32>> {
    solver
        .map(|solver| solver.solve(target))
        .transpose()
        .map(Option::unwrap_or_default)
}

fn solve_stateless(solver: Option<&CpuBlendshapeSolver>, target: &[f32]) -> Result<Vec<f32>> {
    let Some(solver) = solver else {
        return Ok(Vec::new());
    };
    let mut solver = solver.clone();
    let mut parameters = solver.parameters();
    parameters.temporal_regularization = 0.0;
    solver.set_parameters(parameters)?;
    solver.prepare()?;
    solver.reset();
    solver.solve(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{BlendshapeData, EyesRotation};

    fn solver(temporal_regularization: f32) -> CpuBlendshapeSolver {
        let data = BlendshapeData {
            neutral_pose: vec![0.0, 0.0, 0.0],
            delta_poses: vec![1.0, 0.0, 0.0],
            pose_names: vec!["pose".into()],
            pose_mask: None,
        };
        let mut solver = CpuBlendshapeSolver::new(data).unwrap();
        let mut parameters = solver.parameters();
        parameters.temporal_regularization = temporal_regularization;
        solver.set_parameters(parameters).unwrap();
        solver.prepare().unwrap();
        solver
    }

    fn geometry(value: f32) -> RegressionGeometry {
        RegressionGeometry {
            skin: vec![value, 0.0, 0.0],
            tongue: vec![value, 0.0, 0.0],
            jaw_transform: [0.0; 16],
            eyes_rotation: EyesRotation {
                right: [0.0; 3],
                left: [0.0; 3],
            },
        }
    }

    #[test]
    fn random_access_cache_and_invalidation_follow_original_layers() {
        let mut layer = InteractiveBlendshapeLayer::new(Some(solver(100.0)), Some(solver(100.0)));
        let first = layer.compute_frame(1, 2, &geometry(0.75)).unwrap();
        let replay = layer.compute_frame(1, 2, &geometry(0.1)).unwrap();
        assert_eq!(first, replay);
        assert!(!layer.is_valid(BlendshapeInvalidationLayer::Weights));

        layer.compute_frame(0, 2, &geometry(0.25)).unwrap();
        assert!(layer.is_valid(BlendshapeInvalidationLayer::All));
        layer.invalidate_geometry_frame(1);
        assert!(!layer.is_valid(BlendshapeInvalidationLayer::Weights));
        assert!(layer.is_valid(BlendshapeInvalidationLayer::SkinSolverPrepare));

        let _ = layer.skin_solver_mut().unwrap();
        assert!(!layer.is_valid(BlendshapeInvalidationLayer::SkinSolverPrepare));
        assert!(!layer.is_valid(BlendshapeInvalidationLayer::Weights));
    }

    #[test]
    fn all_frames_prepare_and_advance_temporal_state() {
        let mut layer = InteractiveBlendshapeLayer::new(Some(solver(100.0)), None);
        layer.invalidate(BlendshapeInvalidationLayer::SkinSolverPrepare);
        let output = layer
            .compute_all_frames(&[geometry(0.2), geometry(0.8)])
            .unwrap();
        assert_eq!(output.len(), 2);
        assert!(layer.is_valid(BlendshapeInvalidationLayer::All));
    }
}
