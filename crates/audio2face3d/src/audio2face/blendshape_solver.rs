//! SDK-facing host BlendShape solver factory.

use crate::animation::{BlendshapeData, BlendshapeSolverParameters, CpuBlendshapeSolver};
use crate::audio2face::{BlendshapeSolveComponentParameters, BlendshapeSolverParams};
use crate::audio2x::Result;

/// Creates an owning host BlendShape solver from borrowed SDK-facing views.
pub fn create_blendshape_solver(
    component: BlendshapeSolveComponentParameters<'_>,
) -> Result<CpuBlendshapeSolver> {
    let data = BlendshapeData {
        neutral_pose: component.data.neutral_pose.to_vec(),
        delta_poses: component.data.delta_poses.to_vec(),
        pose_names: component
            .data
            .pose_names
            .iter()
            .map(|name| (*name).to_owned())
            .collect(),
        pose_mask: component.data.pose_mask.map(<[usize]>::to_vec),
    };
    let mut solver = CpuBlendshapeSolver::new(data)?;
    solver.set_parameters(component.params.into())?;
    solver.set_active_poses(component.config.active_poses.to_vec())?;
    solver.set_cancel_poses(component.config.cancel_poses.to_vec())?;
    solver.set_symmetry_poses(component.config.symmetry_poses.to_vec())?;
    solver.set_multipliers(component.config.multipliers.to_vec())?;
    solver.set_offsets(component.config.offsets.to_vec())?;
    Ok(solver)
}

impl From<BlendshapeSolverParams> for BlendshapeSolverParameters {
    fn from(value: BlendshapeSolverParams) -> Self {
        Self {
            l1_regularization: value.l1_regularization,
            l2_regularization: value.l2_regularization,
            symmetry_regularization: value.symmetry_regularization,
            temporal_regularization: value.temporal_regularization,
            template_bb_size: value.template_bounding_box_size,
            tolerance: value.tolerance,
        }
    }
}

impl From<BlendshapeSolverParameters> for BlendshapeSolverParams {
    fn from(value: BlendshapeSolverParameters) -> Self {
        Self {
            l1_regularization: value.l1_regularization,
            l2_regularization: value.l2_regularization,
            symmetry_regularization: value.symmetry_regularization,
            temporal_regularization: value.temporal_regularization,
            template_bounding_box_size: value.template_bb_size,
            tolerance: value.tolerance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio2face::{
        BlendshapeSolverConfigView, BlendshapeSolverDataView, BlendshapeSolverParams,
    };

    #[test]
    fn factory_copies_borrowed_views_into_an_owning_solver() {
        let names = ["smile"];
        let component = BlendshapeSolveComponentParameters {
            params: BlendshapeSolverParams::default(),
            config: BlendshapeSolverConfigView {
                active_poses: &[1],
                cancel_poses: &[-1],
                symmetry_poses: &[-1],
                multipliers: &[1.0],
                offsets: &[0.0],
            },
            data: BlendshapeSolverDataView {
                neutral_pose: &[0.0, 0.0, 0.0],
                delta_poses: &[1.0, 0.0, 0.0],
                pose_mask: None,
                pose_names: &names,
            },
        };
        let solver = create_blendshape_solver(component).unwrap();
        assert_eq!(solver.pose_name(0), Some("smile"));
    }
}
