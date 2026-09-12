//! Blendshape data preparation and the SDK-compatible CPU solver path.

use crate::common::{BlendshapeConfig, Error, NpzArchive, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

mod bvls;
#[cfg(all(feature = "cuda", any(test, feature = "tensorrt")))]
pub(crate) mod rhs;
#[cfg(all(test, feature = "cuda"))]
mod sdk_tests;

fn invalid(message: impl Into<String>) -> Error {
    Error::InvalidSchema(message.into())
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlendshapeData {
    pub neutral_pose: Vec<f32>,
    /// Column-major deltas: every complete pose is contiguous.
    pub delta_poses: Vec<f32>,
    pub pose_names: Vec<String>,
    pub pose_mask: Option<Vec<usize>>,
}

impl BlendshapeData {
    /// Loads the original SDK NPZ layout (`poseNames`, `neutral`, poses, optional mask).
    pub fn load_npz(path: impl AsRef<Path>) -> Result<Self> {
        let mut archive = NpzArchive::open(path.as_ref())?;
        let mut names = archive.strings("poseNames")?;
        if names.first().map(String::as_str) != Some("neutral") {
            return Err(invalid("blendshape poseNames must begin with neutral"));
        }
        let neutral_pose = archive.f32("neutral")?;
        names.remove(0);
        let mut delta_poses = Vec::with_capacity(
            neutral_pose
                .len()
                .checked_mul(names.len())
                .ok_or_else(|| invalid("blendshape NPZ dimensions overflow"))?,
        );
        for name in &names {
            let pose = archive.f32(name)?;
            if pose.len() != neutral_pose.len() {
                return Err(invalid(format!(
                    "blendshape pose {name} dimensions do not match neutral"
                )));
            }
            delta_poses.extend(pose);
        }
        let pose_mask = archive
            .contains("frontalMask")
            .then(|| archive.i32("frontalMask"))
            .transpose()?
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| {
                        usize::try_from(value).map_err(|_| {
                            invalid("blendshape frontalMask contains a negative index")
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        let data = Self {
            neutral_pose,
            delta_poses,
            pose_names: names,
            pose_mask,
        };
        data.validate()?;
        Ok(data)
    }

    pub fn validate(&self) -> Result<()> {
        if self.neutral_pose.is_empty() || !self.neutral_pose.len().is_multiple_of(3) {
            return Err(invalid(
                "blendshape neutral pose must contain complete xyz vertices",
            ));
        }
        if self.pose_names.is_empty()
            || self.delta_poses.len()
                != self
                    .neutral_pose
                    .len()
                    .checked_mul(self.pose_names.len())
                    .ok_or_else(|| invalid("blendshape data dimensions overflow"))?
        {
            return Err(invalid(
                "blendshape delta dimensions do not match pose names",
            ));
        }
        if self
            .neutral_pose
            .iter()
            .chain(&self.delta_poses)
            .any(|value| !value.is_finite())
        {
            return Err(invalid("blendshape geometry must be finite"));
        }
        if let Some(mask) = &self.pose_mask {
            let vertex_count = self.neutral_pose.len() / 3;
            let unique = mask.iter().copied().collect::<HashSet<_>>();
            if mask.is_empty()
                || unique.len() != mask.len()
                || unique.iter().any(|i| *i >= vertex_count)
            {
                return Err(invalid(
                    "blendshape pose mask must contain unique in-range vertices",
                ));
            }
        }
        Ok(())
    }

    pub fn pose_count(&self) -> usize {
        self.pose_names.len()
    }

    pub fn evaluate_pose(&self, weights: &[f32]) -> Result<Vec<f32>> {
        self.validate()?;
        if weights.len() != self.pose_count() || weights.iter().any(|value| !value.is_finite()) {
            return Err(invalid("blendshape weight dimensions/values are invalid"));
        }
        let coordinates = self.neutral_pose.len();
        let mut result = self.neutral_pose.clone();
        for (pose, weight) in weights.iter().copied().enumerate() {
            let delta = &self.delta_poses[pose * coordinates..(pose + 1) * coordinates];
            result
                .iter_mut()
                .zip(delta)
                .for_each(|(output, delta)| *output += delta * weight);
        }
        Ok(result)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendshapeSolverParameters {
    pub l1_regularization: f32,
    pub l2_regularization: f32,
    pub symmetry_regularization: f32,
    pub temporal_regularization: f32,
    pub template_bb_size: f32,
    pub tolerance: f32,
}

impl Default for BlendshapeSolverParameters {
    fn default() -> Self {
        Self {
            l1_regularization: 1.0,
            l2_regularization: 3.5,
            symmetry_regularization: 100.0,
            temporal_regularization: 0.0,
            template_bb_size: 54.7,
            tolerance: 1.0e-10,
        }
    }
}

impl From<&BlendshapeConfig> for BlendshapeSolverParameters {
    fn from(config: &BlendshapeConfig) -> Self {
        Self {
            l1_regularization: config.l1_regularization,
            l2_regularization: config.l2_regularization,
            symmetry_regularization: config.symmetry_regularization,
            temporal_regularization: config.temporal_regularization,
            template_bb_size: config.template_bb_size,
            tolerance: config.tolerance,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PreparedBlendshape {
    pub(crate) coordinate_indices: Vec<usize>,
    pub(crate) active_indices: Vec<usize>,
    pub(crate) active_deltas: Vec<f32>,
    pub(crate) active_neutral: Vec<f32>,
    #[cfg(feature = "cuda")]
    pub(crate) matrix: Vec<f64>,
    pub(crate) cpu_matrix: Vec<f32>,
    pub(crate) cancel_pairs: Vec<(usize, usize)>,
    pub(crate) scale_factor: f64,
    previous_weights: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct CpuBlendshapeSolver {
    pub(crate) data: BlendshapeData,
    active_poses: Vec<i32>,
    cancel_poses: Vec<i32>,
    symmetry_poses: Vec<i32>,
    pub(crate) multipliers: Vec<f32>,
    pub(crate) offsets: Vec<f32>,
    pub(crate) parameters: BlendshapeSolverParameters,
    pub(crate) prepared: Option<PreparedBlendshape>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendshapeSolverKind {
    Cpu,
    Gpu,
}

impl BlendshapeSolverKind {
    pub const fn from_gpu_flag(use_gpu_solver: bool) -> Self {
        if use_gpu_solver { Self::Gpu } else { Self::Cpu }
    }
}

impl CpuBlendshapeSolver {
    pub fn new(data: BlendshapeData) -> Result<Self> {
        data.validate()?;
        let count = data.pose_count();
        Ok(Self {
            data,
            active_poses: vec![1; count],
            cancel_poses: vec![-1; count],
            symmetry_poses: vec![-1; count],
            multipliers: vec![1.0; count],
            offsets: vec![0.0; count],
            parameters: BlendshapeSolverParameters::default(),
            prepared: None,
        })
    }

    pub fn from_config(data: BlendshapeData, config: &BlendshapeConfig) -> Result<Self> {
        config.validate()?;
        if data.pose_count() != config.num_poses {
            return Err(invalid("blendshape data/config pose counts do not match"));
        }
        let mut solver = Self::new(data)?;
        solver.active_poses.clone_from(&config.active_poses);
        solver.cancel_poses.clone_from(&config.cancel_poses);
        solver.symmetry_poses.clone_from(&config.symmetry_poses);
        solver.multipliers.clone_from(&config.multipliers);
        solver.offsets.clone_from(&config.offsets);
        solver.parameters = BlendshapeSolverParameters::from(config);
        Ok(solver)
    }

    pub fn data(&self) -> &BlendshapeData {
        &self.data
    }

    pub fn parameters(&self) -> BlendshapeSolverParameters {
        self.parameters
    }

    pub fn pose_name(&self, index: usize) -> Option<&str> {
        self.data.pose_names.get(index).map(String::as_str)
    }

    pub fn pose_index(&self, name: &str) -> Option<usize> {
        self.data.pose_names.iter().position(|value| value == name)
    }

    pub fn active_poses(&self) -> &[i32] {
        &self.active_poses
    }

    pub fn cancel_poses(&self) -> &[i32] {
        &self.cancel_poses
    }

    pub fn symmetry_poses(&self) -> &[i32] {
        &self.symmetry_poses
    }

    pub fn multipliers(&self) -> &[f32] {
        &self.multipliers
    }

    pub fn offsets(&self) -> &[f32] {
        &self.offsets
    }

    pub fn is_prepared(&self) -> bool {
        self.prepared.is_some()
    }

    pub fn set_parameters(&mut self, parameters: BlendshapeSolverParameters) -> Result<()> {
        validate_parameters(parameters)?;
        self.parameters = parameters;
        self.prepared = None;
        Ok(())
    }

    pub fn set_active_poses(&mut self, values: Vec<i32>) -> Result<()> {
        validate_vector_len(self.data.pose_count(), &values, "active poses")?;
        if !values.iter().any(|value| *value != 0) {
            return Err(invalid("blendshape active pose set is empty"));
        }
        self.active_poses = values;
        self.prepared = None;
        Ok(())
    }

    pub fn set_active_pose(&mut self, name: &str, value: i32) -> Result<()> {
        let index = self
            .pose_index(name)
            .ok_or_else(|| invalid(format!("unknown blendshape pose {name}")))?;
        let mut values = self.active_poses.clone();
        values[index] = value;
        self.set_active_poses(values)
    }

    pub fn set_cancel_poses(&mut self, values: Vec<i32>) -> Result<()> {
        validate_vector_len(self.data.pose_count(), &values, "cancel poses")?;
        validate_pairs(&values, "cancel poses")?;
        self.cancel_poses = values;
        self.prepared = None;
        Ok(())
    }

    pub fn set_cancel_pose(&mut self, name: &str, value: i32) -> Result<()> {
        let index = self
            .pose_index(name)
            .ok_or_else(|| invalid(format!("unknown blendshape pose {name}")))?;
        let mut values = self.cancel_poses.clone();
        values[index] = value;
        self.set_cancel_poses(values)
    }

    pub fn set_symmetry_poses(&mut self, values: Vec<i32>) -> Result<()> {
        validate_vector_len(self.data.pose_count(), &values, "symmetry poses")?;
        validate_pairs(&values, "symmetry poses")?;
        self.symmetry_poses = values;
        self.prepared = None;
        Ok(())
    }

    pub fn set_symmetry_pose(&mut self, name: &str, value: i32) -> Result<()> {
        let index = self
            .pose_index(name)
            .ok_or_else(|| invalid(format!("unknown blendshape pose {name}")))?;
        let mut values = self.symmetry_poses.clone();
        values[index] = value;
        self.set_symmetry_poses(values)
    }

    /// Multipliers are applied after solving and therefore do not invalidate Prepare.
    pub fn set_multipliers(&mut self, values: Vec<f32>) -> Result<()> {
        validate_finite_vector(self.data.pose_count(), &values, "multipliers")?;
        self.multipliers = values;
        Ok(())
    }

    pub fn set_multiplier(&mut self, name: &str, value: f32) -> Result<()> {
        let index = self
            .pose_index(name)
            .ok_or_else(|| invalid(format!("unknown blendshape pose {name}")))?;
        let mut values = self.multipliers.clone();
        values[index] = value;
        self.set_multipliers(values)
    }

    /// Offsets are applied after solving and therefore do not invalidate Prepare.
    pub fn set_offsets(&mut self, values: Vec<f32>) -> Result<()> {
        validate_finite_vector(self.data.pose_count(), &values, "offsets")?;
        self.offsets = values;
        Ok(())
    }

    pub fn set_offset(&mut self, name: &str, value: f32) -> Result<()> {
        let index = self
            .pose_index(name)
            .ok_or_else(|| invalid(format!("unknown blendshape pose {name}")))?;
        let mut values = self.offsets.clone();
        values[index] = value;
        self.set_offsets(values)
    }

    pub fn prepare(&mut self) -> Result<()> {
        self.prepare_impl(true)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn prepare_for_gpu(&mut self) -> Result<()> {
        self.prepare_impl(false)
    }

    fn prepare_impl(&mut self, sdk: bool) -> Result<()> {
        validate_parameters(self.parameters)?;
        if !self.active_poses.iter().any(|value| *value != 0) {
            return Err(invalid("blendshape active pose set is empty"));
        }
        validate_pairs(&self.cancel_poses, "cancel poses")?;
        validate_pairs(&self.symmetry_poses, "symmetry poses")?;

        let coordinate_indices: Vec<usize> = match &self.data.pose_mask {
            Some(mask) => mask
                .iter()
                .flat_map(|vertex| [vertex * 3, vertex * 3 + 1, vertex * 3 + 2])
                .collect(),
            None => (0..self.data.neutral_pose.len()).collect(),
        };
        let active_indices = self
            .active_poses
            .iter()
            .enumerate()
            .filter_map(|(index, active)| (*active != 0).then_some(index))
            .collect::<Vec<_>>();
        let rows = coordinate_indices.len();
        let columns = active_indices.len();
        let source_rows = self.data.neutral_pose.len();
        let active_neutral = coordinate_indices
            .iter()
            .map(|index| self.data.neutral_pose[*index])
            .collect::<Vec<_>>();
        let mut active_deltas = vec![0.0; rows * columns];
        for (active_column, source_column) in active_indices.iter().copied().enumerate() {
            for (row, source_row) in coordinate_indices.iter().copied().enumerate() {
                active_deltas[active_column * rows + row] =
                    self.data.delta_poses[source_column * source_rows + source_row];
            }
        }

        let scale_factor = if sdk {
            let mut range = [0.0_f32; 3];
            for (axis, extent) in range.iter_mut().enumerate() {
                let values = self.data.neutral_pose.iter().skip(axis).step_by(3).copied();
                let low = values.clone().fold(f32::INFINITY, f32::min);
                let high = values.fold(f32::NEG_INFINITY, f32::max);
                *extent = high - low;
            }
            let diagonal =
                (range[0] * range[0] + (range[1] * range[1] + range[2] * range[2])).sqrt();
            f64::from((diagonal / self.parameters.template_bb_size).powi(2))
        } else {
            (bounding_box_diagonal(&self.data.neutral_pose)
                / f64::from(self.parameters.template_bb_size))
            .powi(2)
        };
        let block = if sdk {
            bvls::preparation_block(rows, columns)?.0
        } else {
            rows
        };
        let mut matrix = if sdk {
            Vec::new()
        } else {
            vec![0.0_f64; columns * columns]
        };
        let mut cpu_matrix = if sdk {
            vec![0.0_f32; columns * columns]
        } else {
            Vec::new()
        };
        let cpu_scale = scale_factor as f32;
        let cpu_l1 = self.parameters.l1_regularization
            * self.parameters.l1_regularization
            * (0.25 * cpu_scale);
        let cpu_l2 = self.parameters.l2_regularization * (10.0 * cpu_scale);
        let cpu_temporal = self.parameters.temporal_regularization * (100.0 * cpu_scale);
        let cpu_symmetry = self.parameters.symmetry_regularization * (10.0 * cpu_scale);
        for i in 0..columns {
            for j in 0..columns {
                if sdk {
                    let dot = bvls::gram_entry(
                        &active_deltas[i * rows..(i + 1) * rows],
                        &active_deltas[j * rows..(j + 1) * rows],
                        block,
                        i,
                        j,
                        columns,
                    );
                    let mut value = dot + cpu_l1;
                    if i == j {
                        value += cpu_l2;
                        value += cpu_temporal;
                    }
                    cpu_matrix[i * columns + j] = value;
                    continue;
                }
                let dot = (0..rows)
                    .map(|row| {
                        f64::from(active_deltas[i * rows + row])
                            * f64::from(active_deltas[j * rows + row])
                    })
                    .sum::<f64>();
                matrix[i * columns + j] = dot
                    + f64::from(self.parameters.l1_regularization).powi(2) * 0.25 * scale_factor;
            }
        }
        let diagonal = f64::from(self.parameters.l2_regularization) * 10.0 * scale_factor
            + f64::from(self.parameters.temporal_regularization) * 100.0 * scale_factor;
        if !sdk {
            for i in 0..columns {
                matrix[i * columns + i] += diagonal;
            }
        }
        let symmetry_pairs = active_pairs(&self.symmetry_poses, &active_indices);
        let symmetry = f64::from(self.parameters.symmetry_regularization) * 10.0 * scale_factor;
        for (first, second) in symmetry_pairs {
            if sdk {
                let symmetry = cpu_symmetry;
                for (index, delta) in [
                    (first * columns + first, symmetry),
                    (second * columns + second, symmetry),
                    (first * columns + second, -symmetry),
                    (second * columns + first, -symmetry),
                ] {
                    cpu_matrix[index] += delta;
                }
                continue;
            }
            matrix[first * columns + first] += symmetry;
            matrix[second * columns + second] += symmetry;
            matrix[first * columns + second] -= symmetry;
            matrix[second * columns + first] -= symmetry;
        }
        if (0..columns).any(|index| {
            let diagonal = if sdk {
                f64::from(cpu_matrix[index * columns + index])
            } else {
                matrix[index * columns + index]
            };
            !diagonal.is_finite() || diagonal <= f64::EPSILON
        }) {
            return Err(invalid(
                "blendshape normal matrix is singular; geometry or regularization is insufficient",
            ));
        }

        self.prepared = Some(PreparedBlendshape {
            coordinate_indices,
            active_indices: active_indices.clone(),
            active_deltas,
            active_neutral,
            #[cfg(feature = "cuda")]
            matrix,
            cpu_matrix,
            cancel_pairs: active_pairs(&self.cancel_poses, &active_indices),
            scale_factor,
            previous_weights: vec![0.0; columns],
        });
        Ok(())
    }

    pub fn reset(&mut self) {
        if let Some(prepared) = &mut self.prepared {
            prepared.previous_weights.fill(0.0);
        }
    }

    pub fn evaluate_pose(&self, weights: &[f32]) -> Result<Vec<f32>> {
        self.data.evaluate_pose(weights)
    }

    pub fn solve(&mut self, target_pose: &[f32]) -> Result<Vec<f32>> {
        if target_pose.len() != self.data.neutral_pose.len()
            || target_pose.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "blendshape target pose dimensions/values are invalid",
            ));
        }
        let prepared = self
            .prepared
            .as_ref()
            .ok_or_else(|| invalid("blendshape solver must be prepared before Solve"))?;
        let rows = prepared.coordinate_indices.len();
        let atb = (0..prepared.active_indices.len())
            .map(|column| {
                prepared
                    .coordinate_indices
                    .iter()
                    .enumerate()
                    .map(|(row, source)| {
                        prepared.active_deltas[column * rows + row]
                            * (target_pose[*source] - prepared.active_neutral[row])
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        self.solve_from_atb(&atb)
    }

    pub(crate) fn solve_from_atb(&mut self, atb: &[f32]) -> Result<Vec<f32>> {
        let prepared = self
            .prepared
            .as_mut()
            .ok_or_else(|| invalid("blendshape solver must be prepared before Solve"))?;
        let columns = prepared.active_indices.len();
        if atb.len() != columns || atb.iter().any(|value| !value.is_finite()) {
            return Err(invalid("blendshape RHS dimensions/values are invalid"));
        }
        let temporal = self.parameters.temporal_regularization * prepared.scale_factor as f32;
        let rhs = atb
            .iter()
            .zip(&prepared.previous_weights)
            .map(|(value, previous)| *value + temporal * previous)
            .collect::<Vec<_>>();
        let mut upper = vec![1.0_f32; columns];
        let mut weights = bvls::solve_production(
            &prepared.cpu_matrix,
            &rhs,
            &upper,
            self.parameters.tolerance,
        )?;
        for (first, second) in &prepared.cancel_pairs {
            if weights[*first] >= weights[*second] {
                upper[*second] = 1.0e-10;
            } else {
                upper[*first] = 1.0e-10;
            }
        }
        if !prepared.cancel_pairs.is_empty() {
            weights = bvls::solve_production(
                &prepared.cpu_matrix,
                &rhs,
                &upper,
                self.parameters.tolerance,
            )?;
        }
        prepared.previous_weights.clone_from(&weights);

        let mut output = vec![0.0_f32; self.data.pose_count()];
        for (active, source) in prepared.active_indices.iter().copied().enumerate() {
            output[source] = weights[active];
        }
        for ((weight, multiplier), offset) in
            output.iter_mut().zip(&self.multipliers).zip(&self.offsets)
        {
            *weight = *weight * *multiplier + *offset;
        }
        Ok(output)
    }
}

fn bounding_box_diagonal(pose: &[f32]) -> f64 {
    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for vertex in pose.chunks_exact(3) {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex[axis]);
            maximum[axis] = maximum[axis].max(vertex[axis]);
        }
    }
    (0..3)
        .map(|axis| f64::from(maximum[axis] - minimum[axis]).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn active_pairs(values: &[i32], active_indices: &[usize]) -> Vec<(usize, usize)> {
    let mut groups: HashMap<i32, Vec<usize>> = HashMap::new();
    for (active, source) in active_indices.iter().copied().enumerate() {
        let pair = values[source];
        if pair >= 0 {
            groups.entry(pair).or_default().push(active);
        }
    }
    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups.sort_by_key(|(id, _)| *id);
    groups
        .into_iter()
        .filter_map(|(_, indices)| (indices.len() == 2).then_some((indices[0], indices[1])))
        .collect()
}

fn validate_parameters(parameters: BlendshapeSolverParameters) -> Result<()> {
    let strengths = [
        parameters.l1_regularization,
        parameters.l2_regularization,
        parameters.symmetry_regularization,
        parameters.temporal_regularization,
    ];
    if strengths
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || !parameters.template_bb_size.is_finite()
        || parameters.template_bb_size <= 0.0
        || !parameters.tolerance.is_finite()
        || parameters.tolerance <= 0.0
    {
        return Err(invalid("invalid blendshape solver parameters"));
    }
    Ok(())
}

fn validate_vector_len(expected: usize, values: &[i32], name: &str) -> Result<()> {
    if values.len() != expected {
        Err(invalid(format!("{name} length does not match pose count")))
    } else {
        Ok(())
    }
}

fn validate_finite_vector(expected: usize, values: &[f32], name: &str) -> Result<()> {
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        Err(invalid(format!("{name} dimensions/values are invalid")))
    } else {
        Ok(())
    }
}

fn validate_pairs(values: &[i32], name: &str) -> Result<()> {
    let mut counts = HashMap::new();
    for value in values.iter().copied().filter(|value| *value >= 0) {
        *counts.entry(value).or_insert(0_usize) += 1;
    }
    if let Some((pair, count)) = counts.into_iter().find(|(_, count)| *count != 2) {
        Err(invalid(format!(
            "{name} pair id {pair} occurs {count} times instead of twice"
        )))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> BlendshapeData {
        // Two vertices, and three orthogonal-enough shape columns.
        BlendshapeData {
            neutral_pose: vec![0.0, 0.0, 0.0, 2.0, 3.0, 4.0],
            delta_poses: vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x
                0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // y
                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // z
            ],
            pose_names: vec!["x".into(), "y".into(), "z".into()],
            pose_mask: None,
        }
    }

    fn unregularized_solver() -> CpuBlendshapeSolver {
        let mut solver = CpuBlendshapeSolver::new(data()).unwrap();
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
    fn evaluate_and_solve_reconstruct_pose_with_bounded_weights() {
        let mut solver = unregularized_solver();
        let target = solver.evaluate_pose(&[0.25, 0.5, 1.0]).unwrap();
        let weights = solver.solve(&target).unwrap();
        assert!((weights[0] - 0.25).abs() < 1.0e-6);
        assert!((weights[1] - 0.5).abs() < 1.0e-6);
        assert!((weights[2] - 1.0).abs() < 1.0e-6);
        assert!(weights.iter().all(|value| (0.0..=1.0).contains(value)));
        assert_eq!(solver.evaluate_pose(&weights).unwrap(), target);
    }

    #[test]
    fn prepare_state_and_post_solve_parameters_match_sdk_contract() {
        let mut solver = unregularized_solver();
        solver.set_multipliers(vec![2.0, 1.0, 1.0]).unwrap();
        solver.set_offsets(vec![0.1, 0.0, -0.2]).unwrap();
        assert!(solver.is_prepared());
        let target = solver.data().evaluate_pose(&[0.2, 0.0, 0.0]).unwrap();
        let weights = solver.solve(&target).unwrap();
        assert!((weights[0] - 0.5).abs() < 1.0e-6);
        assert_eq!(weights[2], -0.2);
        solver.set_active_poses(vec![1, 0, 1]).unwrap();
        assert!(!solver.is_prepared());
        assert!(solver.solve(&target).is_err());
    }

    #[test]
    fn cancel_pair_suppresses_the_weaker_pose() {
        let mut solver = unregularized_solver();
        solver.set_cancel_poses(vec![4, 4, -1]).unwrap();
        solver.prepare().unwrap();
        let target = solver.data().evaluate_pose(&[0.8, 0.3, 0.0]).unwrap();
        let weights = solver.solve(&target).unwrap();
        assert!(weights[0] > 0.79);
        assert!(weights[1] <= 1.1e-10);
    }

    #[test]
    fn bounded_solve_minimizes_sdk_matrix_residual() {
        // With x[1] pinned to zero, ||A x - b||^2 has its minimum
        // at x[0] = 2/5. The geometry quadratic instead gives 1/2.
        let weights =
            bvls::solve_production(&[2.0, 1.0, 1.0, 2.0], &[1.0, 0.0], &[1.0, 1.0], 1.0e-10)
                .unwrap();
        assert!((weights[0] - 0.4).abs() < 1.0e-6);
        assert_eq!(weights[1], 0.0);
    }

    #[test]
    fn bounded_solve_converges_for_nearly_dependent_columns() {
        let weights = bvls::solve_production(
            &[1.0, 1.0, 1.0, 1.0001],
            &[0.8, 0.80006],
            &[1.0, 1.0],
            1.0e-10,
        )
        .unwrap();
        // Verify the residual in FP32: nearly dependent columns do not support
        // the old FP64 weight precision, even in the original SDK solver.
        assert!((weights[0] + weights[1] - 0.8).abs() < 1.0e-6);
        assert!((weights[0] + 1.0001 * weights[1] - 0.80006).abs() < 1.0e-6);
    }

    #[test]
    fn symmetry_and_temporal_regularization_affect_solution_and_reset() {
        let mut solver = CpuBlendshapeSolver::new(data()).unwrap();
        solver.set_symmetry_poses(vec![0, 0, -1]).unwrap();
        solver
            .set_parameters(BlendshapeSolverParameters {
                l1_regularization: 0.0,
                l2_regularization: 0.1,
                symmetry_regularization: 100.0,
                temporal_regularization: 10.0,
                ..BlendshapeSolverParameters::default()
            })
            .unwrap();
        solver.prepare().unwrap();
        let target = solver.data().evaluate_pose(&[1.0, 0.0, 0.0]).unwrap();
        let first = solver.solve(&target).unwrap();
        let second = solver.solve(&target).unwrap();
        assert!((first[0] - first[1]).abs() < 0.06);
        assert!(second[0] >= first[0]);
        solver.reset();
        let replay = solver.solve(&target).unwrap();
        assert!((replay[0] - first[0]).abs() < 1.0e-6);
    }

    #[test]
    fn validates_mask_empty_active_pairs_and_singular_geometry() {
        let mut invalid_mask = data();
        invalid_mask.pose_mask = Some(vec![0, 0]);
        assert!(CpuBlendshapeSolver::new(invalid_mask).is_err());
        let mut solver = CpuBlendshapeSolver::new(data()).unwrap();
        assert!(solver.set_active_poses(vec![0, 0, 0]).is_err());
        assert!(solver.set_cancel_poses(vec![1, -1, -1]).is_err());
        solver
            .set_parameters(BlendshapeSolverParameters {
                l1_regularization: 0.0,
                l2_regularization: 0.0,
                symmetry_regularization: 0.0,
                temporal_regularization: 0.0,
                ..BlendshapeSolverParameters::default()
            })
            .unwrap();
        solver.data.delta_poses.fill(0.0);
        assert!(solver.prepare().is_err());
    }

    #[test]
    fn regularization_stabilizes_ill_conditioned_shapes() {
        let mut ill_conditioned = data();
        ill_conditioned.delta_poses[6..12].copy_from_slice(&[1.0, 1.0e-7, 0.0, 0.0, 0.0, 0.0]);
        let mut solver = CpuBlendshapeSolver::new(ill_conditioned).unwrap();
        solver
            .set_parameters(BlendshapeSolverParameters {
                l1_regularization: 0.0,
                l2_regularization: 1.0,
                symmetry_regularization: 0.0,
                temporal_regularization: 0.0,
                ..BlendshapeSolverParameters::default()
            })
            .unwrap();
        solver.prepare().unwrap();
        let target = solver.data().evaluate_pose(&[0.5, 0.5, 0.0]).unwrap();
        let weights = solver.solve(&target).unwrap();
        assert!(weights.iter().all(|value| value.is_finite()));
        assert!(weights.iter().all(|value| (0.0..=1.0).contains(value)));
    }

    #[test]
    fn pose_name_getters_and_setters_validate_names() {
        let mut solver = CpuBlendshapeSolver::new(data()).unwrap();
        assert_eq!(solver.pose_name(1), Some("y"));
        assert_eq!(solver.pose_name(3), None);
        solver.set_multiplier("y", 2.0).unwrap();
        solver.set_offset("z", 0.25).unwrap();
        assert_eq!(solver.multipliers()[1], 2.0);
        assert_eq!(solver.offsets()[2], 0.25);
        assert!(solver.set_active_pose("missing", 1).is_err());
        assert!(solver.set_cancel_pose("x", 9).is_err());
    }

    #[test]
    fn gpu_flag_selects_only_the_two_original_factory_paths() {
        assert_eq!(
            BlendshapeSolverKind::from_gpu_flag(false),
            BlendshapeSolverKind::Cpu
        );
        assert_eq!(
            BlendshapeSolverKind::from_gpu_flag(true),
            BlendshapeSolverKind::Gpu
        );
    }
}
