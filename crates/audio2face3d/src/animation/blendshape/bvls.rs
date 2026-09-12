//! Bounded least squares following the SDK's active-set and stopping rules.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
// Adapted from audio2face-core/bvls.cpp; see ../../../LICENSE.

use crate::common::{Error, Result};
mod svd;
pub(super) use svd::{gram_entry, preparation_block};

#[derive(Default)]
#[cfg_attr(test, derive(serde::Serialize))]
pub(super) struct Stats {
    pub subproblems: usize,
    pub svd_calls: usize,
    pub iterations: usize,
    pub feasible_steps: usize,
    pub stop: &'static str,
}

#[cfg(test)]
pub(super) fn solve(
    matrix: &[f64],
    rhs: &[f64],
    upper: &[f64],
    tolerance: f32,
) -> Result<(Vec<f64>, Stats)> {
    let a = matrix.iter().map(|x| *x as f32).collect::<Vec<_>>();
    let b = rhs.iter().map(|x| *x as f32).collect::<Vec<_>>();
    let u = upper.iter().map(|x| *x as f32).collect::<Vec<_>>();
    let (weights, stats) = solve_impl::<true>(&a, &b, &u, tolerance)?;
    Ok((weights.into_iter().map(f64::from).collect(), stats))
}

pub(super) fn solve_production(
    a: &[f32],
    b: &[f32],
    u: &[f32],
    tolerance: f32,
) -> Result<Vec<f32>> {
    Ok(solve_impl::<false>(a, b, u, tolerance)?.0)
}

fn solve_impl<const TRACE: bool>(
    a: &[f32],
    b: &[f32],
    u: &[f32],
    tolerance: f32,
) -> Result<(Vec<f32>, Stats)> {
    let mut stats = Stats {
        stop: "iteration_limit",
        ..Stats::default()
    };
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut bounds = vec![0_i8; n];
    for _ in 0..=n {
        if TRACE {
            stats.subproblems += 1;
            stats.svd_calls += usize::from(bounds.contains(&0));
        }
        constrained_minimum(&mut x, a, b, &bounds)?;
        let mut found = false;
        for i in 0..n {
            if bounds[i] != 0 {
                continue;
            }
            if x[i] < 0.0 {
                x[i] = 0.0;
                bounds[i] = -1;
                found = true;
            } else if x[i] > u[i] {
                x[i] = u[i];
                bounds[i] = 1;
                found = true;
            }
        }
        if !found {
            break;
        }
    }
    update_bounds(&mut bounds, &x, u);
    let (mut cost, mut gradient) = cost_gradient(a, b, &x);
    let mut optimality = kkt(&gradient, &bounds);
    let mut previous = optimality;
    // These limits and early returns intentionally follow SDK BVLS, including
    // returning the current iterate when single-precision cost increases.
    for _ in 0..n {
        if optimality < tolerance {
            stats.stop = "kkt_tolerance";
            break;
        }
        let mut selected = None;
        let mut largest = 0.0_f32;
        for i in 0..n {
            let violation = if x[i] == 0.0 {
                -gradient[i]
            } else if x[i] == u[i] {
                gradient[i]
            } else {
                0.0
            };
            if violation > largest {
                largest = violation;
                selected = Some(i);
            }
        }
        let Some(selected) = selected else {
            stats.stop = "no_bound_to_release";
            break;
        };
        bounds[selected] = 0;
        stats.iterations += 1;
        let mut reached = false;
        for _ in 0..=n {
            if TRACE {
                stats.subproblems += 1;
                stats.svd_calls += usize::from(bounds.contains(&0));
                stats.feasible_steps += 1;
            }
            let mut candidate = x.clone();
            constrained_minimum(&mut candidate, a, b, &bounds)?;
            let mut alpha = 1.0_f32;
            let mut hit = None;
            for i in 0..n {
                if x[i] == candidate[i] {
                    continue;
                }
                let (fraction, bound) = if candidate[i] < 0.0 {
                    (x[i] / (x[i] - candidate[i]), 0.0)
                } else if candidate[i] > u[i] {
                    ((u[i] - x[i]) / (candidate[i] - x[i]), u[i])
                } else {
                    continue;
                };
                if fraction < alpha {
                    alpha = fraction;
                    hit = Some((i, bound));
                }
            }
            for i in 0..n {
                x[i] += alpha * (candidate[i] - x[i]);
            }
            if let Some((i, bound)) = hit {
                x[i] = bound;
            }
            if alpha == 1.0 {
                reached = true;
                break;
            }
            update_bounds(&mut bounds, &x, u);
        }
        if !reached {
            return Err(Error::InvalidSchema(
                "CPU BVLS feasible step did not terminate".into(),
            ));
        }
        let (next_cost, next_gradient) = cost_gradient(a, b, &x);
        if next_cost > cost {
            stats.stop = "cost_increase";
            break;
        }
        cost = next_cost;
        gradient = next_gradient;
        optimality = kkt(&gradient, &bounds);
        if (previous - optimality).abs() < 1.0e-6 {
            stats.stop = "optimality_plateau";
            break;
        }
        previous = optimality;
    }
    Ok((x, stats))
}

fn update_bounds(bounds: &mut [i8], x: &[f32], upper: &[f32]) {
    for ((bound, value), upper) in bounds.iter_mut().zip(x).zip(upper) {
        *bound = if *value == 0.0 {
            -1
        } else if value == upper {
            1
        } else {
            0
        };
    }
}

fn cost_gradient(a: &[f32], b: &[f32], x: &[f32]) -> (f32, Vec<f32>) {
    let n = b.len();
    let residual = (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j] * x[j]).sum::<f32>() - b[i])
        .collect::<Vec<_>>();
    let gradient = (0..n)
        .map(|j| svd::packet_dot_strided(&a[j..], n, &residual))
        .collect();
    (0.5 * svd::squared_norm(&residual), gradient)
}

fn kkt(gradient: &[f32], bounds: &[i8]) -> f32 {
    gradient
        .iter()
        .zip(bounds)
        .map(|(g, bound)| {
            if *bound == 0 {
                g.abs()
            } else {
                g * f32::from(*bound)
            }
        })
        .fold(f32::NEG_INFINITY, f32::max)
}

fn constrained_minimum(x: &mut [f32], a: &[f32], b: &[f32], bounds: &[i8]) -> Result<()> {
    let n = b.len();
    let free = (0..n).filter(|i| bounds[*i] == 0).collect::<Vec<_>>();
    let width = free.len();
    if width == 0 {
        return Ok(());
    }
    let residual = (0..n)
        .map(|row| {
            b[row]
                - (0..n)
                    .filter(|i| bounds[*i] != 0)
                    .map(|i| a[row * n + i] * x[i])
                    .sum::<f32>()
        })
        .collect::<Vec<_>>();
    let mut reduced = vec![0.0; n * width];
    for (column, source) in free.iter().copied().enumerate() {
        for row in 0..n {
            reduced[column * n + row] = a[row * n + source];
        }
    }
    let solution = svd::least_squares(&reduced, n, width, &residual)?;
    for (column, source) in free.iter().copied().enumerate() {
        x[source] = solution[column];
    }
    Ok(())
}

#[test]
fn diagonal_bounds_need_no_extra_decomposition_after_clamping() {
    let (x, stats) = solve(&[1.0, 0.0, 0.0, 1.0], &[-1.0, 2.0], &[1.0, 1.0], 1e-6).unwrap();
    assert_eq!(x, [0.0, 1.0]);
    assert_eq!(stats.svd_calls, 1);
    assert_eq!(stats.iterations, 0);
    assert_eq!(stats.stop, "kkt_tolerance");
}

#[test]
fn coupled_bound_solves_the_residual_objective() {
    let (x, _) = solve(&[1.0, 2.0, 2.0, 5.0], &[-1.0, 1.0], &[1.0, 1.0], 1e-6).unwrap();
    assert_eq!(x[0], 0.0);
    assert!((x[1] - 3.0 / 29.0).abs() < 1e-6, "{x:?}");
}
