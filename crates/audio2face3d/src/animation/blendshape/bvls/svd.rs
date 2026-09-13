// Rust adaptation of Eigen's real JacobiSVD, ColPivHouseholderQR and reductions.
// Copyright (C) 2006-2010 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2008-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2013 Gauthier Brun <brun.gauthier@gmail.com>
// Copyright (C) 2013 Nicolas Carre <nicolas.carre@ensimag.fr>
// Copyright (C) 2013 Jean Ceccato <jean.ceccato@ensimag.fr>
// Copyright (C) 2013 Pierre Zoppitelli <pierre.zoppitelli@ensimag.fr>
// SPDX-License-Identifier: MPL-2.0
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. A copy is in LICENSE-MPL-2.0 at the repository root,
// and in the crate's LICENSE for standalone source packages.

use crate::common::{Error, Result};

/// Single-threaded Eigen SSE FP32 K blocking. Other packet widths need their
/// own product implementation; do not silently claim compatibility with them.
pub(in crate::animation::blendshape) fn preparation_block(
    rows: usize,
    columns: usize,
) -> Result<(usize, usize)> {
    // Eigen retains its cache-size query for the process. Re-querying on every
    // preparation changes the reduction order after migration between hybrid
    // CPU cores with different L1 sizes, even for identical inputs.
    // Eigen uses 32 KiB on x86 when the cache query is unavailable.
    static L1_BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let l1 = *L1_BYTES.get_or_init(|| l1_data_cache_bytes().unwrap_or(32 * 1024));
    Ok((sse_k_block(rows, columns, l1), l1))
}

fn sse_k_block(rows: usize, columns: usize, l1: usize) -> usize {
    if rows.max(columns) < 48 {
        return rows.max(1);
    }
    // Eigen gebp_traits<float,float>: mr=8, nr=4, sizeof(float)=4.
    let maximum = ((l1.saturating_sub(128) / 48) & !7).max(1);
    if rows <= maximum {
        rows.max(1)
    } else if rows.is_multiple_of(maximum) {
        maximum
    } else {
        maximum - 8 * ((maximum - 1 - rows % maximum) / (8 * (rows / maximum + 1)))
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unused_unsafe)] // CPUID intrinsic safety differs across supported Rust versions.
fn l1_data_cache_bytes() -> Option<usize> {
    use std::arch::x86_64::__cpuid_count;
    // SAFETY: CPUID is available on x86-64. Query maximum supported leaves
    // before reading deterministic cache information or AMD extended leaves.
    unsafe {
        let identity = __cpuid_count(0, 0);
        let vendor = [identity.ebx, identity.edx, identity.ecx];
        if vendor == [0x6874_7541, 0x6974_6e65, 0x444d_4163]
            || vendor == [0x6944_4d41, 0x7465_6273, 0x2172_6574]
        {
            if __cpuid_count(0x8000_0000, 0).eax < 0x8000_0006 {
                return None;
            }
            let bytes = (__cpuid_count(0x8000_0005, 0).ecx >> 24) as usize * 1024;
            return (bytes > 0).then_some(bytes);
        }
        if identity.eax < 4 {
            return None;
        }
        let mut l1 = None;
        for index in 0..16 {
            let cache = __cpuid_count(4, index);
            let kind = cache.eax & 0xf;
            if kind == 0 {
                break;
            }
            if matches!(kind, 1 | 3) && (cache.eax >> 5) & 7 == 1 {
                let ways = ((cache.ebx >> 22) + 1) as usize;
                let partitions = (((cache.ebx >> 12) & 0x3ff) + 1) as usize;
                let line = ((cache.ebx & 0xfff) + 1) as usize;
                l1 = ways
                    .checked_mul(partitions)?
                    .checked_mul(line)?
                    .checked_mul(cache.ecx as usize + 1);
            }
        }
        l1
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn l1_data_cache_bytes() -> Option<usize> {
    None
}

#[test]
fn preparation_blocks_preserve_the_number_of_product_sweeps() {
    assert_eq!(sse_k_block(12_786, 39, 49_152), 984);
    for l1 in [16_384_usize, 32_768, 49_152, 65_536] {
        let maximum = ((l1 - 128) / 48) & !7;
        for rows in 48..20_000 {
            let block = sse_k_block(rows, 39, l1);
            assert!(block > 0 && block <= rows && block <= maximum);
            if rows > maximum {
                assert_eq!(block % 8, 0);
                assert_eq!(rows.div_ceil(block), rows.div_ceil(maximum));
            }
        }
    }
    assert_eq!(sse_k_block(12, 4, 32_768), 12);
}

/// FP32 least-squares solve preserving the SDK's SSE reduction order.
/// Matrices are column-major. Reductions follow the SSE four-lane order;
/// this is not a guarantee of bitwise equivalence for every Eigen build.
pub(super) fn least_squares(a: &[f32], rows: usize, cols: usize, b: &[f32]) -> Result<Vec<f32>> {
    let mut scale = a.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    if scale == 0.0 {
        scale = 1.0;
    }
    let mut work = vec![0.0; cols * cols];
    let mut u = vec![0.0; rows * cols];
    let mut v = vec![0.0; cols * cols];
    for i in 0..cols {
        u[i * rows + i] = 1.0;
        v[i * cols + i] = 1.0;
    }
    if rows == cols {
        for (dst, src) in work.iter_mut().zip(a) {
            *dst = src / scale;
        }
    } else {
        let mut qr = a.iter().map(|value| value / scale).collect::<Vec<_>>();
        let mut norms = (0..cols)
            .map(|j| norm(&qr[j * rows..(j + 1) * rows]))
            .collect::<Vec<_>>();
        let mut direct = norms.clone();
        let mut tau = vec![0.0; cols];
        for k in 0..cols {
            let mut pivot = k;
            for j in k + 1..cols {
                if norms[j] > norms[pivot] {
                    pivot = j;
                }
            }
            if pivot != k {
                for i in 0..rows {
                    qr.swap(k * rows + i, pivot * rows + i);
                }
                for i in 0..cols {
                    v.swap(k * cols + i, pivot * cols + i);
                }
                norms.swap(k, pivot);
                direct.swap(k, pivot);
            }
            let tail = &mut qr[k * rows + k..(k + 1) * rows];
            let tail_sq = squared_norm(&tail[1..]);
            let first = tail[0];
            if tail_sq <= f32::MIN_POSITIVE {
                tail[1..].fill(0.0);
            } else {
                let mut beta = (first * first + tail_sq).sqrt();
                if first >= 0.0 {
                    beta = -beta;
                }
                for value in &mut tail[1..] {
                    *value /= first - beta;
                }
                tau[k] = (beta - first) / beta;
                tail[0] = beta;
            }
            for j in k + 1..cols {
                let tmp = packet_dot(
                    &qr[k * rows + k + 1..(k + 1) * rows],
                    &qr[j * rows + k + 1..(j + 1) * rows],
                ) + qr[j * rows + k];
                qr[j * rows + k] -= tau[k] * tmp;
                for i in k + 1..rows {
                    qr[j * rows + i] -= (tau[k] * qr[k * rows + i]) * tmp;
                }
                if norms[j] != 0.0 {
                    let ratio = qr[j * rows + k].abs() / norms[j];
                    let temp = ((1.0 + ratio) * (1.0 - ratio)).max(0.0);
                    let accuracy = temp * (norms[j] / direct[j]).powi(2);
                    if accuracy <= f32::EPSILON.sqrt() {
                        direct[j] = norm(&qr[j * rows + k + 1..(j + 1) * rows]);
                        norms[j] = direct[j];
                    } else {
                        norms[j] *= temp.sqrt();
                    }
                }
            }
        }
        for j in 0..cols {
            for i in 0..=j {
                work[j * cols + i] = qr[j * rows + i];
            }
        }
        for k in (0..cols).rev() {
            for j in 0..cols {
                let tmp = packet_dot(
                    &qr[k * rows + k + 1..(k + 1) * rows],
                    &u[j * rows + k + 1..(j + 1) * rows],
                ) + u[j * rows + k];
                u[j * rows + k] -= tau[k] * tmp;
                for i in k + 1..rows {
                    u[j * rows + i] -= (tau[k] * qr[k * rows + i]) * tmp;
                }
            }
        }
    }
    let mut max_diag = (0..cols)
        .map(|i| work[i * cols + i].abs())
        .fold(0.0_f32, f32::max);
    let mut finished = false;
    for _ in 0..100 {
        finished = true;
        for p in 1..cols {
            for q in 0..p {
                let threshold = (2.0 * f32::EPSILON * max_diag).max(f32::MIN_POSITIVE);
                if work[q * cols + p].abs() <= threshold && work[p * cols + q].abs() <= threshold {
                    continue;
                }
                finished = false;
                let a = work[p * cols + p];
                let b = work[q * cols + p];
                let c = work[p * cols + q];
                let d = work[q * cols + q];
                let delta = c - b;
                let (c1, s1) = if delta.abs() < f32::MIN_POSITIVE {
                    (1.0, 0.0)
                } else {
                    let ratio = (a + d) / delta;
                    let tmp = (1.0 + ratio * ratio).sqrt();
                    (ratio / tmp, 1.0 / tmp)
                };
                let aa = c1 * a + s1 * c;
                let bb = c1 * b + s1 * d;
                let dd = c1 * d - s1 * b;
                let (cr, sr) = if 2.0 * bb.abs() < f32::MIN_POSITIVE {
                    (1.0, 0.0)
                } else {
                    let tau = (aa - dd) / (2.0 * bb.abs());
                    let w = (tau * tau + 1.0).sqrt();
                    let t = 1.0 / if tau > 0.0 { tau + w } else { tau - w };
                    let sign = if t > 0.0 { 1.0 } else { -1.0 };
                    let cosine = 1.0 / (t * t + 1.0).sqrt();
                    (cosine, -sign * (bb / bb.abs()) * t.abs() * cosine)
                };
                let cl = c1 * cr + s1 * sr;
                let sl = -c1 * sr + s1 * cr;
                for j in 0..cols {
                    let x = work[j * cols + p];
                    let y = work[j * cols + q];
                    work[j * cols + p] = cl * x + sl * y;
                    work[j * cols + q] = cl * y - sl * x;
                }
                rotate_columns(&mut u, rows, p, q, cl, sl);
                rotate_columns(&mut work, cols, p, q, cr, -sr);
                rotate_columns(&mut v, cols, p, q, cr, -sr);
                max_diag = max_diag
                    .max(work[p * cols + p].abs())
                    .max(work[q * cols + q].abs());
            }
        }
        if finished {
            break;
        }
    }
    if !finished {
        return Err(Error::InvalidSchema(
            "CPU Jacobi SVD did not converge".into(),
        ));
    }
    let mut singular = (0..cols)
        .map(|i| work[i * cols + i].abs() * scale)
        .collect::<Vec<_>>();
    for i in 0..cols {
        if work[i * cols + i] < 0.0 {
            for value in &mut u[i * rows..(i + 1) * rows] {
                *value = -*value;
            }
        }
    }
    for i in 0..cols {
        let mut pivot = i;
        for j in i + 1..cols {
            if singular[j] > singular[pivot] {
                pivot = j;
            }
        }
        singular.swap(i, pivot);
        for row in 0..rows {
            u.swap(i * rows + row, pivot * rows + row);
        }
        for row in 0..cols {
            v.swap(i * cols + row, pivot * cols + row);
        }
    }
    let threshold = (singular[0] * cols as f32 * f32::EPSILON).max(f32::MIN_POSITIVE);
    let mut result = vec![0.0; cols];
    for j in 0..cols {
        if singular[j] < threshold {
            break;
        }
        let tmp = (1.0 / singular[j]) * packet_dot(&u[j * rows..(j + 1) * rows], b);
        for i in 0..cols {
            result[i] += v[j * cols + i] * tmp;
        }
    }
    Ok(result)
}

fn norm(values: &[f32]) -> f32 {
    squared_norm(values).sqrt()
}

// Eigen GeneralBlockPanelKernel SSE tail-row accumulation. The block size is
// supplied by the diagnostic, not hard-coded from one machine into production.
pub(in crate::animation::blendshape) fn gram_entry(
    a: &[f32],
    b: &[f32],
    block: usize,
    row: usize,
    col: usize,
    width: usize,
) -> f32 {
    let mut total = 0.0;
    for start in (0..a.len()).step_by(block) {
        let end = (start + block).min(a.len());
        let lanes = if col >= width / 4 * 4 || row < width / 8 * 8 {
            1
        } else if row < width / 4 * 4 {
            2
        } else {
            4
        };
        let peel = if lanes == 2 { 8 } else { lanes };
        let peeled_end = start + (end - start) / peel * peel;
        let mut sums = [0.0; 4];
        for k in start..peeled_end {
            sums[(k - start) % lanes] += a[k] * b[k];
        }
        let mut partial = if lanes == 4 {
            (sums[0] + sums[1]) + (sums[2] + sums[3])
        } else {
            sums[0] + sums[1]
        };
        for k in peeled_end..end {
            partial += a[k] * b[k];
        }
        total += partial;
    }
    total
}

// Match Eigen's SSE packet reduction order without adding arithmetic passes.
// The lane arrays also permit LLVM to vectorize the independent accumulations.
pub(super) fn packet_dot(a: &[f32], b: &[f32]) -> f32 {
    packet_dot_strided(a, 1, b)
}

pub(super) fn packet_dot_strided(a: &[f32], stride: usize, b: &[f32]) -> f32 {
    let mut lanes = [0.0; 4];
    let end = b.len() / 4 * 4;
    for i in (0..end).step_by(4) {
        for lane in 0..4 {
            lanes[lane] += a[(i + lane) * stride] * b[i + lane];
        }
    }
    let mut sum = (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
    for i in end..b.len() {
        sum += a[i * stride] * b[i];
    }
    sum
}

pub(super) fn squared_norm(a: &[f32]) -> f32 {
    if a.len() < 4 {
        return a.iter().map(|x| x * x).sum();
    }
    let mut first = [0.0; 4];
    let mut second = [0.0; 4];
    let end = a.len() / 8 * 8;
    for i in (0..end).step_by(8) {
        for lane in 0..4 {
            first[lane] += a[i + lane] * a[i + lane];
            second[lane] += a[i + lane + 4] * a[i + lane + 4];
        }
    }
    for lane in 0..4 {
        first[lane] += second[lane];
    }
    let tail = a.len() / 4 * 4;
    if tail > end {
        for lane in 0..4 {
            first[lane] += a[end + lane] * a[end + lane];
        }
    }
    let mut sum = (first[0] + first[2]) + (first[1] + first[3]);
    for x in &a[tail..] {
        sum += x * x;
    }
    sum
}

fn rotate_columns(a: &mut [f32], rows: usize, p: usize, q: usize, c: f32, s: f32) {
    for i in 0..rows {
        let x = a[p * rows + i];
        let y = a[q * rows + i];
        a[p * rows + i] = c * x + s * y;
        a[q * rows + i] = c * y - s * x;
    }
}

#[test]
fn square_and_rectangular_least_squares() {
    let square = least_squares(&[3.0, 1.0, 1.0, 2.0], 2, 2, &[4.0, 3.0]).unwrap();
    assert!(square.iter().all(|v| (*v - 1.0).abs() < 1e-5), "{square:?}");
    let rectangular =
        least_squares(&[1.0, 0.0, 1.0, 0.0, 1.0, 1.0], 3, 2, &[1.0, 2.0, 4.0]).unwrap();
    assert!((rectangular[0] - 4.0 / 3.0).abs() < 1e-5, "{rectangular:?}");
    assert!((rectangular[1] - 7.0 / 3.0).abs() < 1e-5, "{rectangular:?}");
}

#[test]
fn rank_deficient_system_returns_minimum_norm_solution() {
    let x = least_squares(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 3, 2, &[1.0, 2.0, 3.0]).unwrap();
    assert!((x[0] - 0.2).abs() < 1e-5, "{x:?}");
    assert!((x[1] - 0.4).abs() < 1e-5, "{x:?}");
}
