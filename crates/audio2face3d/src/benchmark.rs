use crate::common::Result;
use serde_json::{Value, json};
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Percentiles {
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkPhase {
    pub name: String,
    pub iterations: usize,
    pub percentiles: Percentiles,
    /// Completed iterations per second across the complete measured phase.
    pub throughput_per_second: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkReport {
    pub warmup_iterations: usize,
    pub phases: Vec<BenchmarkPhase>,
    pub peak_memory_mib: Option<u64>,
}

impl BenchmarkReport {
    pub fn to_json(&self) -> Value {
        json!({
            "warmup_iterations": self.warmup_iterations,
            "peak_memory_mib": self.peak_memory_mib,
            "phases": self.phases.iter().map(|phase| json!({
                "name": phase.name,
                "iterations": phase.iterations,
                "p50_ns": phase.percentiles.p50_ns,
                "p95_ns": phase.percentiles.p95_ns,
                "p99_ns": phase.percentiles.p99_ns,
                "throughput_per_second": phase.throughput_per_second,
            })).collect::<Vec<_>>(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkRunner {
    pub warmup_iterations: usize,
    pub measured_iterations: usize,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self {
            warmup_iterations: 10,
            measured_iterations: 100,
        }
    }
}

impl BenchmarkRunner {
    pub fn run<Build, Cache, Steady, Post, EndToEnd, Memory>(
        &self,
        mut build: Build,
        mut cache: Cache,
        mut steady: Steady,
        mut post_process: Post,
        mut end_to_end: EndToEnd,
        mut memory: Memory,
    ) -> Result<BenchmarkReport>
    where
        Build: FnMut() -> Result<()>,
        Cache: FnMut() -> Result<()>,
        Steady: FnMut() -> Result<()>,
        Post: FnMut() -> Result<()>,
        EndToEnd: FnMut() -> Result<()>,
        Memory: FnMut() -> Option<u64>,
    {
        if self.warmup_iterations == 0 || self.measured_iterations == 0 {
            return Err(crate::common::Error::InvalidSchema(
                "benchmark warm-up and measured iterations must be non-zero".into(),
            ));
        }
        let mut peak = memory();
        let build = measure("build", 1, &mut build, &mut memory, &mut peak)?;
        let cache = measure("cache", 1, &mut cache, &mut memory, &mut peak)?;
        let warmup = measure(
            "warmup",
            self.warmup_iterations,
            &mut steady,
            &mut memory,
            &mut peak,
        )?;
        let steady = measure(
            "steady-state",
            self.measured_iterations,
            &mut steady,
            &mut memory,
            &mut peak,
        )?;
        let post = measure(
            "post-process",
            self.measured_iterations,
            &mut post_process,
            &mut memory,
            &mut peak,
        )?;
        let e2e = measure(
            "end-to-end",
            self.measured_iterations,
            &mut end_to_end,
            &mut memory,
            &mut peak,
        )?;
        Ok(BenchmarkReport {
            warmup_iterations: self.warmup_iterations,
            phases: vec![build, cache, warmup, steady, post, e2e],
            peak_memory_mib: peak,
        })
    }
}

fn measure<F, M>(
    name: &str,
    iterations: usize,
    function: &mut F,
    memory: &mut M,
    peak: &mut Option<u64>,
) -> Result<BenchmarkPhase>
where
    F: FnMut() -> Result<()>,
    M: FnMut() -> Option<u64>,
{
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        function()?;
        samples.push(u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX));
        if let Some(current) = memory() {
            *peak = Some(peak.unwrap_or(0).max(current));
        }
    }
    samples.sort_unstable();
    let total_ns = samples.iter().map(|value| u128::from(*value)).sum::<u128>();
    let throughput_per_second = iterations as f64 * 1_000_000_000.0 / total_ns.max(1) as f64;
    Ok(BenchmarkPhase {
        name: name.into(),
        iterations,
        percentiles: Percentiles {
            p50_ns: percentile(&samples, 50),
            p95_ns: percentile(&samples, 95),
            p99_ns: percentile(&samples, 99),
        },
        throughput_per_second,
    })
}

fn percentile(samples: &[u64], percentile: usize) -> u64 {
    let index = samples
        .len()
        .saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1)
        .min(samples.len().saturating_sub(1));
    samples[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_all_phases_and_calculates_percentiles() {
        let runner = BenchmarkRunner {
            warmup_iterations: 2,
            measured_iterations: 4,
        };
        let report = runner
            .run(
                || Ok(()),
                || Ok(()),
                || Ok(()),
                || Ok(()),
                || Ok(()),
                || Some(42),
            )
            .unwrap();
        assert_eq!(
            report
                .phases
                .iter()
                .map(|phase| phase.name.as_str())
                .collect::<Vec<_>>(),
            [
                "build",
                "cache",
                "warmup",
                "steady-state",
                "post-process",
                "end-to-end"
            ]
        );
        assert_eq!(report.peak_memory_mib, Some(42));
        assert_eq!(report.to_json()["phases"].as_array().unwrap().len(), 6);
        assert!(
            report
                .phases
                .iter()
                .all(|phase| phase.throughput_per_second.is_finite())
        );
    }
}
