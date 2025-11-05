use crate::summarize::{coefficient_of_variation, mean, percentile, std_deviation};
use sightglass_data::Measurement;
use std::collections::HashMap;

/// Statistics calculated for a benchmark grouped by prefix.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkStats {
    pub cv: f64,
    pub std: f64,
    pub mean: f64,
    pub median: f64,
    pub p50: f64,
    pub p25: f64,
    pub p75: f64,
    pub min: f64,
    pub max: f64,
    pub p25_delta_pct: f64,
    pub mean_delta_pct: f64,
    // Statistical significance fields
    pub is_significant: bool,
    pub significance_level: f64,
    pub confidence_interval_half_width: Option<f64>,
    pub effect_size_mean_diff: Option<f64>,
    pub speedup_ratio: Option<f64>,
    pub speedup_confidence_interval: Option<f64>,
}

/// Data structure representing aggregated statistics for a benchmark.
#[derive(Debug)]
pub struct BenchmarkData {
    pub name: String,
    pub stats_by_prefix: HashMap<String, BenchmarkStats>,
    pub baseline_stats: BenchmarkStats,
}

/// Calculate statistics for all benchmarks grouped by prefix.
pub fn calculate_benchmark_stats<'a>(
    measurements: &[Measurement<'a>],
    baseline_prefix: &str,
) -> HashMap<String, HashMap<String, BenchmarkStats>> {
    let mut results: HashMap<String, HashMap<String, BenchmarkStats>> = HashMap::new();

    // Group measurements by benchmark and prefix
    let mut grouped: HashMap<String, HashMap<String, Vec<u64>>> = HashMap::new();

    for measurement in measurements {
        // Only process execution phase cycles measurements
        if measurement.phase != sightglass_data::Phase::Execution || measurement.event != "cycles" {
            continue;
        }

        let benchmark = extract_benchmark_name(&measurement.wasm);
        let prefix = extract_prefix_from_measurement(measurement);

        grouped
            .entry(benchmark)
            .or_default()
            .entry(prefix)
            .or_default()
            .push(measurement.count);
    }

    // Calculate statistics for each group
    for (benchmark, prefixes) in grouped {
        let mut benchmark_results = HashMap::new();
        let baseline_counts = prefixes.get(baseline_prefix).cloned();
        let significance_level = 0.05; // Default significance level

        for (prefix, counts) in prefixes {
            let stats = if prefix == baseline_prefix {
                calculate_stats_for_counts(&counts, None, significance_level, None)
            } else {
                calculate_stats_for_counts(
                    &counts,
                    baseline_counts.as_deref(),
                    significance_level,
                    None,
                )
            };
            benchmark_results.insert(prefix, stats);
        }

        results.insert(benchmark, benchmark_results);
    }

    results
}

/// Extract benchmark name from wasm file path.
fn extract_benchmark_name(wasm_path: &str) -> String {
    wasm_path
        .strip_prefix("benchmarks/")
        .unwrap_or(wasm_path)
        .strip_suffix("/benchmark.wasm")
        .unwrap_or(wasm_path)
        .strip_suffix(".wasm")
        .unwrap_or(wasm_path)
        .to_string()
}

/// Extract prefix from measurement (could be enhanced to use actual prefix logic).
fn extract_prefix_from_measurement<'a>(measurement: &Measurement<'a>) -> String {
    // This would need to be implemented based on how prefixes are determined
    // For now, using engine name as a placeholder
    measurement.engine.to_string()
}

/// Calculate statistics for a group of count measurements.
fn calculate_stats_for_counts(
    counts: &[u64],
    baseline_counts: Option<&[u64]>,
    significance_level: f64,
    _effect_size_data: Option<&sightglass_data::EffectSize>,
) -> BenchmarkStats {
    if counts.is_empty() {
        return BenchmarkStats {
            cv: 0.0,
            std: 0.0,
            mean: 0.0,
            median: 0.0,
            p50: 0.0,
            p25: 0.0,
            p75: 0.0,
            min: 0.0,
            max: 0.0,
            p25_delta_pct: 0.0,
            mean_delta_pct: 0.0,
            is_significant: false,
            significance_level: 0.05,
            confidence_interval_half_width: None,
            effect_size_mean_diff: None,
            speedup_ratio: None,
            speedup_confidence_interval: None,
        };
    }

    let sorted_counts = counts.to_vec();
    let mean_val = mean(counts);
    let std_val = std_deviation(counts);
    let p25_val = percentile(&mut sorted_counts.clone(), 25.0);
    let p50_val = percentile(&mut sorted_counts.clone(), 50.0);
    let p75_val = percentile(&mut sorted_counts.clone(), 75.0);
    let min_val = *counts.iter().min().unwrap() as f64;
    let max_val = *counts.iter().max().unwrap() as f64;
    let cv_val = coefficient_of_variation(counts);

    // Calculate statistical significance if we have baseline data
    let (
        p25_delta_pct,
        mean_delta_pct,
        is_significant,
        confidence_interval_half_width,
        effect_size_mean_diff,
        speedup_ratio,
        speedup_confidence_interval,
    ) = if let Some(baseline_counts) = baseline_counts {
        let baseline_mean = mean(baseline_counts);
        let p25_delta = (p25_val - percentile(&mut baseline_counts.to_vec(), 25.0))
            / (p25_val + percentile(&mut baseline_counts.to_vec(), 25.0))
            * 100.0;
        let mean_delta = (mean_val - baseline_mean) / (mean_val + baseline_mean) * 100.0;

        // Use behrens-fisher for statistical significance
        let current_stats: behrens_fisher::Stats = counts.iter().map(|&c| c as f64).collect();
        let baseline_stats: behrens_fisher::Stats =
            baseline_counts.iter().map(|&c| c as f64).collect();

        if let Ok(ci) = behrens_fisher::confidence_interval(
            1.0 - significance_level,
            current_stats,
            baseline_stats,
        ) {
            let mean_diff = mean_val - baseline_mean;
            let is_sig = mean_diff.abs() > ci.abs();
            let speedup = if baseline_mean > 0.0 {
                mean_val / baseline_mean
            } else {
                1.0
            };
            let speedup_ci = if baseline_mean > 0.0 {
                ci / baseline_mean
            } else {
                0.0
            };

            (
                p25_delta,
                mean_delta,
                is_sig,
                Some(ci),
                Some(mean_diff),
                Some(speedup),
                Some(speedup_ci),
            )
        } else {
            (p25_delta, mean_delta, false, None, None, None, None)
        }
    } else {
        (0.0, 0.0, false, None, None, None, None)
    };

    BenchmarkStats {
        cv: cv_val,
        std: std_val,
        mean: mean_val,
        median: p50_val,
        p50: p50_val,
        p25: p25_val,
        p75: p75_val,
        min: min_val,
        max: max_val,
        p25_delta_pct,
        mean_delta_pct,
        is_significant,
        significance_level,
        confidence_interval_half_width,
        effect_size_mean_diff,
        speedup_ratio,
        speedup_confidence_interval,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sightglass_data::Phase;

    #[test]
    fn test_extract_benchmark_name() {
        assert_eq!(
            extract_benchmark_name("benchmarks/foo/benchmark.wasm"),
            "foo"
        );
        assert_eq!(extract_benchmark_name("benchmarks/bar.wasm"), "bar");
        assert_eq!(extract_benchmark_name("simple.wasm"), "simple");
    }

    #[test]
    fn test_calculate_stats_for_counts() {
        let counts = vec![1, 2, 3, 4, 5];
        let stats = calculate_stats_for_counts(&counts, None);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.p50, 3.0);
        assert!(stats.cv > 0.0);
    }
}
