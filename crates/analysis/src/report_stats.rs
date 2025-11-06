use crate::summarize::{coefficient_of_variation, mean, percentile, std_deviation};
use sightglass_data::{extract_benchmark_name, Measurement, Phase};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during benchmark analysis
#[derive(Debug, Error)]
pub enum AnalysisError {
    #[error("No measurements available for analysis")]
    InsufficientData,
    #[error("Insufficient sample size: need at least {required} samples, got {actual}")]
    InsufficientSampleSize { required: usize, actual: usize },
    #[error("No measurements found for event '{event}' in phase '{phase:?}'")]
    NoMatchingMeasurements { phase: Phase, event: String },
    #[error("Invalid significance level: {level} (must be between 0 and 1)")]
    InvalidSignificanceLevel { level: f64 },
    #[error("Statistical analysis failed: {0}")]
    StatisticalError(String),
}

/// Configuration for benchmark report generation
#[derive(Debug, Clone)]
pub struct ReportConfig {
    pub primary_event: String,
    pub target_phase: Phase,
    pub significance_level: f64,
    pub baseline_prefix: Option<String>,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            primary_event: "cycles".to_string(),
            target_phase: Phase::Execution,
            significance_level: 0.05,
            baseline_prefix: None,
        }
    }
}

impl ReportConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_event(mut self, event: impl Into<String>) -> Self {
        self.primary_event = event.into();
        self
    }

    pub fn with_phase(mut self, phase: Phase) -> Self {
        self.target_phase = phase;
        self
    }

    pub fn with_significance_level(mut self, level: f64) -> Self {
        self.significance_level = level;
        self
    }

    pub fn with_baseline_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.baseline_prefix = Some(prefix.into());
        self
    }
}

/// Statistics calculated for a benchmark grouped by prefix.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchmarkStats {
    pub cv: f64,
    pub std: f64,
    pub mean: f64,
    pub median: f64, // p50 - consolidated into median
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
    config: &ReportConfig,
) -> Result<HashMap<String, HashMap<String, BenchmarkStats>>, AnalysisError> {
    // Early validation
    if measurements.is_empty() {
        return Err(AnalysisError::InsufficientData);
    }

    let mut results: HashMap<String, HashMap<String, BenchmarkStats>> = HashMap::new();

    // Group measurements by benchmark and prefix
    let mut grouped: HashMap<String, HashMap<String, Vec<u64>>> = HashMap::new();

    for measurement in measurements {
        // Only process measurements matching the configured phase and event
        if measurement.phase != config.target_phase || measurement.event != config.primary_event {
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

    // Check if we found any matching measurements
    if grouped.is_empty() {
        return Err(AnalysisError::NoMatchingMeasurements {
            phase: config.target_phase,
            event: config.primary_event.clone(),
        });
    }

    // Calculate statistics for each group
    for (benchmark, prefixes) in grouped {
        let mut benchmark_results = HashMap::new();

        // Determine baseline prefix from config or use first available
        let baseline_prefix = config
            .baseline_prefix
            .clone()
            .or_else(|| prefixes.keys().next().cloned())
            .unwrap_or_else(|| "baseline".to_string());

        let baseline_counts = prefixes.get(&baseline_prefix).cloned();
        let significance_level = config.significance_level;

        for (prefix, counts) in prefixes {
            let stats = if prefix == baseline_prefix {
                calculate_stats_for_measurements(&counts, None, significance_level)?
            } else {
                calculate_stats_for_measurements(
                    &counts,
                    baseline_counts.as_deref(),
                    significance_level,
                )?
            };
            benchmark_results.insert(prefix, stats);
        }

        results.insert(benchmark, benchmark_results);
    }

    Ok(results)
}


/// Extract prefix from measurement (could be enhanced to use actual prefix logic).
fn extract_prefix_from_measurement<'a>(measurement: &Measurement<'a>) -> String {
    // This would need to be implemented based on how prefixes are determined
    // For now, using engine name as a placeholder
    measurement.engine.to_string()
}

/// Calculate statistics for a group of measurements.
fn calculate_stats_for_measurements(
    measurements: &[u64],
    baseline_measurements: Option<&[u64]>,
    significance_level: f64,
) -> Result<BenchmarkStats, AnalysisError> {
    // Validate minimum sample size for measurements
    if measurements.len() < 3 {
        return Err(AnalysisError::InsufficientSampleSize {
            required: 3,
            actual: measurements.len(),
        });
    }

    // Validate baseline sample size if provided
    if let Some(baseline) = baseline_measurements {
        if baseline.len() < 3 {
            return Err(AnalysisError::InsufficientSampleSize {
                required: 3,
                actual: baseline.len(),
            });
        }
    }

    // Validate significance level
    if !(0.0..=1.0).contains(&significance_level) {
        return Err(AnalysisError::InvalidSignificanceLevel {
            level: significance_level,
        });
    }

    let sorted_measurements = measurements.to_vec();
    let mean_val = mean(measurements);
    let std_val = std_deviation(measurements);
    let p25_val = percentile(&mut sorted_measurements.clone(), 25.0);
    let p50_val = percentile(&mut sorted_measurements.clone(), 50.0);
    let p75_val = percentile(&mut sorted_measurements.clone(), 75.0);
    let min_val = *measurements.iter().min().unwrap() as f64;
    let max_val = *measurements.iter().max().unwrap() as f64;
    let cv_val = coefficient_of_variation(measurements);

    // Calculate statistical significance if we have baseline data
    let (
        p25_delta_pct,
        mean_delta_pct,
        is_significant,
        confidence_interval_half_width,
        effect_size_mean_diff,
        speedup_ratio,
        speedup_confidence_interval,
    ) = if let Some(baseline_measurements) = baseline_measurements {
        let baseline_mean = mean(baseline_measurements);
        let p25_delta = (p25_val - percentile(&mut baseline_measurements.to_vec(), 25.0))
            / (p25_val + percentile(&mut baseline_measurements.to_vec(), 25.0))
            * 100.0;
        let mean_delta = (mean_val - baseline_mean) / (mean_val + baseline_mean) * 100.0;

        // Use behrens-fisher for statistical significance
        let current_stats: behrens_fisher::Stats = measurements.iter().map(|&c| c as f64).collect();
        let baseline_stats: behrens_fisher::Stats =
            baseline_measurements.iter().map(|&c| c as f64).collect();

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

    Ok(BenchmarkStats {
        cv: cv_val,
        std: std_val,
        mean: mean_val,
        median: p50_val, // p50 consolidated into median
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_calculate_stats_for_measurements() {
        let measurements = vec![1, 2, 3, 4, 5];
        let stats = calculate_stats_for_measurements(&measurements, None, 0.05)
            .expect("Should calculate stats successfully");

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 3.0); // changed from p50 to median
        assert!(stats.cv > 0.0);
    }

    #[test]
    fn test_insufficient_sample_size() {
        let measurements = vec![1, 2];
        let result = calculate_stats_for_measurements(&measurements, None, 0.05);
        assert!(matches!(result, Err(AnalysisError::InsufficientSampleSize { required: 3, actual: 2 })));
    }

    #[test]
    fn test_invalid_significance_level() {
        let measurements = vec![1, 2, 3, 4, 5];
        let result = calculate_stats_for_measurements(&measurements, None, 1.5);
        assert!(matches!(result, Err(AnalysisError::InvalidSignificanceLevel { level }) if level == 1.5));
    }
}
