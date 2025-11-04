use std::{
    cell::Cell,
    collections::HashMap,
    fs::File,
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use serde::Serialize;
use sightglass_analysis::{effect_size, report_stats::BenchmarkStats};
use sightglass_data::{EffectSize, Format, Measurement, Phase};
use structopt::StructOpt;
use vega_lite_4::{
    AxisBuilder, ColorClassBuilder, EdEncodingBuilder, LegendBuilder, Mark, NormalizedSpecBuilder,
    XClassBuilder, YClassBuilder,
};

const TEMPLATE: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/report.jinja"));

/// Generate an HTML report for a given set of raw inputs
#[derive(Debug, StructOpt)]
#[structopt(name = "report")]
pub struct ReportCommand {
    /// The format of the input data. Either 'json' or 'csv'; if not provided
    /// then we will attempt to infer it from provided filenames (default: json).
    #[structopt(short = "i", long = "input-format")]
    input_format: Option<Format>,

    /// Output HTML file path
    #[structopt(short = "o", long = "output-file", default_value = "report.html")]
    output_path: PathBuf,

    /// Name of the baseline to use; if not provided, the first engine encountered
    /// in the ordered input files will be used.
    #[structopt(short = "b", long = "baseline-engine")]
    baseline_engine: Option<String>,

    /// Significance level for statistical tests (default: 0.05 for 95% confidence)
    #[structopt(long = "significance-level", default_value = "0.05")]
    significance_level: f64,

    /// Path to the file(s) that will be read from, or none to indicate stdin (default).
    #[structopt(min_values = 1)]
    input_files: Vec<PathBuf>,
}

#[derive(Debug, Serialize)]
struct Chart {
    json: String,
    id: u64,
}

#[derive(Debug, Serialize)]
struct BenchmarkData {
    name: String,
    chart: Chart,
    stats_by_prefix: HashMap<String, BenchmarkStats>,
    baseline_stats: BenchmarkStats,
}

#[derive(Debug, Serialize)]
struct SightglassStats {
    baseline_prefix: String,
    benchmarks: Vec<BenchmarkData>,
}

fn prefix_from_path(path: impl AsRef<Path>) -> String {
    path.as_ref()
        .file_stem()
        .unwrap_or_else(|| path.as_ref().as_os_str())
        .to_string_lossy()
        .to_string()
}

fn extract_prefix_from_engine(engine: &str) -> String {
    // Since we now use the filename as the engine name directly, just return it
    engine.to_string()
}

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

fn find_effect_size<'a>(
    effect_sizes: &'a [EffectSize<'a>],
    benchmark: &str,
    engine_a: &str,
    engine_b: &str,
) -> Option<&'a EffectSize<'a>> {
    effect_sizes.iter().find(|es| {
        let benchmark_name = extract_benchmark_name(&es.wasm);
        benchmark_name == benchmark
            && (es.a_engine.as_ref() == engine_a && es.b_engine.as_ref() == engine_b
                || es.a_engine.as_ref() == engine_b && es.b_engine.as_ref() == engine_a)
    })
}

fn calculate_stats_for_counts(
    counts: &[u64],
    baseline_counts: Option<&[u64]>,
    significance_level: f64,
    effect_size_data: Option<&EffectSize>,
) -> BenchmarkStats {
    use sightglass_analysis::summarize::{
        coefficient_of_variation, mean, percentile, std_deviation,
    };

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
            significance_level,
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
        let baseline_p25 = percentile(&mut baseline_counts.to_vec(), 25.0);
        let p25_delta = (p25_val - baseline_p25) / (p25_val + baseline_p25) * 100.0;
        let mean_delta = (mean_val - baseline_mean) / (mean_val + baseline_mean) * 100.0;

        // Use effect size data if available
        let (is_sig, ci, mean_diff, speedup, speedup_ci) = if let Some(es) = effect_size_data {
            let (speedup_val, speedup_ci_val) = if es.a_mean < es.b_mean {
                es.b_speed_up_over_a()
            } else {
                es.a_speed_up_over_b()
            };

            (
                es.is_significant(),
                Some(es.half_width_confidence_interval),
                Some(es.b_mean - es.a_mean),
                Some(speedup_val),
                Some(speedup_ci_val),
            )
        } else {
            // Fallback calculation when no effect size data available
            let mean_diff = mean_val - baseline_mean;
            let speedup = if baseline_mean > 0.0 {
                mean_val / baseline_mean
            } else {
                1.0
            };
            (false, None, Some(mean_diff), Some(speedup), None)
        };

        (
            p25_delta, mean_delta, is_sig, ci, mean_diff, speedup, speedup_ci,
        )
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

fn parse_input(
    format: Option<Format>,
    path: impl AsRef<Path>,
) -> anyhow::Result<Vec<Measurement<'static>>> {
    let format = format
        .or_else(|| match path.as_ref().extension()?.to_str()? {
            "json" => Some(Format::Json),
            "csv" => Some(Format::Csv {
                headers: Cell::new(true),
            }),
            _ => None,
        })
        .unwrap_or(Format::Json);

    let prefix = prefix_from_path(path.as_ref());
    let file = File::open(&path)?;

    let measurements: Vec<Measurement> = match format {
        Format::Json => {
            let reader = BufReader::new(file);

            // First try to parse as a JSON array
            if let Ok(measurements) = serde_json::from_reader::<_, Vec<Measurement>>(reader) {
                measurements
            } else {
                // Fall back to JSONL format (one JSON object per line)
                // Need to reopen the file since the reader was consumed
                let file_retry = File::open(path.as_ref())?;
                let reader_retry = BufReader::new(file_retry);
                let mut measurements = Vec::new();
                for line in BufRead::lines(reader_retry) {
                    let line = line?;
                    eprintln!("line = {:?}", line);
                    if let Ok(measurement) = serde_json::from_str::<Measurement>(&line) {
                        measurements.push(measurement);
                    }
                }
                measurements
            }
        }
        Format::Csv { .. } => {
            let mut reader = csv::Reader::from_reader(file);
            let mut measurements = Vec::new();
            for result in reader.deserialize() {
                let measurement: Measurement = result?;
                measurements.push(measurement);
            }
            measurements
        }
    };

    // Filter to execution phase cycles only and add prefix
    let filtered_measurements: Vec<Measurement<'static>> = measurements
        .into_iter()
        .filter(|m| m.phase == Phase::Execution && m.event == "cycles")
        .map(|m| {
            // Convert to owned strings for 'static lifetime and use prefix as engine name
            Measurement {
                arch: std::borrow::Cow::Owned(m.arch.into_owned()),
                engine: std::borrow::Cow::Owned(prefix.clone()),
                wasm: std::borrow::Cow::Owned(m.wasm.into_owned()),
                process: m.process,
                iteration: m.iteration,
                phase: m.phase,
                event: std::borrow::Cow::Owned(m.event.into_owned()),
                count: m.count,
            }
        })
        .collect();

    Ok(filtered_measurements)
}

impl ReportCommand {
    fn plot_benchmark(
        &self,
        bstats: &BenchmarkStats,
        benchmark: &str,
        measurements: &[Measurement],
    ) -> anyhow::Result<String> {
        use vega_lite_4::{self as vl, VegaliteBuilder};

        // Create a simple data structure for vega-lite
        #[derive(Debug, serde::Serialize)]
        struct ChartDataPoint {
            count: u64,
            prefix: String,
            p25_delta_pct: f64,
        }

        let chart_data: Vec<ChartDataPoint> = measurements
            .iter()
            .map(|m| ChartDataPoint {
                count: m.count,
                prefix: extract_prefix_from_engine(&m.engine),
                p25_delta_pct: (100.0 * (m.count as f64 - bstats.p25)
                    / ((m.count as f64 + bstats.p25) / 2.0)),
            })
            .collect();
        let cycles_chart = NormalizedSpecBuilder::default()
            .data(&chart_data)
            .mark(Mark::Boxplot)
            .encoding(
                EdEncodingBuilder::default()
                    .x(XClassBuilder::default()
                        .field("count")
                        .position_def_type(vl::Type::Quantitative)
                        .axis(AxisBuilder::default().title("cycles").build()?)
                        .build()?)
                    .y(YClassBuilder::default()
                        .field("prefix")
                        .position_def_type(vl::Type::Nominal)
                        .build()?)
                    .color(
                        ColorClassBuilder::default()
                            .field("prefix")
                            .legend(LegendBuilder::default().title("Engine").build()?)
                            .build()?,
                    )
                    .build()?,
            )
            .build()?;
        let pct_chart = NormalizedSpecBuilder::default()
            .data(&chart_data)
            .mark(Mark::Boxplot)
            .encoding(
                EdEncodingBuilder::default()
                    .x(XClassBuilder::default()
                        .field("p25_delta_pct")
                        .position_def_type(vl::Type::Quantitative)
                        .axis(
                            AxisBuilder::default()
                                .title("delta p25 as percentage")
                                .build()?,
                        )
                        .build()?)
                    .y(YClassBuilder::default()
                        .field("prefix")
                        .position_def_type(vl::Type::Nominal)
                        .build()?)
                    .color(
                        ColorClassBuilder::default()
                            .field("prefix")
                            .legend(LegendBuilder::default().title("Prefix").build()?)
                            .build()?,
                    )
                    .build()?,
            )
            .build()?;

        let chart = VegaliteBuilder::default()
            .title(benchmark)
            .hconcat(vec![cycles_chart, pct_chart])
            .build()?;

        Ok(chart.to_string()?)
    }

    fn baseline_prefix(&self, measurements: &[Measurement]) -> String {
        if let Some(baseline) = &self.baseline_engine {
            baseline.clone()
        } else {
            // Use the first engine found in measurements as the baseline
            measurements
                .iter()
                .map(|m| extract_prefix_from_engine(&m.engine))
                .next()
                .unwrap_or_else(|| "baseline".to_string())
        }
    }

    fn compute_stats(&self, measurements: &[Measurement]) -> anyhow::Result<SightglassStats> {
        eprintln!("compute_stats! measurements={measurements:?}");

        // First calculate effect sizes for all measurements
        let effect_sizes = effect_size::calculate(self.significance_level, measurements)?;
        // Group measurements by benchmark name
        let mut benchmark_groups: HashMap<String, Vec<&Measurement>> = HashMap::new();

        for measurement in measurements {
            eprintln!("measurent: {measurement:?}");
            let benchmark = extract_benchmark_name(&measurement.wasm);
            benchmark_groups
                .entry(benchmark)
                .or_insert_with(Vec::new)
                .push(measurement);
        }

        let baseline_prefix = self.baseline_prefix(measurements);
        let mut benchmarks_data: Vec<BenchmarkData> = Vec::new();

        for (benchmark_name, benchmark_measurements) in benchmark_groups {
            // Group measurements by prefix for this benchmark
            let mut prefix_groups: HashMap<String, Vec<u64>> = HashMap::new();

            for measurement in &benchmark_measurements {
                let prefix = extract_prefix_from_engine(&measurement.engine);
                prefix_groups
                    .entry(prefix)
                    .or_insert_with(Vec::new)
                    .push(measurement.count);
            }

            // Calculate stats for each prefix
            let mut stats_by_prefix = HashMap::new();
            let baseline_counts = prefix_groups.get(&baseline_prefix);
            let significance_level = self.significance_level;

            for (prefix, counts) in &prefix_groups {
                eprintln!("prefix={prefix}, counts.len()={}", counts.len());
                let effect_size_data = if prefix != &baseline_prefix {
                    find_effect_size(&effect_sizes, &benchmark_name, prefix, &baseline_prefix)
                } else {
                    None
                };

                let stats = if prefix == &baseline_prefix {
                    calculate_stats_for_counts(counts, None, significance_level, None)
                } else {
                    calculate_stats_for_counts(
                        counts,
                        baseline_counts.map(|c| c.as_slice()),
                        significance_level,
                        effect_size_data,
                    )
                };
                stats_by_prefix.insert(prefix.clone(), stats);
            }

            let baseline_stats = stats_by_prefix
                .get(&baseline_prefix)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Unable to find baseline stats for benchmark: {}",
                        benchmark_name
                    )
                })?
                .clone();

            // Convert Vec<&Measurement> to Vec<Measurement> for the plot function
            let owned_measurements: Vec<Measurement> = benchmark_measurements
                .iter()
                .map(|m| (*m).clone())
                .collect();
            let chart_json =
                self.plot_benchmark(&baseline_stats, &benchmark_name, &owned_measurements)?;
            let id = {
                let mut h = std::hash::DefaultHasher::new();
                benchmark_name.hash(&mut h);
                h.finish()
            };

            benchmarks_data.push(BenchmarkData {
                name: benchmark_name,
                chart: Chart {
                    json: chart_json,
                    id,
                },
                stats_by_prefix,
                baseline_stats,
            });
        }

        // Sort benchmarks by name for consistent ordering
        benchmarks_data.sort_by(|a, b| a.name.cmp(&b.name));

        Ok(SightglassStats {
            baseline_prefix: baseline_prefix.to_string(),
            benchmarks: benchmarks_data,
        })
    }

    pub fn execute(&self) -> anyhow::Result<()> {
        let mut all_measurements = Vec::new();

        eprintln!("input_files: {:?}", self.input_files);
        for input_file in &self.input_files {
            let measurements = parse_input(self.input_format.clone(), input_file)?;
            eprintln!("measurements for {input_file:?} = {:?}", measurements);
            all_measurements.extend(measurements);
        }

        let stats = self.compute_stats(&all_measurements)?;
        self.generate_html(stats)?;

        Ok(())
    }

    fn generate_html(&self, stats: SightglassStats) -> anyhow::Result<()> {
        let mut env = minijinja::Environment::new();
        env.add_template("report", TEMPLATE)?;
        env.add_filter("floatfmt", |v: f64| format!("{v:0.2}"));
        env.add_filter("intfmt", |v: f64| format!("{:.0}", v));
        let template = env.get_template("report")?;

        let ctx = minijinja::context!(
            stats => stats,
        );

        let mut f = std::fs::File::create(&self.output_path)?;
        f.write_all(template.render(ctx)?.as_bytes())?;

        Ok(())
    }
}
