use std::{
    cell::Cell,
    collections::HashMap,
    fs::File,
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use serde::Serialize;
use sightglass_analysis::report_stats::{calculate_benchmark_stats, BenchmarkStats, ReportConfig};
use sightglass_data::{extract_benchmark_name, Format, Measurement, Phase};
use structopt::StructOpt;
use vega_lite_4::{
    AxisBuilder, ColorClassBuilder, EdEncodingBuilder, LegendBuilder, Mark, NormalizedSpecBuilder,
    XClassBuilder, YClassBuilder,
};

const TEMPLATE: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/report.jinja"));

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

    /// Primary event to analyze (default: cycles)
    #[structopt(long = "event", default_value = "cycles")]
    primary_event: String,

    /// Target phase to analyze (default: execution)
    #[structopt(long = "phase", default_value = "execution")]
    target_phase: Phase,

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


fn get_available_events(measurements: &[Measurement]) -> String {
    let mut events: Vec<&str> = measurements
        .iter()
        .map(|m| m.event.as_ref())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    events.sort();
    events.join(", ")
}

fn get_available_phases(measurements: &[Measurement]) -> String {
    let mut phases: Vec<String> = measurements
        .iter()
        .map(|m| format!("{:?}", m.phase))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    phases.sort();
    phases.join(", ")
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

    // Convert to owned strings for 'static lifetime and use prefix as engine name
    let filtered_measurements: Vec<Measurement<'static>> = measurements
        .into_iter()
        .map(|m| {
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
        measurements: &[&Measurement],
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

    fn compute_stats(&self, measurements: &[Measurement]) -> anyhow::Result<SightglassStats> {
        // Create ReportConfig from CLI arguments
        let config = ReportConfig::new()
            .with_event(&self.primary_event)
            .with_phase(self.target_phase)
            .with_significance_level(self.significance_level);

        let config = if let Some(ref baseline) = self.baseline_engine {
            config.with_baseline_prefix(baseline)
        } else {
            config
        };

        // Use the new calculate_benchmark_stats function with ReportConfig
        let benchmark_stats = calculate_benchmark_stats(measurements, &config)?;

        // Check if we found any matching data
        if benchmark_stats.is_empty() {
            anyhow::bail!(
                "No measurements found matching the specified criteria:\n\
                 - Event: {}\n\
                 - Phase: {:?}\n\
                 \n\
                 Available events: {}\n\
                 Available phases: {}",
                config.primary_event,
                config.target_phase,
                get_available_events(measurements),
                get_available_phases(measurements)
            );
        }

        let mut benchmarks_data: Vec<BenchmarkData> = Vec::new();

        // Determine baseline prefix for display
        let first_available_prefix = benchmark_stats
            .values()
            .next()
            .and_then(|stats| stats.keys().next())
            .map(|s| s.clone());

        let baseline_prefix_for_display = config
            .baseline_prefix
            .clone()
            .or(first_available_prefix)
            .unwrap_or_else(|| "baseline".to_string());

        // Convert the calculated benchmark stats to our display format
        for (benchmark_name, stats_by_prefix) in benchmark_stats {
            // Get baseline prefix from config or use first available
            let baseline_prefix = config
                .baseline_prefix
                .as_deref()
                .or_else(|| stats_by_prefix.keys().next().map(|s| s.as_str()))
                .unwrap_or("baseline");

            let baseline_stats = stats_by_prefix
                .get(baseline_prefix)
                .cloned()
                .unwrap_or_else(|| stats_by_prefix.values().next().unwrap().clone());

            // Get measurements for this benchmark that match our config filters
            let benchmark_measurements: Vec<&Measurement> = measurements
                .iter()
                .filter(|m| {
                    extract_benchmark_name(&m.wasm) == benchmark_name
                        && m.phase == config.target_phase
                        && m.event == config.primary_event
                })
                .collect();

            let chart_json =
                self.plot_benchmark(&baseline_stats, &benchmark_name, &benchmark_measurements)?;
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
            baseline_prefix: baseline_prefix_for_display,
            benchmarks: benchmarks_data,
        })
    }

    pub fn execute(&self) -> anyhow::Result<()> {
        let mut all_measurements = Vec::new();

        for input_file in &self.input_files {
            let measurements = parse_input(self.input_format.clone(), input_file)?;
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
