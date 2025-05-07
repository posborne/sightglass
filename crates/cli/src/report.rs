use std::{
    cell::Cell, collections::HashMap, fs::File, hash::{Hash, Hasher}, io::Write, path::{Path, PathBuf}
};

use polars::prelude::*;
use serde::Serialize;
use sightglass_data::Format;
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

    /// Name of the baseline to use; if not provided, the first input file
    /// will be used as the baseline.
    #[structopt(short = "b", long = "baseline-prefix")]
    baseline_prefix: Option<String>,

    /// Path to the file(s) that will be read from, or none to indicate stdin (default).
    #[structopt(min_values = 1)]
    input_files: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkStats {
    cv: f64,
    std: f64,
    mean: f64,
    median: f64,
    p50: f64,
    p25: f64,
    p75: f64,
    min: f64,
    max: f64,
    p25_delta_pct: f64,
    mean_delta_pct: f64,
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
    path.as_ref().file_stem().unwrap_or_else(|| path.as_ref().as_os_str()).to_string_lossy().to_string()
}

fn parse_input(format: Option<Format>, path: impl AsRef<Path>) -> PolarsResult<LazyFrame> {
    let format = format.or_else(|| match path.as_ref().extension()?.to_str()? {
        "json" => Some(Format::Json),
        "csv" => Some(Format::Csv { headers: Cell::new(true) }),
        _ => None,
    }).unwrap_or(Format::Json);

    let prefix = prefix_from_path(path.as_ref());
    
    let file = File::open(path)?;
    let df = match format {
        Format::Json => JsonReader::new(file).finish()?,
        Format::Csv { .. } => CsvReader::new(file).finish()?,
    };

    // filter/transform and add in additional prefix data
    Ok(df.lazy()
        .filter(col("phase").eq(lit("Execution")))
        .filter(col("event").eq(lit("cycles")))
        .with_column(
            // simplify the wasm file path to a benchmark name
            col("wasm")
                .str()
                .strip_prefix(lit("benchmarks/"))
                .str()
                .strip_suffix(lit("/benchmark.wasm"))
                .str()
                .strip_suffix(lit(".wasm"))
                .alias("benchmark"),
        )
        .with_column(lit(prefix.as_str()).alias("prefix")))
}

fn extract_agg_stats(
    agg_df: &DataFrame,
    baseline: Option<&BenchmarkStats>,
) -> Option<BenchmarkStats> {
    let (mut min, mut max, mut mean, mut p25, mut p50, mut p75, mut std) =
        (None, None, None, None, None, None, None);
    for column in agg_df.iter() {
        let f64_value: Option<f64> = column.try_f64().map(|ca| ca.get(0)).flatten();
        let i64_value: Option<i64> = column.try_i64().map(|ca| ca.get(0)).flatten();
        match column.name().as_str() {
            "mean" => mean = f64_value,
            "p50" => p50 = f64_value,
            "p25" => p25 = f64_value,
            "p75" => p75 = f64_value,
            "min" => min = i64_value,
            "max" => max = i64_value,
            "std" => std = f64_value,
            _ => (),
        }
    }

    let (pct_delta_p25, pct_delta_mean) = if let Some(baseline) = baseline {
        (
            (p25? - baseline.p25) / (p25? + baseline.p25) * 100.0,
            (mean? - baseline.mean) / (mean? + baseline.mean) * 100.0,
        )
    } else {
        (0.0, 0.0)
    };

    Some(BenchmarkStats {
        cv: (std? / mean?) * 100.0,
        std: std?,
        mean: mean?,
        median: p50?,
        p50: p50?,
        p25: p25?,
        p75: p75?,
        min: min? as f64,
        max: max? as f64,
        p25_delta_pct: pct_delta_p25,
        mean_delta_pct: pct_delta_mean,
    })
}

fn unique_column_values(df: &DataFrame, colname: &str) -> anyhow::Result<Vec<String>> {
    let unique_df = df
        .clone()
        .lazy()
        .select([col(colname)])
        .unique(None, UniqueKeepStrategy::First)
        .collect()?;
    let values_series = &unique_df[0];
    Ok(values_series
        .str()?
        .iter()
        .map(|v| v.unwrap().to_string())
        .collect())
}

impl ReportCommand {
    fn plot_benchmark(
        &self,
        bstats: &BenchmarkStats,
        benchmark: &str,
        df: DataFrame,
    ) -> anyhow::Result<String> {
        use vega_lite_4::{self as vl, VegaliteBuilder};

        // first, generate plot showing absolute performance of each prefix
        let cycles_chart = NormalizedSpecBuilder::default()
            .data(df.clone())
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
                            .legend(LegendBuilder::default().title("Prefix").build()?)
                            .build()?,
                    )
                    .build()?,
            )
            .build()?;

        // first, generate plot showing absolute performance of each prefix
        let with_pct_df = df
            .lazy()
            .with_column(
                (lit(100) * (col("count") - lit(bstats.p25))
                    / ((col("count") + lit(bstats.p25)) / lit(2)))
                .alias("p25_delta_pct"),
            )
            .collect()?;
        let pct_chart = NormalizedSpecBuilder::default()
            .data(with_pct_df)
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

    fn baseline_prefix(&self) -> &str {
        self.baseline_prefix
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or_else(|| {
                let first = self.input_files.first().unwrap();
                first.file_stem().unwrap().to_str().unwrap()
            })
    }

    fn aggregates_df(&self, full_df: &DataFrame) -> anyhow::Result<DataFrame> {
        // dataframe with just the baseline aggregates prefixed with 'baseline_'
        Ok(full_df
            .clone()
            .lazy()
            .group_by_stable(&[col("benchmark"), col("prefix")])
            .agg(&[
                col("count").mean().alias("mean"),
                col("count")
                    .quantile(lit(0.25), QuantileMethod::Nearest)
                    .alias("p25"),
                col("count")
                    .quantile(lit(0.50), QuantileMethod::Nearest)
                    .alias("p50"),
                col("count")
                    .quantile(lit(0.75), QuantileMethod::Nearest)
                    .alias("p75"),
                col("count").min().alias("min"),
                col("count").max().alias("max"),
                col("count").std(0).alias("std"),
            ])
            .collect()?)
    }

    fn stats_for_benchmark_prefix(
        &self,
        agg_df: &DataFrame,
        benchmark: &str,
        prefix: &str,
    ) -> anyhow::Result<BenchmarkStats> {
        let baseline_prefix = self.baseline_prefix();

        // could be cached
        let baseline_stats = if baseline_prefix == prefix {
            None
        } else {
            Some(self.stats_for_benchmark_prefix(agg_df, benchmark, &baseline_prefix)?)
        };

        let prefix_matches = col("prefix").eq(lit(prefix));
        let benchmark_matches = col("benchmark").eq(lit(benchmark));

        let stats = agg_df
            .clone()
            .lazy()
            .filter(prefix_matches.and(benchmark_matches))
            .collect()?;

        let stats = extract_agg_stats(&stats, baseline_stats.as_ref())
            .ok_or_else(|| anyhow::anyhow!("Failed to extract agg stats!"))?;

        Ok(stats)
    }

    fn compute_stats(&self, full_df: &DataFrame) -> anyhow::Result<SightglassStats> {
        let agg_df = self.aggregates_df(&full_df)?;
        let benchmarks = unique_column_values(&agg_df, "benchmark")?;
        let prefixes = unique_column_values(&agg_df, "prefix")?;

        let mut benchmarks_data: Vec<BenchmarkData> = Vec::default();
        for benchmark in benchmarks {
            // filter to only rows for this benchmark
            let stats_by_prefix = prefixes
                .iter()
                .map(|prefix| {
                    let stats = self.stats_for_benchmark_prefix(&agg_df, &benchmark, prefix)?;
                    Ok((prefix.clone(), stats))
                })
                .collect::<Vec<anyhow::Result<(String, BenchmarkStats)>>>()
                .into_iter()
                .collect::<anyhow::Result<HashMap<String, BenchmarkStats>>>()?;

            let baseline_stats = stats_by_prefix
                .iter()
                .find(|(p, _s)| self.baseline_prefix() == p.as_str())
                .map(|(_p, s)| s)
                .ok_or_else(|| anyhow::anyhow!("Unable to find baseline?"))?
                .clone();

            let benchmark_df = full_df
                .clone()
                .lazy()
                .filter(col("benchmark").eq(lit(benchmark.as_str())))
                .collect()?;
            let chart_json = self.plot_benchmark(&baseline_stats, &benchmark, benchmark_df)?;
            let id = {
                let mut h = std::hash::DefaultHasher::new();
                benchmark.hash(&mut h);
                h.finish()
            };
            benchmarks_data.push(BenchmarkData {
                name: benchmark,
                chart: Chart {
                    json: chart_json,
                    id,
                },
                stats_by_prefix,
                baseline_stats,
            });
        }

        let template_data = SightglassStats {
            baseline_prefix: self.baseline_prefix().to_string(),
            benchmarks: benchmarks_data,
        };

        Ok(template_data)
    }

    pub fn execute(&self) -> anyhow::Result<()> {
        let frames = self
            .input_files
            .iter()
            .map(|input| {
                let is_baseline = input.file_stem().unwrap() == self.baseline_prefix();
                let lazy_frame = parse_input(self.input_format.clone(), input)?
                    .lazy()
                    .with_column(lit(is_baseline).alias("is_baseline"));
                Ok(lazy_frame)
            })
            .collect::<anyhow::Result<Vec<LazyFrame>>>()?;

        let full_df = concat(&frames, UnionArgs::default())?.collect()?;
        let stats = self.compute_stats(&full_df)?;
        self.generate_html(stats)?;

        Ok(())
    }

    fn generate_html(&self, stats: SightglassStats) -> anyhow::Result<()> {
        let mut env = minijinja::Environment::new();
        env.add_template("report", TEMPLATE)?;
        env.add_filter("floatfmt", |v: f64| format!("{v:0.2}"));
        let template = env.get_template("report")?;

        let ctx = minijinja::context!(
            stats => stats,
        );

        let mut f = std::fs::File::create(&self.output_path)?;
        f.write_all(template.render(ctx)?.as_bytes())?;

        Ok(())
    }
}
