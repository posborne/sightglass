//! Common data definitions for sightglass.
//!
//! These are in one place, pulled out from the rest of the crates, so that many
//! different crates can serialize and deserialize data by using the same
//! definitions.

#![deny(missing_docs, missing_debug_implementations)]

mod format;
pub use format::Format;

use serde::{Deserialize, Serialize};
use std::{borrow::Cow, str::FromStr};

/// A single measurement, for example instructions retired when compiling a Wasm
/// module.
///
/// This is often used with the `'static` lifetime when recording measurements,
/// where we can use string literals for various fields. When reading data, it
/// can be used with a non-static lifetime to avoid many small allocations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Measurement<'a> {
    /// The CPU architecture on which this measurement was taken, for example
    /// "aarch64" or "x86_64".
    pub arch: Cow<'a, str>,

    /// The file path of the wasmtime benchmark API shared library used to
    /// record this measurement.
    pub engine: Cow<'a, str>,

    /// The file path of the Wasm benchmark program.
    pub wasm: Cow<'a, str>,

    /// The id of the process within which this measurement was taken.
    pub process: u32,

    /// This measurement was the `n`th measurement of this phase taken within a
    /// process.
    pub iteration: u32,

    /// The phase in a Wasm program's lifecycle that was measured: compilation,
    /// instantiation, or execution.
    pub phase: Phase,

    /// The event that was measured: micro seconds of wall time, CPU cycles
    /// executed, instructions retired, cache misses, etc.
    pub event: Cow<'a, str>,

    /// The event counts.
    ///
    /// The meaning and units depend on what the `event` is: it might be a count
    /// of microseconds if the event is wall time, or it might be a count of
    /// instructions if the event is instructions retired.
    pub count: u64,

    /// The engine-specific flags used when recording this measurement.
    /// This allows disambiguation when the same engine is used with different
    /// configurations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_flags: Option<Cow<'a, str>>,
}

/// A phase in a Wasm program's lifecycle.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub enum Phase {
    /// The compilation phase, where Wasm bytes are translated into native
    /// machine code.
    Compilation,
    /// The instantiation phase, where imports are provided and memories,
    /// globals, and tables are initialized.
    Instantiation,
    /// The execution phase, where functions are called and instructions are
    /// executed.
    Execution,
}

impl std::fmt::Display for Phase {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Phase::Compilation => write!(f, "compilation"),
            Phase::Instantiation => write!(f, "instantiation"),
            Phase::Execution => write!(f, "execution"),
        }
    }
}

impl FromStr for Phase {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_ascii_lowercase();
        match s.as_str() {
            "compilation" => Ok(Self::Compilation),
            "instantiation" => Ok(Self::Instantiation),
            "execution" => Ok(Self::Execution),
            _ => Err("invalid phase".into()),
        }
    }
}

/// A summary of grouped measurements.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Summary<'a> {
    /// The CPU architecture on which this measurement was taken, for example
    /// "aarch64" or "x86_64".
    pub arch: Cow<'a, str>,

    /// The file path of the wasmtime benchmark API shared library used to
    /// record this measurement.
    pub engine: Cow<'a, str>,

    /// The file path of the Wasm benchmark program.
    pub wasm: Cow<'a, str>,

    /// The phase in a Wasm program's lifecycle that was measured: compilation,
    /// instantiation, or execution.
    pub phase: Phase,

    /// The event that was measured: micro seconds of wall time, CPU cycles
    /// executed, instructions retired, cache misses, etc.
    pub event: Cow<'a, str>,

    /// The minimum value of the `count` field.
    pub min: u64,

    /// The maximum value of the `count` field.
    pub max: u64,

    /// The median value of the `count` field.
    pub median: u64,

    /// The arithmetic mean of the `count` field.
    pub mean: f64,

    /// The mean deviation (note: not standard deviation) of the `count` field.
    pub mean_deviation: f64,
}

/// The effect size (and confidence interval) between two different engines
/// (i.e. two different commits of Wasmtime).
///
/// This allows us to justify statements like "we are 99% confident that the new
/// register allocator is 13.6% faster (± 1.7%) than the old register
/// allocator."
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EffectSize<'a> {
    /// The CPU architecture on which this measurement was taken, for example
    /// "aarch64" or "x86_64".
    pub arch: Cow<'a, str>,

    /// The file path of the Wasm benchmark program.
    pub wasm: Cow<'a, str>,

    /// The phase in a Wasm program's lifecycle that was measured: compilation,
    /// instantiation, or execution.
    pub phase: Phase,

    /// The event that was measured: micro seconds of wall time, CPU cycles
    /// executed, instructions retired, cache misses, etc.
    pub event: Cow<'a, str>,

    /// The first engine being compared.
    ///
    /// This is the file path of the wasmtime benchmark API shared library used
    /// to record this measurement.
    pub a_engine: Cow<'a, str>,

    /// The first engine's result's arithmetic mean of the `count` field.
    pub a_mean: f64,

    /// The second engine being compared.
    ///
    /// This is the file path of the wasmtime benchmark API shared library used
    /// to record this measurement.
    pub b_engine: Cow<'a, str>,

    /// The second engine's result's arithmetic mean of the `count` field.
    pub b_mean: f64,

    /// The significance level for the confidence interval.
    ///
    /// This is always between 0.0 and 1.0. Typical values are 0.01 and 0.05
    /// which correspond to 99% confidence and 95% confidence respectively.
    pub significance_level: f64,

    /// The half-width confidence interval, i.e. the `i` in
    ///
    /// ```text
    /// b_mean - a_mean ± i
    /// ```
    pub half_width_confidence_interval: f64,
}

impl EffectSize<'_> {
    /// Is the difference between `self.a_mean` and `self.b_mean` statistically
    /// significant?
    pub fn is_significant(&self) -> bool {
        (self.a_mean - self.b_mean).abs() > self.half_width_confidence_interval.abs()
    }

    /// Return `b`'s speedup over `a` and the speedup's confidence interval.
    pub fn b_speed_up_over_a(&self) -> (f64, f64) {
        (
            self.b_mean / self.a_mean,
            self.half_width_confidence_interval / self.a_mean,
        )
    }

    /// Return `a`'s speed up over `b` and the speed up's confidence interval.
    pub fn a_speed_up_over_b(&self) -> (f64, f64) {
        (
            self.a_mean / self.b_mean,
            self.half_width_confidence_interval / self.b_mean,
        )
    }
}

/// Extract benchmark name from wasm file path.
///
/// This function handles various path formats commonly used in Sightglass:
/// - `./benchmarks/foo/benchmark.wasm` -> `foo`
/// - `benchmarks/bar/benchmark.wasm` -> `bar`
/// - `benchmarks/foo/bar.wasm` -> `foo/bar`
/// - `simple.wasm` -> `simple`
///
/// # Examples
///
/// ```
/// use sightglass_data::extract_benchmark_name;
///
/// assert_eq!(extract_benchmark_name("./benchmarks/foo/benchmark.wasm"), "foo");
/// assert_eq!(extract_benchmark_name("benchmarks/bar/benchmark.wasm"), "bar");
/// assert_eq!(extract_benchmark_name("benchmarks/foo/bar.wasm"), "foo/bar");
/// assert_eq!(extract_benchmark_name("simple.wasm"), "simple");
/// ```
pub fn extract_benchmark_name(wasm_path: &str) -> String {
    let mut path = wasm_path;

    // Remove prefix variations
    if let Some(stripped) = path.strip_prefix("./benchmarks/") {
        path = stripped;
    } else if let Some(stripped) = path.strip_prefix("benchmarks/") {
        path = stripped;
    }

    // Remove suffix variations
    if let Some(stripped) = path.strip_suffix("/benchmark.wasm") {
        path = stripped;
    } else if let Some(stripped) = path.strip_suffix(".wasm") {
        path = stripped;
    }

    path.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_benchmark_name() {
        // Standard benchmark.wasm format with ./benchmarks/ prefix
        assert_eq!(
            extract_benchmark_name("./benchmarks/foo/benchmark.wasm"),
            "foo"
        );

        // Standard benchmark.wasm format without ./ prefix
        assert_eq!(
            extract_benchmark_name("benchmarks/bar/benchmark.wasm"),
            "bar"
        );

        // Direct .wasm file in benchmarks directory
        assert_eq!(extract_benchmark_name("benchmarks/simple.wasm"), "simple");
        assert_eq!(extract_benchmark_name("./benchmarks/simple.wasm"), "simple");

        // Nested paths with .wasm extension
        assert_eq!(extract_benchmark_name("benchmarks/foo/bar.wasm"), "foo/bar");
        assert_eq!(
            extract_benchmark_name("./benchmarks/nested/path/test.wasm"),
            "nested/path/test"
        );

        // Simple .wasm files without benchmarks prefix
        assert_eq!(extract_benchmark_name("simple.wasm"), "simple");
        assert_eq!(extract_benchmark_name("test.wasm"), "test");

        // Edge cases - no extensions or prefixes
        assert_eq!(extract_benchmark_name("somefile"), "somefile");
        assert_eq!(extract_benchmark_name("path/to/file"), "path/to/file");
    }
}
