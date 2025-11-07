use super::util::sightglass_cli;
use assert_cmd::prelude::*;
use predicates::prelude::*;
use scraper::{Html, Selector};
use std::fs;
use tempfile::TempDir;

// Helper function to get the original single-engine test data file path at compile time
fn test_results_json() -> &'static str {
    // CARGO_MANIFEST_DIR points to the directory containing Cargo.toml for this crate
    // For crates/cli, this will be /path/to/project/crates/cli
    // We want to get to crates/cli/tests/results.json
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/results.json")
}

// Helper functions for multi-engine test data (preferred for most tests)
fn multi_engine_v38_json() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/multi_engine_v38.json")
}

fn multi_engine_v38_epoch_json() -> &'static str {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/multi_engine_v38_epoch.json"
    )
}

// Helper function to get the original baseline engine path for legacy single-engine data
fn baseline_engine_path() -> &'static str {
    // This matches the engine path used in the original single-engine test data exactly
    "../../engines/wasmtime/libengine.so"
}

// Helper function to run report command and return HTML content
fn run_report_command(args: &[&str]) -> String {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");

    let mut cmd = sightglass_cli();
    cmd.arg("report").arg("--output-file").arg(&output_path);

    for arg in args {
        cmd.arg(arg);
    }

    cmd.assert().success();
    fs::read_to_string(&output_path).unwrap()
}

// Helper function to run report command with custom significance level
fn run_report_with_significance(level: &str, input_files: &[&str]) -> String {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");

    let mut cmd = sightglass_cli();
    cmd.arg("report")
        .arg("--significance-level")
        .arg(level)
        .arg("--output-file")
        .arg(&output_path);

    for file in input_files {
        cmd.arg(file);
    }

    cmd.assert().success();
    fs::read_to_string(&output_path).unwrap()
}

// Helper function to run report command with custom baseline
fn run_report_with_baseline(baseline: &str, input_files: &[&str]) -> String {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");

    let mut cmd = sightglass_cli();
    cmd.arg("report")
        .arg("--baseline-engine")
        .arg(baseline)
        .arg("--output-file")
        .arg(&output_path);

    for file in input_files {
        cmd.arg(file);
    }

    cmd.assert().success();
    fs::read_to_string(&output_path).unwrap()
}

// Helper function to count elements matching a CSS selector
fn count_css_elements(html_content: &str, selector: &str) -> usize {
    let document = Html::parse_document(html_content);
    let css_selector = Selector::parse(selector).unwrap();
    document.select(&css_selector).count()
}

#[test]
fn report_help() {
    sightglass_cli()
        .arg("report")
        .arg("--help")
        .assert()
        .success()
        .stdout(
            predicate::str::contains("Generate an HTML report for a given set of raw inputs")
                .and(predicate::str::contains("--input-format"))
                .and(predicate::str::contains("--output-file"))
                .and(predicate::str::contains("--baseline-engine"))
                .and(predicate::str::contains("--significance-level"))
                .and(predicate::str::contains("--event"))
                .and(predicate::str::contains("--phase")),
        );
}

#[test]
fn report_with_existing_json() {
    // Use multi-engine data to showcase the report's real value
    let html_content =
        run_report_command(&[multi_engine_v38_json(), multi_engine_v38_epoch_json()]);

    // Check for specific performance regression indicators based on our test data
    // The epoch-interruption engine should be slower for all benchmarks
    let slower_in_table = count_css_elements(&html_content, "table .slower");
    assert_eq!(
        slower_in_table, 3,
        "Should have exactly 3 'slower' indicators in the table for the three benchmarks"
    );

    // Verify specific performance degradation percentages from our data
    assert!(html_content.contains("19.44% slower") || html_content.contains("20.99% slower"));
    assert!(html_content.contains("7.52% slower"));

    // Check for statistical significance indicators
    let table_count = count_css_elements(&html_content, "#results-table");
    assert_eq!(
        table_count, 1,
        "Should have exactly one results table with ID 'results-table'"
    );

    // Check for baseline engine designation
    assert!(html_content.contains("(baseline)"));

    // Verify Vega-Lite data contains our specific benchmark titles
    assert!(html_content.contains("\"title\":\"bz2\""));
    assert!(html_content.contains("\"title\":\"pulldown-cmark\""));
    assert!(html_content.contains("\"title\":\"spidermonkey\""));
}

#[test]
fn report_format_inference_from_extension() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");
    let json_path = temp_dir.path().join("data.json");

    // Copy existing test data to file with .json extension
    fs::copy(test_results_json(), &json_path).unwrap();

    // Test that format is inferred from .json extension
    sightglass_cli()
        .arg("report")
        .arg("--output-file")
        .arg(&output_path)
        .arg(&json_path)
        .assert()
        .success();
}

#[test]
fn report_with_custom_baseline_engine() {
    let html_content = run_report_with_baseline(baseline_engine_path(), &[test_results_json()]);

    // Verify the custom baseline engine is properly marked
    assert!(html_content.contains("libengine.so"));
    assert!(html_content.contains("(baseline)"));
}

#[test]
fn report_with_custom_significance_level() {
    let html_content = run_report_with_significance("0.01", &[test_results_json()]);

    // Should show 99.00% confidence (100 - 1 = 99) in the methodology section
    assert!(html_content.contains("99.00%"));
    assert!(html_content.contains("confidence"));
}

#[test]
fn report_with_custom_event() {
    run_report_command(&["--event", "nanoseconds", test_results_json()]);
}

#[test]
fn report_with_custom_phase() {
    run_report_command(&["--phase", "compilation", test_results_json()]);
}

#[test]
fn report_error_invalid_significance_level() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");

    sightglass_cli()
        .arg("report")
        .arg("--significance-level")
        .arg("1.5") // Invalid: must be between 0 and 1
        .arg("--output-file")
        .arg(&output_path)
        .arg(test_results_json())
        .assert()
        .failure()
        .stderr(predicate::str::contains("Invalid significance level"));
}

#[test]
fn report_error_missing_input_files() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");

    sightglass_cli()
        .arg("report")
        .arg("--output-file")
        .arg(&output_path)
        .assert()
        .failure();
}

#[test]
fn report_error_nonexistent_input_file() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");

    sightglass_cli()
        .arg("report")
        .arg("--output-file")
        .arg(&output_path)
        .arg("nonexistent.json")
        .assert()
        .failure();
}

#[test]
fn report_error_invalid_event() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");

    sightglass_cli()
        .arg("report")
        .arg("--event")
        .arg("nonexistent_event")
        .arg("--output-file")
        .arg(&output_path)
        .arg(test_results_json())
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("No measurements found")
                .and(predicate::str::contains("nonexistent_event")),
        );
}

#[test]
fn report_multiple_input_files() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("report.html");
    let json_path1 = temp_dir.path().join("data1.json");
    let json_path2 = temp_dir.path().join("data2.json");

    // Copy test data to multiple files
    fs::copy(test_results_json(), &json_path1).unwrap();
    fs::copy(test_results_json(), &json_path2).unwrap();

    sightglass_cli()
        .arg("report")
        .arg("--output-file")
        .arg(&output_path)
        .arg(&json_path1)
        .arg(&json_path2)
        .assert()
        .success();
}

#[test]
fn report_default_output_filename() {
    let temp_dir = TempDir::new().unwrap();
    let original_dir = std::env::current_dir().unwrap();

    // Change to temp directory so default output goes there
    std::env::set_current_dir(&temp_dir).unwrap();

    sightglass_cli()
        .arg("report")
        .arg(test_results_json())
        .assert()
        .success();

    // Should create report.html in current directory
    let default_output = temp_dir.path().join("report.html");
    assert!(default_output.exists());

    // Restore original directory
    std::env::set_current_dir(original_dir).unwrap();
}

#[test]
fn report_marks_inconsistent_high_cv() {
    // Use multi-engine data which should have some high CV measurements
    let html_content =
        run_report_command(&[multi_engine_v38_json(), multi_engine_v38_epoch_json()]);

    // Check for inconsistent/noisy measurements (CV > 5%)
    let inconsistent_count = count_css_elements(&html_content, ".inconsistent");
    assert!(
        inconsistent_count > 0,
        "Should have at least one inconsistent measurement marked"
    );

    // Verify the inconsistent marking includes CV percentage
    assert!(html_content.contains("CV:") && html_content.contains("%"));
}

#[test]
fn report_baseline_selection_override() {
    let input_files = &[multi_engine_v38_json(), multi_engine_v38_epoch_json()];

    // First report with default baseline (first engine in data)
    let html_content1 = run_report_command(input_files);

    // Second report with explicit baseline override
    let html_content2 = run_report_with_baseline(
        "engines/wasmtime/wasmtime-v38/libengine.dylib [-W epoch-interruption=y]",
        input_files,
    );

    // Default baseline should be the regular engine (full name)
    assert!(html_content1.contains("engines/wasmtime/wasmtime-v38/libengine.dylib (baseline)"));
    assert!(!html_content1.contains("[-W epoch-interruption=y] (baseline)"));

    // Override baseline should be the epoch engine
    assert!(html_content2.contains(
        "engines/wasmtime/wasmtime-v38/libengine.dylib [-W epoch-interruption=y] (baseline)"
    ));

    // The statistical results should be inverted - what was "slower" should now be "faster"
    // Since we're changing which engine is the baseline
    assert_ne!(
        html_content1, html_content2,
        "Reports with different baselines should produce different results"
    );
}

#[test]
fn report_different_significance_levels() {
    let input_files = &[multi_engine_v38_json(), multi_engine_v38_epoch_json()];

    // Strict significance level (0.01 = 99% confidence)
    let html_content_strict = run_report_with_significance("0.01", input_files);

    // Lenient significance level (0.10 = 90% confidence)
    let html_content_lenient = run_report_with_significance("0.10", input_files);

    // Verify different confidence intervals are reported
    assert!(html_content_strict.contains("99.00%"));
    assert!(html_content_lenient.contains("90.00%"));

    // Count significant results (slower/faster classes) - lenient should have more
    let significant_strict = count_css_elements(&html_content_strict, ".slower")
        + count_css_elements(&html_content_strict, ".faster");
    let significant_lenient = count_css_elements(&html_content_lenient, ".slower")
        + count_css_elements(&html_content_lenient, ".faster");

    // Lenient significance level should find same or more significant results
    assert!(significant_lenient >= significant_strict,
        "Lenient significance level should find at least as many significant results. Strict: {}, Lenient: {}",
        significant_strict, significant_lenient);
}
