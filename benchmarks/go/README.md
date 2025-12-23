# Go/TinyGo Benchmarks for Sightglass

This directory contains JSON serialization and regex matching benchmarks that can be compiled with both TinyGo and standard Go.

## Benchmarks

- **json**: JSON serialization/deserialization using the standard library
- **regex**: Regular expression matching using the standard library

## Status

✅ **TinyGo benchmarks work perfectly** and can be used for performance analysis.

⚠️ **Standard Go benchmarks currently fail** due to a WASI implementation incompatibility. See [WASI-ISSUE.md](./WASI-ISSUE.md) for detailed analysis.

## Building

```bash
# Build all variants (TinyGo and Go)
cd benchmarks
./build.sh go/

# This produces:
# - tinygo-json.wasm, tinygo-regex.wasm (working)
# - go-json.wasm, go-regex.wasm (not working with current wasmtime-bench-api)
```

## Running Benchmarks

```bash
# Run TinyGo benchmarks
cargo run --release -- benchmark \
  -e ../wasmtime-v39.0.1.dylib \
  --processes 10 --iterations-per-process 10 \
  -- benchmarks/go/tinygo-json.wasm benchmarks/go/tinygo-regex.wasm

# Standard Go benchmarks will fail with file I/O errors
```

## Implementation Details

### Unified Source Code

Both TinyGo and standard Go compile from the same source files (`json/main.go`, `regex/main.go`). The code uses `//go:wasmimport` directives which are supported by both compilers.

### Benchmark Lifecycle

1. Read input file (`json.input` or `regex.input`)
2. Call `benchStart()` imported from sightglass
3. Perform benchmark work (JSON ops or regex matching)
4. Call `benchEnd()` imported from sightglass
5. Print results to stderr

### File Organization

- Input files are hardlinked with multiple names for compatibility:
  - `json.input` → `tinygo-json.input`, `go-json.input`
  - `regex.input` → `tinygo-regex.input`, `go-regex.input`

- Expected output files:
  - `tinygo-json.stderr.expected`, `go-json.stderr.expected`
  - `tinygo-regex.stderr.expected`, `go-regex.stderr.expected`

## Performance Results (TinyGo)

Example results from 3 iterations:

### JSON Benchmark
- **Compilation**: 59.1M cycles (avg)
- **Instantiation**: 144k cycles (avg)
- **Execution**: 51.2M cycles (avg)

### Regex Benchmark
- **Compilation**: 37.2M cycles (avg)
- **Instantiation**: 37.5k cycles (avg)  
- **Execution**: 2.13B cycles (avg)

## Future Work

To enable standard Go benchmarks:
1. Update wasmtime-bench-api to map working directory as `/` instead of `.`
2. Or wait for Go to fix WASI preview 1 preopen discovery
3. Or use WASI preview 2 when available

## References

- [WASI-ISSUE.md](./WASI-ISSUE.md) - Detailed analysis of Go WASI bug
- [Go Issue #60732](https://github.com/golang/go/issues/60732) - Upstream bug report
- [Wasmtime bench-api](https://github.com/bytecodealliance/wasmtime/blob/main/crates/bench-api/src/lib.rs)
