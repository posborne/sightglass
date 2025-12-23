#!/bin/sh
set -e

# Configuration
# Set GO_COMPILER to "go" to use standard Go, or "tinygo" for TinyGo
GO_COMPILER="${GO_COMPILER:-tinygo}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Compiler-specific settings
if [ "$GO_COMPILER" = "go" ]; then
    COMPILER_CMD="go"
    BUILD_ARGS="build -o"
    WASI_ENV="GOOS=wasip1 GOARCH=wasm"
elif [ "$GO_COMPILER" = "tinygo" ]; then
    COMPILER_CMD="tinygo"
    BUILD_ARGS="build -o"
    WASI_ENV=""
    TINYGO_FLAGS="-target=wasi -opt=2 -gc=leaking"
else
    echo "Error: GO_COMPILER must be 'go' or 'tinygo'"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

cd "$SCRIPT_DIR"

# Auto-discover benchmarks: any directory with a main.go
for dir in */; do
    # Remove trailing slash
    benchmark=$(basename "$dir")

    # Skip if not a benchmark directory (no main.go)
    [ ! -f "$dir/main.go" ] && continue

    echo "Building $benchmark benchmark with $GO_COMPILER..."

    cd "$dir"

    # Build with selected compiler
    if [ "$GO_COMPILER" = "go" ]; then
        # Standard Go uses GOOS=wasip1 and doesn't need tinygo build tag
        env $WASI_ENV "$COMPILER_CMD" $BUILD_ARGS "$OUTPUT_DIR/tinygo-$benchmark.wasm" .
    else
        # TinyGo needs the tinygo build tag to select the right bench implementation
        "$COMPILER_CMD" $BUILD_ARGS "$OUTPUT_DIR/tinygo-$benchmark.wasm" $TINYGO_FLAGS -tags=tinygo .
    fi

    cd "$SCRIPT_DIR"

    echo "> Built tinygo-$benchmark.wasm"
done

echo "All benchmarks built successfully with $GO_COMPILER"
