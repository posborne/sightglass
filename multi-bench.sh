#!/usr/bin/env sh

set -e

ENGINE=${ENGINE:-"engines/wasmtime/libengine.dylib"}
ITERATIONS=${ITERATIONS:-"12"}
PASSES=${PASSES:-"4"}
BENCHMARK=${BENCHMARK:-"benchmarks/all.suite"}

run_benchmark() {
    prefix=$1
    pass=$2
    shift
    shift
    cargo run -- benchmark --benchmark-phase=execution \
        --engine "${ENGINE}" \
        --processes=1 \
        --iterations-per-process=${ITERATIONS} \
        --raw -o "$prefix-$pass.json" \
        -m cycles \
        "$@" \
        -- "${BENCHMARK}"
    # ./benchmarks/sightglass-to-continuous.py result-$pass.json -o agg-$pass.json
}

prefix=$1
shift
for i in `seq 1 ${PASSES}`; do
    echo "Pass ${i}/${PASSES}"
    run_benchmark $prefix $i "$@"
done
