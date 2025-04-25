#!/usr/bin/env sh

set -e

ENGINE=${ENGINE:-"engines/wasmtime/libengine.dylib"}

run_benchmark() {
    prefix=$1
    pass=$2
    shift
    shift
    cargo run -- benchmark --benchmark-phase=execution \
        --engine "${ENGINE}" \
        --processes=1 \
        --iterations-per-process=12 \
        --raw -o "$prefix-$pass.json" \
        -m cycles \
        "$@" \
        -- benchmarks/all.suite
    # ./benchmarks/sightglass-to-continuous.py result-$pass.json -o agg-$pass.json
}

prefix=$1
shift
iterations=4
for i in `seq 0 ${iterations}`; do
    echo "Iteration ${i}/${iterations}"
    run_benchmark $prefix $i "$@"
done
