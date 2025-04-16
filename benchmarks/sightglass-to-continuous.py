#!/usr/bin/env python3
#
# Helper script that transforms JSON benchmark results
# from sightglass into the format that can be used by
# the https://github.com/marketplace/actions/continuous-benchmark
# Github action, custom JSON.

import argparse
from dataclasses import dataclass
import json
import os
import pathlib

# Sightglass Data format:
# [
#   {
#     "arch": "...",
#     "engine": "...",
#     "wasm": "path/to/benchmark.wasm",
#     "process": "...",
#     "iteration": 0,
#     "phase": "...",
#     "event": "nanoseconds/cycles/...",
#     "count: "..."
#   },
#   ...
# ]
#
# Continuous Benchmark Custom (smaller is better) JSON Format:
# [
#   {
#     "name": "...",
#     "unit": "...",
#     "value": "...",
#     // optionally, range/extra for tooltips
#   }
# ]
#
# With the transform, we don't aim for a full translation of data.
# Instead, we create data points for compilation cycles with the
# value being the mean cycle time.  In the extras (tooltip), we
# provide other metadata including:
# - Engine Git Hash
# - Max
# - Min
# - Time measure

@dataclass
class ReducedSample:
    name: str
    arch: str
    event: str
    engine: str
    count: int

    def series_name(self):
        return f"{self.arch} - {self.name}"

def wasm_path_to_benchmark_name(wasm_path: str) -> str:
    splits = wasm_path.split("/")
    if splits[-1] == "benchmark.wasm":
        # E.g. noop/benchmark.wasm -> noop
        return splits[-2]
    else:
        # E.g. libsodium/libsodium-box7.wasm -> libsodium-box7
        return splits[-1].replace(".wasm", "")


def transform(sightglass_data: [dict]) -> [dict]:
    series_map = {}
    for sample in sightglass_data:
        phase = sample["phase"]
        if phase != "Execution":
            continue

        s = ReducedSample(
            arch=sample["arch"],
            engine=sample["engine"],
            name=wasm_path_to_benchmark_name(sample["wasm"]),
            event=sample["event"],
            count=int(sample["count"])
        )
        series_map.setdefault(s.series_name(), []).append(s)

    output = []
    for series_name, samples in series_map.items():
        cycle_samples = [s.count for s in samples if s.event == "cycles"]

        # yes, this could be done more efficiently
        count_cycles = len(cycle_samples)
        min_cycles = min(cycle_samples)
        max_cycles = max(cycle_samples)
        sum_cycles = sum(cycle_samples)
        avg_cycles = sum_cycles / count_cycles

        # there might be a better stastical measure here as this doesn't
        # factor in confidence levels or anything, but is intended to give
        # a picture of whether there are any significant outlier samples.
        max_variance_from_mean = max(max_cycles - avg_cycles, avg_cycles - min_cycles)
        max_variance_as_pct = (max_variance_from_mean / avg_cycles) * 100

        output.append({
            "name": series_name,
            "unit": "cycles",
            "value": avg_cycles,
            "extra": f"Max: {max_cycles}, Min: {min_cycles}, ±{max_variance_as_pct:.2f}%"
        })

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_in", type=pathlib.Path, help="Sightglass JSON input path")
    parser.add_argument("json_out", type=pathlib.Path, help="Custom JSON output path")
    args = parser.parse_args()
    with open(args.json_in) as json_in:
        input_data = json.load(json_in)
    transformed = transform(input_data)
    with open(args.json_out, "w") as json_out:
        json.dump(transformed, json_out)

if __name__ == '__main__':
    main()
