#!/usr/bin/env bash
# Run all bench/ drivers and aggregate the `BENCH config=...` lines into a
# single CSV at `bench/results.csv`. With `--profile`, also wrap each driver
# in `nsys profile` and stash the trace under `profiling/`.
#
# Usage:
#     scripts/bench.sh           # baseline timing only
#     scripts/bench.sh --profile # + nsys trace per driver
set -euo pipefail

cd "$(dirname "$0")/.."

CSV="bench/results.csv"
DO_PROFILE=0
if [[ "${1:-}" == "--profile" ]]; then
    DO_PROFILE=1
    mkdir -p profiling
fi

BENCHES=(
    bench_advection_2d
    bench_advection_3d
    bench_euler_2d
    bench_euler_3d
    bench_imhd_2d
    bench_imhd_3d
    bench_imhd_glm_2d
    bench_imhd_glm_3d
)

echo "config,steps,elapsed_s,steps_per_s" > "$CSV"

for name in "${BENCHES[@]}"; do
    src="bench/${name}.mojo"
    echo ">>> $name"
    if [[ "$DO_PROFILE" -eq 1 ]]; then
        nsys profile \
            --trace=cuda,nvtx \
            --output="profiling/${name}.nsys-rep" \
            --force-overwrite=true \
            pixi run mojo run -I src "$src" \
            | tee "/tmp/bench_${name}.out"
    else
        pixi run mojo run -I src "$src" | tee "/tmp/bench_${name}.out"
    fi

    line=$(grep '^BENCH ' "/tmp/bench_${name}.out" | tail -n 1)
    if [[ -z "$line" ]]; then
        echo "    !! no BENCH line emitted by $name" >&2
        continue
    fi
    config=$(echo "$line" | sed -n 's/.*config=\([^ ]*\).*/\1/p')
    steps=$(echo "$line" | sed -n 's/.*steps=\([^ ]*\).*/\1/p')
    elapsed_s=$(echo "$line" | sed -n 's/.*elapsed_s=\([^ ]*\).*/\1/p')
    steps_per_s=$(echo "$line" | sed -n 's/.*steps_per_s=\([^ ]*\).*/\1/p')
    printf '%s,%s,%s,%s\n' "$config" "$steps" "$elapsed_s" "$steps_per_s" >> "$CSV"
done

echo
echo "results -> $CSV"
column -t -s, "$CSV"
