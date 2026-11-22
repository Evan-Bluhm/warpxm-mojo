#!/usr/bin/env bash
# Run the warpxm-mojo regression suite.
#
# Every `tests/*.mojo` file is a standalone program. A test passes iff it
# exits with status 0 AND prints the literal line `PASSED`. Tests are
# independent of each other, so they are compiled and run in parallel.
#
# Usage:
#   pixi run test                    # whole suite, via the pixi task
#   ./scripts/run_tests.sh           # whole suite, directly
#   ./scripts/run_tests.sh euler 3d  # only tests whose name contains
#                                    # "euler" or "3d" (substring match)
#   ./scripts/run_tests.sh -j 4      # at most 4 tests at once (default 16,
#                                    # capped at the number of CPU cores)
#   ./scripts/run_tests.sh -j 1      # strictly sequential
#   ./scripts/run_tests.sh -k        # stop launching tests after a failure
#   ./scripts/run_tests.sh -v        # print the full output of failing tests
#   ./scripts/run_tests.sh -t        # also list the ten slowest tests
#
# Exit code: 0 if every selected test passed, 1 otherwise.

set -u

JOBS=16
STOP_ON_FAIL=0
VERBOSE=0
SHOW_SLOWEST=0
usage() {
    cat <<EOF
Usage: $(basename "$0") [-j N] [-k] [-v] [-t] [-h] [name-substring ...]
  -j N  run at most N tests at once (default 16, capped at CPU count)
  -k    stop launching tests after the first failure
  -v    verbose — print the full output of any failing test
  -t    list the ten slowest tests at the end
  -h    this help
EOF
}
while getopts "j:kvth" opt; do
    case "$opt" in
        j) JOBS="$OPTARG" ;;
        k) STOP_ON_FAIL=1 ;;
        v) VERBOSE=1 ;;
        t) SHOW_SLOWEST=1 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: -j expects a positive integer, got '$JOBS'" >&2
    exit 2
fi
# `nproc` alone honours OMP_NUM_THREADS, which is often 1; ask for all cores.
CORES=$(nproc --all 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)
if (( JOBS > CORES )); then JOBS=$CORES; fi

# Run from the repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if ! command -v mojo >/dev/null 2>&1; then
    echo "error: mojo not in PATH. Run under \`pixi run\` or activate the env." >&2
    exit 1
fi

shopt -s nullglob
tests=()
for t in tests/*.mojo; do
    name="$(basename "$t" .mojo)"
    if (( $# == 0 )); then
        tests+=("$name")
    else
        for pat in "$@"; do
            if [[ "$name" == *"$pat"* ]]; then tests+=("$name"); break; fi
        done
    fi
done
total=${#tests[@]}
if (( total == 0 )); then
    echo "error: no tests/*.mojo files matched" >&2
    exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/warpxm-tests.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# One test: write <name>.log, <name>.time (seconds) and <name>.status.
run_one() {
    local name="$1" start end rc
    if [[ -e "$WORK/stop" ]]; then
        echo skipped > "$WORK/$name.status"
        return 0
    fi
    start=$(date +%s.%N)
    mojo run -I src "tests/$name.mojo" > "$WORK/$name.log" 2>&1
    rc=$?
    end=$(date +%s.%N)
    awk -v a="$start" -v b="$end" 'BEGIN { printf "%.1f\n", b - a }' \
        > "$WORK/$name.time"
    if (( rc == 0 )) && grep -q "^PASSED$" "$WORK/$name.log"; then
        echo passed > "$WORK/$name.status"
        printf "  %-50s PASSED  %6ss\n" "$name" "$(cat "$WORK/$name.time")"
    else
        echo failed > "$WORK/$name.status"
        printf "  %-50s FAILED  %6ss\n" "$name" "$(cat "$WORK/$name.time")"
        if (( STOP_ON_FAIL == 1 )); then : > "$WORK/stop"; fi
    fi
}
export -f run_one
export WORK STOP_ON_FAIL

printf "Running %d regression tests, %d at a time...\n\n" "$total" "$JOBS"
suite_start=$(date +%s.%N)
printf '%s\n' "${tests[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$1"' _ {}
suite_end=$(date +%s.%N)

passed=0
failed_names=()
skipped=0
for name in "${tests[@]}"; do
    case "$(cat "$WORK/$name.status" 2>/dev/null || echo failed)" in
        passed) passed=$((passed + 1)) ;;
        skipped) skipped=$((skipped + 1)) ;;
        *) failed_names+=("$name") ;;
    esac
done

if (( VERBOSE == 1 )); then
    for name in "${failed_names[@]}"; do
        printf "\n--- %s ---\n" "$name"
        sed 's/^/    | /' "$WORK/$name.log" 2>/dev/null
    done
fi

if (( SHOW_SLOWEST == 1 )); then
    printf "\nSlowest tests:\n"
    for name in "${tests[@]}"; do
        [[ -f "$WORK/$name.time" ]] && printf "%s %s\n" "$(cat "$WORK/$name.time")" "$name"
    done | sort -rn | head -10 | awk '{ printf "  %6ss  %s\n", $1, $2 }'
fi

printf "\nPassed: %d / %d in %s s\n" "$passed" "$total" \
    "$(awk -v a="$suite_start" -v b="$suite_end" 'BEGIN { printf "%.0f", b - a }')"
if (( skipped > 0 )); then
    printf "Skipped after first failure: %d\n" "$skipped"
fi

if (( ${#failed_names[@]} > 0 )); then
    printf "Failed:\n"
    for name in "${failed_names[@]}"; do
        printf "  - %s\n" "$name"
    done
    exit 1
fi
if (( skipped > 0 )); then exit 1; fi
exit 0
