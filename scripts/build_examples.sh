#!/usr/bin/env bash
# Type-check every example by running `mojo build -o /dev/null` against
# each `examples/*.mojo` driver. This catches compile errors and stale
# import paths without actually running the driver (which would take
# minutes per example for the long-time KH / OT runs).
#
# Usage:
#   pixi run build              # via pixi task
#   ./scripts/build_examples.sh
#   ./scripts/build_examples.sh -k    # stop at first failure
#   ./scripts/build_examples.sh -v    # verbose: show full mojo output on failure
#   ./scripts/build_examples.sh -h    # this help
#
# Exit code: 0 if every example built, 1 otherwise.

set -u

STOP_ON_FAIL=0
VERBOSE=0
usage() {
    cat <<EOF
Usage: $(basename "$0") [-k] [-v] [-h]
  -k  stop at first failure
  -v  verbose — print the full mojo output of any failing build
  -h  this help
EOF
}
while getopts "kvh" opt; do
    case "$opt" in
        k) STOP_ON_FAIL=1 ;;
        v) VERBOSE=1 ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if ! command -v mojo >/dev/null 2>&1; then
    echo "error: mojo not in PATH. Run under \`pixi run\` or activate the env." >&2
    exit 1
fi

shopt -s nullglob
examples=(examples/*.mojo)
total=${#examples[@]}
if [[ $total -eq 0 ]]; then
    echo "error: no examples/*.mojo files found" >&2
    exit 1
fi

passed=0
failed_names=()

# Mojo build writes the executable to the current dir; redirect to a temp
# location so we don't clutter the repo.
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

printf "Building %d examples...\n\n" "$total"

for e in "${examples[@]}"; do
    name="$(basename "$e" .mojo)"
    printf "  %-50s " "$name"
    # --emit object skips the link step — we only want to check that
    # the source compiles and type-checks, not produce an executable
    # (which would require libm linker flags for code using sin/cos/exp).
    output=$(mojo build -I src --emit object -o "$build_dir/$name.o" "$e" 2>&1)
    if [[ $? -eq 0 ]]; then
        printf "OK\n"
        passed=$((passed + 1))
    else
        printf "FAILED\n"
        failed_names+=("$name")
        if [[ $VERBOSE -eq 1 ]]; then
            echo "$output" | sed 's/^/    | /'
        fi
        if [[ $STOP_ON_FAIL -eq 1 ]]; then
            break
        fi
    fi
done

printf "\n"
printf "Built: %d / %d\n" "$passed" "$total"

if [[ ${#failed_names[@]} -gt 0 ]]; then
    printf "Failed:\n"
    for name in "${failed_names[@]}"; do
        printf "  - %s\n" "$name"
    done
    exit 1
fi
exit 0
