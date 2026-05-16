#!/usr/bin/env bash
# Run the full warpxm-mojo regression suite.
#
# Iterates over every `tests/*.mojo` file, runs it with `mojo run -I src`,
# and reports pass / fail. Robust against test output that contains a
# trailing newline or extra lines: a test counts as passed iff its
# output contains the literal line `PASSED`.
#
# Usage:
#   pixi run test                # via pixi task
#   ./scripts/run_tests.sh       # directly
#   ./scripts/run_tests.sh -k    # stop at first failure (-k = no, default)
#   ./scripts/run_tests.sh -v    # verbose: print the full output of each test
#
# Exit code: 0 if every test passed, 1 otherwise.

set -u

STOP_ON_FAIL=0
VERBOSE=0
usage() {
    cat <<EOF
Usage: $(basename "$0") [-k] [-v] [-h]
  -k  stop at first failure
  -v  verbose — print the full output of any failing test
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

# Run from the repo root regardless of where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if ! command -v mojo >/dev/null 2>&1; then
    echo "error: mojo not in PATH. Run under \`pixi run\` or activate the env." >&2
    exit 1
fi

shopt -s nullglob
tests=(tests/*.mojo)
total=${#tests[@]}
if [[ $total -eq 0 ]]; then
    echo "error: no tests/*.mojo files found" >&2
    exit 1
fi

passed=0
failed_names=()

printf "Running %d regression tests...\n\n" "$total"

for t in "${tests[@]}"; do
    name="$(basename "$t" .mojo)"
    printf "  %-50s " "$name"
    output=$(mojo run -I src "$t" 2>&1)
    if echo "$output" | grep -q "^PASSED$"; then
        printf "PASSED\n"
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
printf "Passed: %d / %d\n" "$passed" "$total"

if [[ ${#failed_names[@]} -gt 0 ]]; then
    printf "Failed:\n"
    for name in "${failed_names[@]}"; do
        printf "  - %s\n" "$name"
    done
    exit 1
fi
exit 0
