#!/usr/bin/env bash
# Run the multi-rank regression suite (`tests/mpi/*.mojo`).
#
# Each test is built once with `scripts/mpi_build.sh` and run under
# `mpirun` at several rank counts. It passes iff, at every rank count, it
# exits 0 and prints `PASSED`, and the `CHECKSUM` lines it prints (hashes
# of the final fields, `mesh.checksum(q)`) are the same at every rank
# count: the decomposition must not change the answer by a single bit.
#
# The rank counts come from a `# MPI_RANKS: 1 2 4` line in the test, or
# default to `1 2 3 4`. All ranks share whatever GPUs are visible, so this
# runs on a one-GPU workstation.
#
# Usage:
#   pixi run test-mpi               # whole suite
#   pixi run test-mpi euler 3d      # tests whose name contains a pattern
#   ./scripts/run_mpi_tests.sh -v   # print the output of failing runs
#   ./scripts/run_mpi_tests.sh -j 2 # build at most 2 tests at once (default 4)
#
# Environment: MPIRUN (default `mpirun`), MPIRUN_FLAGS (default
# `--oversubscribe` when the launcher accepts it).

set -u

JOBS=4
VERBOSE=0
while getopts "j:vh" opt; do
    case "$opt" in
        j) JOBS="$OPTARG" ;;
        v) VERBOSE=1 ;;
        h) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) exit 2 ;;
    esac
done
shift $((OPTIND - 1))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MPIRUN="${MPIRUN:-mpirun}"
for tool in mojo "$MPIRUN" "${MPICC:-mpicc}"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "error: $tool not found (run under \`pixi run\`; MPI comes from the system)" >&2
        exit 1
    fi
done
if [[ -z "${MPIRUN_FLAGS+x}" ]]; then
    MPIRUN_FLAGS=""
    if "$MPIRUN" --oversubscribe -np 1 true >/dev/null 2>&1; then
        MPIRUN_FLAGS="--oversubscribe"
    fi
fi

shopt -s nullglob
tests=()
for t in tests/mpi/*.mojo; do
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
    echo "error: no tests/mpi/*.mojo files matched" >&2
    exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/warpxm-mpi-tests.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

# Build the shim once, up front, so that parallel builds do not race on it.
mkdir -p build
"${MPICC:-mpicc}" -O2 -fPIC -c src/warpxm/parallel/mpi_shim.c -o build/mpi_shim.o || exit 1

build_one() {
    local name="$1"
    if ./scripts/mpi_build.sh -o "$WORK/$name" "tests/mpi/$name.mojo" \
        > "$WORK/$name.build.log" 2>&1; then
        echo ok > "$WORK/$name.build"
    else
        echo failed > "$WORK/$name.build"
    fi
}
export -f build_one
export WORK

printf "Building %d MPI tests...\n" "$total"
printf '%s\n' "${tests[@]}" | xargs -P "$JOBS" -I{} bash -c 'build_one "$1"' _ {}

passed=0
failed_names=()
printf "\n"
for name in "${tests[@]}"; do
    if [[ "$(cat "$WORK/$name.build")" != ok ]]; then
        printf "  %-44s BUILD FAILED\n" "$name"
        failed_names+=("$name")
        (( VERBOSE == 1 )) && sed 's/^/    | /' "$WORK/$name.build.log"
        continue
    fi
    ranks="$(sed -n 's/^# MPI_RANKS:[[:space:]]*//p' "tests/mpi/$name.mojo" | head -1)"
    ranks="${ranks:-1 2 3 4}"
    ok=1
    reference=""
    detail=""
    for np in $ranks; do
        log="$WORK/$name.np$np.log"
        # Each test runs in its own directory so that output files of
        # different rank counts cannot be confused.
        mkdir -p "$WORK/$name.np$np.run"
        (cd "$WORK/$name.np$np.run" && \
            $MPIRUN $MPIRUN_FLAGS -np "$np" "$WORK/$name" > "$log" 2>&1)
        rc=$?
        sums="$(grep -a '^CHECKSUM' "$log")"
        if (( rc != 0 )) || ! grep -aq '^PASSED$' "$log"; then
            ok=0; detail="np=$np failed"
        elif [[ -z "$reference" ]]; then
            reference="$sums"
        elif [[ "$sums" != "$reference" ]]; then
            ok=0; detail="np=$np differs from np=${ranks%% *}"
        fi
        if (( ok == 0 )); then
            (( VERBOSE == 1 )) && sed 's/^/    | /' "$log"
            break
        fi
    done
    if (( ok == 1 )); then
        printf "  %-44s PASSED  (np = %s)\n" "$name" "$ranks"
        passed=$((passed + 1))
    else
        printf "  %-44s FAILED  (%s)\n" "$name" "$detail"
        failed_names+=("$name")
    fi
done

printf "\nPassed: %d / %d\n" "$passed" "$total"
if (( ${#failed_names[@]} > 0 )); then
    printf "Failed:\n"
    for name in "${failed_names[@]}"; do printf "  - %s\n" "$name"; done
    exit 1
fi
exit 0
