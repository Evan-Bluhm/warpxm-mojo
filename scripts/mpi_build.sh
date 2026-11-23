#!/usr/bin/env bash
# Build a driver for multi-rank (MPI) runs.
#
#   ./scripts/mpi_build.sh examples/euler_sod_2d_vtu.mojo   # -> build/mpi/euler_sod_2d_vtu
#   ./scripts/mpi_build.sh -o my_binary path/to/driver.mojo
#   mpirun -np 4 build/mpi/euler_sod_2d_vtu
#
# This is `mojo build` with `-D WXM_MPI=1` (which switches
# `warpxm/parallel/mpi.mojo` from its serial stand-ins to MPI) plus the C
# shim and libmpi on the link line. The MPI compiler wrapper is taken from
# $MPICC (default `mpicc`); the mojo command from $MOJO (default `mojo`; it
# may be several words, e.g. `apptainer exec mojo.sif mojo`). Extra
# arguments after the driver are passed to `mojo build`, after those in
# $MOJO_BUILD_FLAGS (e.g. "--target-accelerator sm_90" to build on a node
# without a GPU).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MPICC="${MPICC:-mpicc}"
MOJO="${MOJO:-mojo}"

out=""
if [[ "${1:-}" == "-o" ]]; then
    out="$2"
    shift 2
fi
if (( $# < 1 )); then
    echo "usage: $(basename "$0") [-o output] driver.mojo [mojo build args...]" >&2
    exit 2
fi
driver="$1"
shift
if [[ -z "$out" ]]; then
    out="$ROOT/build/mpi/$(basename "$driver" .mojo)"
fi
mkdir -p "$(dirname "$out")" "$ROOT/build"

if ! command -v "$MPICC" >/dev/null 2>&1; then
    echo "error: $MPICC not found; install an MPI or set MPICC" >&2
    exit 1
fi

# The shim is rebuilt only when its source (or the compiler) changes.
shim_src="$ROOT/src/warpxm/parallel/mpi_shim.c"
shim_obj="$ROOT/build/mpi_shim.o"
if [[ ! -f "$shim_obj" || "$shim_src" -nt "$shim_obj" ]]; then
    "$MPICC" -O2 -fPIC -c "$shim_src" -o "$shim_obj.$$"
    mv "$shim_obj.$$" "$shim_obj"
fi

# Link flags of the MPI library: Open MPI spells the query `--showme:link`,
# MPICH and its derivatives `-link_info` (which also prints the compiler).
if link_flags="$("$MPICC" --showme:link 2>/dev/null)"; then
    :
else
    link_flags="$("$MPICC" -link_info | cut -d' ' -f2-)"
fi
linker_args=(-Xlinker "$shim_obj")
for flag in $link_flags; do
    case "$flag" in
        -L*|-l*) linker_args+=(-Xlinker "$flag") ;;
        -Wl,-rpath,*) linker_args+=(-Xlinker -rpath -Xlinker "${flag#-Wl,-rpath,}") ;;
        *) ;;  # compiler-driver flags (-pthread, -Wl,--enable-new-dtags, ...)
    esac
done

# $MOJO is deliberately unquoted: it may carry a container prefix.
exec $MOJO build ${MOJO_BUILD_FLAGS:-} -I "$ROOT/src" -D WXM_MPI=1 "${linker_args[@]}" \
    -o "$out" "$driver" "$@"
