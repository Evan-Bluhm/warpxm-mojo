# Source this on UW Tillicum (Rocky 9, H200) before building or running:
#
#     source scripts/tillicum-env.sh
#
# Tillicum's glibc (2.34) is new enough for Mojo, so unlike Klone no
# container is needed: pixi and the pinned toolchain run natively. pixi
# and its package cache live in group storage, not in the small home
# quota. Login nodes have no GPU, so builds name the target (H200 =
# Hopper, sm_90) through MOJO_BUILD_FLAGS, which scripts/run_tests.sh -b,
# scripts/mpi_build.sh and scripts/run_mpi_tests.sh -b honour.

: "${PIXI_HOME:=/gpfs/projects/shumlak/evan/pixi}"
export PIXI_HOME
export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$PIXI_HOME/cache}"
export RATTLER_CACHE_DIR="$PIXI_CACHE_DIR"
case ":$PATH:" in
    *":$PIXI_HOME/bin:"*) ;;
    *) export PATH="$PIXI_HOME/bin:$PATH" ;;
esac

# The default Open MPI is built with CUDA support (smcuda BTL), which the
# halo exchange uses to hand device pointers straight to MPI.
module load gcc/13.4.0 cuda/12.9.1 openmpi/5.0.8 >/dev/null 2>&1

export MOJO_BUILD_FLAGS="${MOJO_BUILD_FLAGS:---target-accelerator sm_90}"
