# MPI for multi-rank runs, behind one compile-time switch.
#
# A build with `-D WXM_MPI=1` links `mpi_shim.c` and libmpi (use
# `scripts/mpi_build.sh`); every function here then calls MPI. Without the
# define, each function is the single-rank answer (rank 0 of 1, reductions
# return their argument) and compiles to a constant: a serial build has no
# MPI dependency and runs exactly the code it ran before MPI support
# existed.
#
# Nothing outside `warpxm/parallel/` calls MPI directly. A driver uses
# `rank_device_context()` to pick its GPU, `root_print(...)` to print from
# one rank, and `global_sum` / `global_min` / `global_max` to combine a
# quantity it accumulated itself over its own elements. MPI is initialised on first use
# and finalised at exit by the shim.

from std.ffi import external_call, c_int
from std.sys.defines import is_defined
from max.gpu.host import DeviceContext

from warpxm.core.types import Real


comptime USE_MPI: Bool = is_defined["WXM_MPI"]()
"""True in a build with `-D WXM_MPI=1`."""


def rank() -> Int:
    """This process's rank, 0 in a serial build."""
    comptime if USE_MPI:
        return Int(external_call["wxm_mpi_rank", c_int]())
    else:
        return 0


def size() -> Int:
    """Number of ranks, 1 in a serial build."""
    comptime if USE_MPI:
        return Int(external_call["wxm_mpi_size", c_int]())
    else:
        return 1


def num_ranks() -> Int:
    """Number of ranks sharing the run (`size()` under a name that reads
    well in a driver)."""
    return size()


def is_root() -> Bool:
    """True on rank 0. Guard driver `print`s with it."""
    return rank() == 0


def root_print[*Ts: Writable](*values: *Ts):
    """`print` on rank 0 only. The arguments are evaluated on every rank,
    which is what a diagnostic such as `mesh.integrate(q, c)` or
    `spatial.suggest_dt(...)` needs: those reduce over all ranks, so every
    rank has to make the call. (Inside `if is_root():` they would be made
    by rank 0 alone, and it would wait for the others for ever.)"""
    if is_root():
        print(*values)


def gpu_aware() -> Bool:
    """True when MPI accepts device pointers (CUDA- or ROCm-aware build,
    or `WXM_GPU_AWARE_MPI=1`); the halo exchange then skips its host
    staging copies."""
    comptime if USE_MPI:
        return Int(external_call["wxm_mpi_gpu_aware", c_int]()) != 0
    else:
        return False


def rank_device_context() raises -> DeviceContext:
    """The device this rank should use: GPU `k mod (number of visible
    GPUs)` for the k-th rank on a node. With one GPU visible per rank
    (`CUDA_VISIBLE_DEVICES` set by the launcher, or Slurm's
    `--gpus-per-task=1`) and in a serial build this is device 0."""
    comptime if USE_MPI:
        var node_rank = Int(external_call["wxm_mpi_node_rank", c_int]())
        return DeviceContext(
            device_id=node_rank % DeviceContext.number_of_devices()
        )
    else:
        return DeviceContext()


def wtime() -> Float64:
    """Wall-clock seconds (`MPI_Wtime`); 0 in a serial build."""
    comptime if USE_MPI:
        return external_call["wxm_mpi_wtime", Float64]()
    else:
        return 0.0


def barrier():
    """Block until every rank has reached this call."""
    comptime if USE_MPI:
        external_call["wxm_mpi_barrier", NoneType]()


def global_sum(x: Real) -> Real:
    """Sum of `x` over all ranks."""
    comptime if USE_MPI:
        return external_call["wxm_mpi_sum", Float64](x)
    else:
        return x


def global_min(x: Real) -> Real:
    """Minimum of `x` over all ranks."""
    comptime if USE_MPI:
        return external_call["wxm_mpi_min", Float64](x)
    else:
        return x


def global_max(x: Real) -> Real:
    """Maximum of `x` over all ranks."""
    comptime if USE_MPI:
        return external_call["wxm_mpi_max", Float64](x)
    else:
        return x


def global_sum(x: UInt64) -> UInt64:
    """Wrapping sum of `x` over all ranks (for order-independent
    checksums)."""
    comptime if USE_MPI:
        return external_call["wxm_mpi_sum_u64", UInt64](x)
    else:
        return x
