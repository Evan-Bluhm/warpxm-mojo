# `warpxm.parallel`

Multi-rank support: the MPI binding, the domain decomposition and the
halo exchange. The overview and the driver rules are on
[Running on several GPUs](../concepts/parallel.md).

## `mpi`

```mojo
from warpxm.parallel import mpi
```

One compile-time switch decides what this module is. In a build with
`-D WXM_MPI=1` (`pixi run mpi-build`) every function calls MPI through the
C shim `parallel/mpi_shim.c`. Otherwise each returns the single-rank
answer and compiles to a constant. MPI is initialised on first use and
finalised at exit; a rank that exits with an error aborts the job.

| Function | Meaning |
| -------- | ------- |
| `mpi.USE_MPI` | comptime `Bool`: is this an MPI build |
| `rank()`, `size()` | this rank and the number of ranks (`0`, `1` in a serial build) |
| `num_ranks()`, `is_root()` | `size()`; `rank() == 0` |
| `root_print(...)` | `print` on rank 0; arguments evaluated on every rank |
| `rank_device_context()` | `DeviceContext` for this rank's GPU |
| `global_sum(x)`, `global_min(x)`, `global_max(x)` | reduce a `Real` over all ranks (`global_sum` also takes `UInt64`) |
| `gpu_aware()` | does MPI accept device pointers (`WXM_GPU_AWARE_MPI` overrides) |
| `barrier()`, `wtime()` | `MPI_Barrier`, `MPI_Wtime` |

The prelude re-exports `is_root`, `num_ranks`, `rank_device_context`,
`root_print` and the three reductions.

Reductions are collective: every rank must make the call. The same holds
for the mesh and solver methods built on them (`mesh.integrate`,
`integrate_fn`, `l2_error`, `checksum`, `solver.suggest_dt`).

## `partition`

```mojo
from warpxm.parallel.partition import Partition, decompose, process_grid
```

`process_grid(n, ndim, size)` picks the grid of boxes: the factorisation
of `size` with the least interface area that fits the element counts `n`.
`decompose(n, periodic, ndim, rank, size)` returns one rank's
`Partition`:

| Member | Meaning |
| ------ | ------- |
| `global_n`, `procs` | elements and boxes along each axis |
| `coords`, `start`, `n` | this rank's box, its first global element and its element counts |
| `neighbor[2·axis + side]` | rank across each side, `-1` for none |
| `global_element(e)` | index in the whole mesh of local element `e` |

A side has no neighbour on a physical boundary, and on an axis with a
single box (a periodic axis then wraps onto the rank itself, as on a
serial mesh). Meshes call `decompose` themselves; `mesh.partition` holds
the result.

## `halo`

```mojo
from warpxm.parallel.halo import HaloPlan, HaloBuffers, exchange_halo
```

| Type | Owner | Holds |
| ---- | ----- | ----- |
| `HaloPlan` | the mesh (`mesh.halo`) | neighbour ranks, ghost counts per direction, the list of elements to send |
| `HaloBuffers` | each exchanged field (`q.halo`, the limiter) | device send / receive buffers and, without GPU-aware MPI, pinned host copies |

```mojo
exchange_halo[NPE, P1, NC, TRACE](ctx, plan, buf, buf_len, halo)
```

refreshes the ghost elements of the field in `buf`: `NPE` nodes per
element with `P1` nodes along each configuration axis, `NC` values per
node. With `TRACE = True` only the nodes on the shared face travel (what
the DG surface term reads); with `TRACE = False` the field holds `NC`
values per *element*, such as the limiter's cell means. It compiles to
nothing in a serial build and returns at once on a mesh without ghosts.

Drivers do not call it. `launch_dg` does, through

```mojo
from warpxm.spatial.dg_launch import exchange_ghosts
exchange_ghosts(ctx, mesh, q)
```

which is also what a hand-written kernel that reads neighbour elements
needs before it is launched. `MoeRossmanith.apply` and
`DGVlasovSolver.compute_rhs` make their own exchanges.

### Ghost numbering

Directions are `d = 2·axis + side`. The ghosts of direction `d` are one
contiguous block after the owned elements, blocks in increasing `d`,
starting at `num_elements + mesh.halo.ghost_offset(d)`. Within a block
they follow the layer of elements on that face in the mesh's element
order with the face's axis left out. The sender lists its elements in the
same order, so no indices travel with the data.
