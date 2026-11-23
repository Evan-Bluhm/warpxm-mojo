# Running on several GPUs

A problem too large for one device is split over several, one MPI rank
per GPU. This page covers how that works and what a driver has to do.
The API is on the [`parallel`](../api/parallel.md) page.

## Build and run

```fish
pixi run mpi-build examples/imhd_orszag_tang_2d_vtu.mojo   # -> build/mpi/<name>
mpirun -np 4 build/mpi/imhd_orszag_tang_2d_vtu
```

`mpi-build` is `mojo build` with `-D WXM_MPI=1` and the MPI library on the
link line. MPI itself comes from the system (`mpicc`, `mpirun`); it is not
a pixi dependency. Without the define, the same driver is the serial
program it always was: `pixi run example <name>` and `pixi run test` do
not need MPI installed, and every MPI call compiles to its single-rank
answer.

Each rank needs a GPU. With one GPU visible per rank (Slurm
`--gpus-per-task=1`, or `CUDA_VISIBLE_DEVICES` set per rank by the
launcher) nothing more is needed; when a node's GPUs are visible to all
its ranks, `rank_device_context()` gives rank `k` on the node GPU
`k mod (number of GPUs)`. Ranks that share a GPU work, slowly: that is
how the multi-rank tests run on a workstation.

On UW Klone, `scripts/klone-run` wraps the Slurm allocation, the
Apptainer image (`mojo.def`), the build and `mpirun`:

```bash
NP=4 scripts/klone-run imhd_orszag_tang_2d_vtu
```

## How the mesh is split

The mesh constructor takes the *global* mesh, as before, and keeps this
rank's box of it. `BlockMesh` cuts the element grid into a Cartesian grid
of boxes, choosing the factorisation of the rank count with the least
interface area. The decomposition is pure arithmetic on (rank, size), so
no communication is needed to agree on it.

Each rank stores its own elements followed by one layer of **ghost
elements**: copies of the neighbouring ranks' elements across each
partition face.

```text
state buffer:   | owned elements (num_elements) | ghosts (num_ghost) |
```

The mesh's face tables point at a ghost exactly as they point at any
other neighbour, so the DG kernel and the limiter need no knowledge of
the decomposition. On one rank there are no ghosts and the buffers are
the ones a serial build allocates.

Before every right-hand side, `launch_dg` refreshes the ghosts of its
input state: a pack kernel gathers the face nodes of the elements next to
each partition face, MPI exchanges them, and an unpack kernel writes them
into the ghosts. Only the face trace travels, `(P+1)^(NDIM-1)` of the
`(P+1)^NDIM` nodes of each element; per stage that is `6 / (N (P+1))` of
a 3D state with `N` elements along each side of a box. The limiter
likewise exchanges one cell mean per ghost element. DG and the limiter
couple face neighbours only, so a rank talks to at most `2·NDIM` others.

If MPI accepts device pointers (a CUDA- or ROCm-aware build), the device
buffers go straight to `MPI_Isend` / `MPI_Irecv`; otherwise they are
staged through pinned host memory. The choice is made at start-up from
the MPI library and can be forced with `WXM_GPU_AWARE_MPI=0` or `=1`.

The kinetic `PhaseMesh` is cut the same way along its configuration axes
(velocity space stays whole on each rank), in step with its companion
`BlockMesh`.

## What a driver has to do

Three things make a driver run on any number of ranks.

**Pass the ghost count** wherever the element count goes:

```mojo
var ctx = rank_device_context()
var mesh = BlockMesh[2, P](ctx, ...)                       # global extents
var q = State(ctx, mesh.num_elements, mesh.num_ghost)
var stepper = SSPRK3[2, P, EULER_NC](ctx, mesh.num_elements, mesh.num_ghost)
var limiter = MoeRossmanith[2, P, EULER_NC](ctx, mesh.num_elements, mesh.num_ghost)
```

`num_ghost` defaults to 0, so a driver that only ever runs on one rank
may leave it out; on several ranks the first right-hand side then raises
an error that says what to change.

**Print with `root_print`**, which evaluates its arguments on every rank
and prints on rank 0:

```mojo
root_print("t =", t, " mass =", mesh.integrate(q, IDX_RHO))
```

`mesh.integrate`, `integrate_fn`, `l2_error`, `checksum` and
`solver.suggest_dt` return the value over the whole mesh, on every rank.
They are collective: every rank must call them, in the same order. That
is why the argument of a root-only print must not be guarded by
`if is_root():`; rank 0 would wait for the others for ever.

**Combine what you accumulate yourself.** `mesh.fill` and
`mesh.visit_nodes` act on this rank's elements, and so does any loop over
`mesh.num_elements`. Reduce a quantity gathered that way with
`global_sum`, `global_min` or `global_max`:

```mojo
var min_rho: Real = 1.0e30
def visit(x: Real, y: Real, w: Real, v: Array[Real, EULER_NC]) {mut min_rho}:
    min_rho = fmin(min_rho, v[IDX_RHO])
mesh.visit_nodes(q, visit)
min_rho = global_min(min_rho)
```

Element indices are local to the rank, so `State.index(elem, node, comp)`
with a hard-coded element is a single-rank idiom.

Everything else (the `compute_rhs` / `adjust` / `suggest_dt` closures,
sources, positivity preservers, steppers) is unchanged.

## Output

`write_vtu` and the `dump_<phys>_frame` presets are called on every rank
with the same path. On several ranks each writes a piece,
`<stem>_p<rank>.vtu`, and rank 0 writes `<stem>.pvtu`, which lists them
and is the file to open. `FrameWriter.write_pvd()` writes the time series
on rank 0, pointing at the `.pvtu` files.

## Reproducibility

The decomposition does not change the answer. Every node goes through the
same arithmetic whichever rank owns it, and the CFL step is an exact
global maximum, so SSPRK3 runs are bitwise identical for any number of
ranks. `pixi run test-mpi` checks exactly that: it runs each test in
`tests/mpi/` at several rank counts and compares `mesh.checksum(q)`, a
hash of the field that is independent of the decomposition.

Sums over ranks (`integrate`, `l2_error`, the Dormand-Prince error norm)
are accumulated in a different order, so they agree to round-off; an
adaptive Dormand-Prince run can therefore take slightly different steps
on different rank counts.

## Limits

- One element layer of ghosts: enough for DG and the limiter, not for a
  wider stencil.
- The exchange is blocking: interior elements do not start their
  right-hand side while the halo is in flight.
- A mesh needs at least as many elements along an axis as boxes along it.
- `mojo run` cannot run an MPI build reliably; build with `mpi-build` and
  launch the binary with `mpirun`.
