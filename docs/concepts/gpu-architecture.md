# GPU architecture in Mojo

warpxm-mojo runs entirely on the GPU. This page explains how a typical
DG kernel is launched, what `comptime` buys us, and how the device-side
plumbing fits together.

## One thread per node, a few elements per block

Every DG kernel uses the convention (`EB = DG_CORE_ELEMS_PER_BLOCK = 4`):

* `grid_dim = ceildiv(num_elements, EB)`
* `block_dim = (P+1, EB)` (1D), `(P+1, P+1, EB)` (2D) or
  `(P+1, P+1, (P+1) * EB)` (3D)

i.e. one CUDA / HIP / Metal **thread** per GLL node, with `EB`
independent elements packed into each **block** so that small elements
still fill a warp. With $P = 2$ and a 2D problem that's
$3 \times 3 \times 4 = 36$ threads per block. Each thread:

1. Loads its own conserved-variable values into a register-resident
   `Array[Real, NC]` and evaluates the flux $\mathbf{F}(\mathbf{q})$
   at its node.
2. Writes those flux values into a per-element **shared-memory** stage
   (`f_shared` in `dg_core.mojo`); a `barrier()` then makes
   them visible to the rest of the block.
3. Reads the flux values written by the other $P$ threads along each
   tensor axis and accumulates the volume-derivative contribution.
4. If it's a face node (e.g., `in_ == 0` for the left face), reads its
   neighbour element's face-node state and adds the surface term.
5. Writes its share of the new RHS back to global memory.

No warp shuffles, no atomics — but shared memory + one `barrier()` call
are how the tensor-product derivative is fanned across the block.

Every thread of a block must reach that barrier at the same call site:
a barrier on a divergent path deadlocks the warp. When the element count
is not a multiple of `EB`, the last block is only partly filled, so the
threads past the last element do not return early. They redo the last
element's flux into their own shared-memory slot, pass the barrier with
the others, and stop before writing anything
(`tests/test_advection_odd_mesh.mojo`).

## `comptime` everywhere

Polynomial order `P`, component count `NC`, dimension, and even
`USE_HLLD` are **compile-time parameters**. They appear in struct and
kernel signatures as `[P: Int, NC: Int, ...]`:

```mojo
def _imhd_rhs_kernel[NDIM: Int, P: Int, USE_HLLD: Bool](
    q: RealTensor,
    rhs: RealTensor,
    ...
    params: KernelParams,
):
    var gamma = params[0]
    ...
    def num_flux(q_l, q_r, nx, ny, nz) {imm} -> Array[Real, MHD_NC]:
        comptime if USE_HLLD:
            return IMHD.numerical_flux_hlld_static(...)
        else:
            return IMHD.numerical_flux_static(...)
    _dg_compute_rhs_core[NDIM, P, MHD_NC, ...](
        ..., flux_x, flux_y, flux_z, num_flux, apply_bc,
    )
```

The dimension is a `comptime` parameter too: `BlockMesh[NDIM, P]`,
`DGState[NDIM, P, NC]`, the DG core and solvers, the adjusters,
`SSPRK3[NDIM, P, NC]` and the pointwise source kernels are written once
for 1D, 2D and 3D. In the DG core every loop over axes is a
`comptime for`, so nothing of the genericity is left in the kernel.

`comptime for` unrolls the loop. The inner volume-derivative loops in
every DG kernel are unrolled per polynomial order — there is no runtime
`P` involved in the hot path.

## `has_accelerator()` as a precondition

```mojo
comptime assert has_accelerator(), "warpxm-mojo requires a GPU"
```

This appears at the top of every example's `main()`. It's a *compile-
time* assertion: a build for a CPU-only target won't even compile, let
alone run. There is no runtime fallback path. See
[Design departures](design-departures.md) for the motivation.

## `TileTensor` for buffers

We don't pass raw `UnsafePointer`s to kernels — we wrap each
`DeviceBuffer[Real.dtype]` in a `TileTensor` with a known layout:

```mojo
var q_layout = row_major(Int(q.buf_len))
var q_view = TileTensor(q.buf, q_layout)
ctx.enqueue_function[kernel](q_view, ..., grid_dim=..., block_dim=...)
```

For the DG kernels this plumbing is written once, in
`spatial/dg_launch.mojo`. A flat buffer of runtime length has the same
layout type whatever its length (`FlatLayout`), so every DG kernel
shares one signature and `launch_dg[kernel](ctx, mesh, q, rhs, params)`
can enqueue any of them, in any dimension. Two Mojo restrictions
shape that signature: a function cannot forward a variadic argument pack
into `enqueue_function` next to fixed arguments, and a user struct is not
`DevicePassable`. The physics scalars therefore travel as one
`KernelParams` vector (`SIMD[WXM_DTYPE, 8]`), packed with
`kernel_params(...)` on the host and unpacked by position in the kernel.
See [`spatial`](../api/spatial.md#dg_launch).

`TileTensor` makes the per-element addressing safe (it carries the
layout) and works with Mojo's aliasing checker. The trade-off: identical
buffers passed twice to a kernel as `MutAnyOrigin` arguments are
*rejected* by the checker, which is why the Dormand-Prince stepper has
five specialised combine kernels (`_combine1`–`_combine5`, for 1–5
input arities) rather than one variadic kernel; `_combine5` is the
only one called from two stages.

## Index types

* `Real` is `Scalar[DType.float64]`. All physics quantities are double
  precision.
* `INDEX_DTYPE` is `Int32`. Element / face index buffers use it; the
  domain sizes we run at fit comfortably.

These are defined in [`core/types.mojo`](../api/core.md). Float64 is the
only precision that has been run or tested.

Kernel scalar arguments that are integers are passed as `Int64` (`Int`
is not `DevicePassable`) and converted with `Int(...)` in the kernel
body.

## SQRT on NVIDIA float64

`std.math.sqrt` doesn't currently provide a `float64` overload on NVIDIA
GPUs. `math_compat.sqrt` calls the `llvm.sqrt` intrinsic instead, which
lowers to the native double-precision square root:

```mojo
from warpxm.core.math_compat import sqrt   # not std.math.sqrt
```

Every kernel that needs a square root (fast magnetosonic speed, HLLD
star-state Alfvén speed, etc.) routes through this helper.

Do not write `x ** Real(0.5)` in a kernel. It compiles, but to a software
`pow`; an earlier version of this helper did exactly that, and it was the
largest single cost in the Euler and MHD right-hand-side kernels.

## Launch overhead and CPU-bound bottlenecks

Each DG step issues ~10 GPU kernels (volume + face flux + per-axis
combine + limiter + positivity, etc.). For very small meshes (NX < 64,
P = 1) the launch overhead can dominate. The practical lower bound on
problem size we've validated against is `NX = NY = 32, P = 2` (~10000
GLL nodes); below that, kernels are dispatching faster than they
execute. For typical research-scale problems (NX ≥ 64, P ≥ 2) launches
are a few percent of wall-clock time.

## Cross-vendor portability

The same Mojo source compiles for NVIDIA, AMD CDNA, and Apple silicon
without conditional compilation. Vendor differences in warp size,
shared-memory size etc. don't matter for our small-block-size kernels.
We have *not* yet validated AMD / Apple with the full regression suite;
NVIDIA (A100, H100, RTX) is the primary CI target. If you hit a vendor-
specific issue, please open a bug.
