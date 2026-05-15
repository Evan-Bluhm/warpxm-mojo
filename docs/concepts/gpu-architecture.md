# GPU architecture in Mojo

warpxm-mojo runs entirely on the GPU. This page explains how a typical
DG kernel is launched, what `comptime` buys us, and how the device-side
plumbing fits together.

## One thread per node, blocks per element

Every DG kernel uses the convention:

* `grid_dim = num_elements`
* `block_dim = (P+1, P+1)` (2D) or `(P+1, P+1, P+1)` (3D)

i.e. one CUDA / HIP / Metal **thread** per GLL node, one **block** per
element. With $P = 2$ and a 2D problem that's a $3 \times 3$ block of 9
threads. Each thread:

1. Loads its own node into a register-resident `InlineArray[Real, NC]`.
2. Computes its contribution to the volume term by reading the *other*
   nodes in the same element (a small constant number — $P+1$ on each
   tensor axis).
3. If it's a face node (e.g., `in_ == 0` for the left face), reads its
   neighbour's face node and adds the surface term.
4. Writes back its share of the new RHS.

No shared memory, no warp shuffles, no atomics. The block size is
small ($(P+1)^d$ is between 4 and 125 across our supported orders),
which means low occupancy but very high arithmetic-to-load ratio.

## `comptime` everywhere

Polynomial order `P`, component count `NC`, dimension, and even
`USE_HLLD` are **compile-time parameters**. They appear in struct and
kernel signatures as `[P: Int, NC: Int, ...]`:

```mojo
def _imhd_rhs_kernel_2d[
    P: Int, USE_HLLD: Bool,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    ...
):
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1
    ...
    comptime for jn in range(P1):
        comptime for in_ in range(P1):
            ...
```

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
var q_layout = row_major(Idx(q.buf_len))
var q_view = TileTensor(q.buf, q_layout)
ctx.enqueue_function[kernel](q_view, ..., grid_dim=..., block_dim=...)
```

`TileTensor` makes the per-element addressing safe (it carries the
layout) and works with Mojo's aliasing checker. The trade-off: identical
buffers passed twice to a kernel as `MutAnyOrigin` arguments are
*rejected* by the checker, which is why the Dormand-Prince stepper has
six specialised combine kernels for `(1, 2, 3, 4, 5, 5)`-input arities
rather than one variadic kernel.

## Index types

* `Real` is `Scalar[DType.float64]`. All physics quantities are double
  precision.
* `INDEX_DTYPE` is `Int32`. Element / face index buffers use it; the
  domain sizes we run at fit comfortably.

These are defined in [`core/types.mojo`](../api/core.md). Switching to
single precision would be a one-line edit but hasn't been needed.

## SQRT on NVIDIA float64

`std.math.sqrt` doesn't currently provide a `float64` overload on NVIDIA
GPUs. We work around it with `x ** Real(0.5)` exposed as
`math_compat.sqrt`:

```mojo
from warpxm.core.math_compat import sqrt   # not std.math.sqrt
```

Every kernel that needs a square root (fast magnetosonic speed, HLLD
star-state Alfvén speed, etc.) routes through this helper.

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
We have *not* yet validated AMD / Apple at the full 49-test scale;
NVIDIA (A100, H100, RTX) is the primary CI target. If you hit a vendor-
specific issue, please open a bug.
