# Uniform-gravity body force for 5-moment Euler, in any dimension.
#
# The DG kernels write `rhs = -∇·F(q)`. Anything that adds to the
# right-hand side outside the divergence form is a separate pointwise
# kernel in this folder, called from the driver's `compute_rhs` closure
# after the spatial solver: this file (gravity), `lorentz_source.mojo`
# (Maxwell → Euler), `maxwell_current_source.mojo` (Euler → Maxwell) and
# `imhd_glm_source.mojo` (parabolic ψ damping). Each reads `q` and adds to
# `rhs` in place, and is generic over `NDIM` because the state buffer is
# flat.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state import DGState


# ---------------------------------------------------------------------------
# Kernel: add the uniform-gravity body force to rhs.
#
#     S[ρ]    = 0
#     S[ρu_x] = ρ · g_x
#     S[ρu_y] = ρ · g_y
#     S[ρu_z] = ρ · g_z
#     S[E]    = ρu · g                      (the work done by gravity)
#
# One thread per (element, node) — flat over the whole grid.
# ---------------------------------------------------------------------------

def _gravity_source_kernel[
    P: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    gx: Real,
    gy: Real,
    gz: Real,
    num_total_nodes_i64: Int64,
):
    var num_total_nodes = Int(num_total_nodes_i64)
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime NC = EULER_NC

    var n = Int(global_idx.x)
    if n >= num_total_nodes:
        return

    var base = n * NC
    var rho = rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
    var mx = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MX])
    var my = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MY])
    var mz = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MZ])

    var src_mx: Scalar[WXM_DTYPE] = rho * gx
    var src_my: Scalar[WXM_DTYPE] = rho * gy
    var src_mz: Scalar[WXM_DTYPE] = rho * gz
    # Work done by gravity on the fluid element: g·(ρu) = g·m.
    var src_e: Scalar[WXM_DTYPE] = mx * gx + my * gy + mz * gz

    var rhs_mx = rebind[Scalar[WXM_DTYPE]](rhs[base + IDX_MX])
    var rhs_my = rebind[Scalar[WXM_DTYPE]](rhs[base + IDX_MY])
    var rhs_mz = rebind[Scalar[WXM_DTYPE]](rhs[base + IDX_MZ])
    var rhs_e = rebind[Scalar[WXM_DTYPE]](rhs[base + IDX_E])
    rhs[base + IDX_MZ] = rebind[rhs.ElementType](rhs_mz + src_mz)
    rhs[base + IDX_MX] = rebind[rhs.ElementType](rhs_mx + src_mx)
    rhs[base + IDX_MY] = rebind[rhs.ElementType](rhs_my + src_my)
    rhs[base + IDX_E] = rebind[rhs.ElementType](rhs_e + src_e)


# ---------------------------------------------------------------------------
# Host-side driver.
# ---------------------------------------------------------------------------

def apply_gravity_source[NDIM: Int, P: Int](
    var ctx: DeviceContext,
    mut q: DGState[NDIM, P, EULER_NC],
    mut rhs: DGState[NDIM, P, EULER_NC],
    gx: Real,
    gy: Real,
    gz: Real = 0.0,
) raises:
    """Add the uniform-gravity source to `rhs` in place.

    Called after the Euler spatial solver and before the stage combine
    step of the SSPRK3 integrator.
    """
    var num_total_nodes = q.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var layout = row_major(Int(q.buf_len))

    comptime kernel = _gravity_source_kernel[
        P, type_of(layout), type_of(layout),
    ]
    ctx.enqueue_function[kernel](
        TileTensor(q.buf, layout),
        TileTensor(rhs.buf, layout),
        gx, gy, gz, Int64(num_total_nodes),
        grid_dim=grid, block_dim=BLOCK,
    )
