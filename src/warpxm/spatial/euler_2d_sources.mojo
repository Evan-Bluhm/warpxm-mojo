# Source-term kernels for the 2D 5-moment Euler solver.
#
# The DG kernel in `dg_euler_2d.mojo` writes `rhs = -∇·F(q)`. Anything that
# adds to the right-hand side outside the divergence form — body forces,
# Lorentz coupling, reaction sources, GLM cleaning — lives in this file as
# a separate post-pass kernel. Each kernel reads `q` and `rhs` of the same
# `DGState2D[P, NC]` and updates `rhs` in place.
#
# Today's catalogue is intentionally minimal: just a uniform-gravity body
# force. The Lorentz / two-fluid coupling and the MixedGLM source will
# follow the same shape — one kernel each, dispatched from the stepper
# after the spatial divergence kernel.

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.euler_2d import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_E
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state_2d import DGState2D


# ---------------------------------------------------------------------------
# Kernel: add the uniform-gravity body force to rhs.
#
#     S[ρ]    = 0
#     S[ρu_x] = ρ · g_x
#     S[ρu_y] = ρ · g_y
#     S[ρu_z] = 0
#     S[E]    = ρu_x · g_x + ρu_y · g_y     (= ρ·u·g, the work done by gravity)
#
# One thread per (element, node) — flat over the whole grid.
# ---------------------------------------------------------------------------

def _gravity_source_kernel_2d[
    P: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    gx: Real,
    gy: Real,
    num_total_nodes: Int,
):
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

    var src_mx: Scalar[WXM_DTYPE] = rho * gx
    var src_my: Scalar[WXM_DTYPE] = rho * gy
    # Work done by gravity on the fluid element: g·(ρu) = g·m, where
    # m = (ρu_x, ρu_y) is the momentum density we already loaded.
    var src_e: Scalar[WXM_DTYPE] = mx * gx + my * gy

    var rhs_mx = rebind[Scalar[WXM_DTYPE]](rhs[base + IDX_MX])
    var rhs_my = rebind[Scalar[WXM_DTYPE]](rhs[base + IDX_MY])
    var rhs_e = rebind[Scalar[WXM_DTYPE]](rhs[base + IDX_E])
    rhs[base + IDX_MX] = rebind[rhs.ElementType](rhs_mx + src_mx)
    rhs[base + IDX_MY] = rebind[rhs.ElementType](rhs_my + src_my)
    rhs[base + IDX_E] = rebind[rhs.ElementType](rhs_e + src_e)


# ---------------------------------------------------------------------------
# Host-side driver.
# ---------------------------------------------------------------------------

def apply_gravity_source_2d[P: Int](
    var ctx: DeviceContext,
    mut q: DGState2D[P, EULER_NC],
    mut rhs: DGState2D[P, EULER_NC],
    gx: Real,
    gy: Real,
) raises:
    """Add the uniform-gravity source to `rhs` in place.

    Called after `DGEulerSolver2D.compute_rhs` and before the stage combine
    step of the SSPRK3 integrator.
    """
    var num_total_nodes = q.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var layout = row_major(Idx(q.buf_len))

    comptime kernel = _gravity_source_kernel_2d[
        P, type_of(layout), type_of(layout),
    ]
    ctx.enqueue_function[kernel](
        TileTensor(q.buf, layout),
        TileTensor(rhs.buf, layout),
        gx, gy, num_total_nodes,
        grid_dim=grid, block_dim=BLOCK,
    )
