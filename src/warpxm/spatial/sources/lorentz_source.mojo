# Lorentz-force source kernel: Maxwell → 5-moment Euler coupling.
#
# Per fluid element with charge-to-mass ratio α = q / m:
#
#     S[ρ]    = 0
#     S[ρu_x] = α · ρ · E_x + α · ( ρu_y · B_z − ρu_z · B_y )
#     S[ρu_y] = α · ρ · E_y + α · ( ρu_z · B_x − ρu_x · B_z )
#     S[ρu_z] = α · ρ · E_z + α · ( ρu_x · B_y − ρu_y · B_x )
#     S[E]    = α · ( ρu_x · E_x + ρu_y · E_y + ρu_z · E_z )
#
# i.e. `S_momentum = α·(ρE + ρu × B)` and `S_energy = α · (ρu) · E`.
# The magnetic component does no work; only E enters the energy source.
#
# Reads the per-node (E, B) directly from a `DGState[NDIM, P, MAXWELL_NC]` so
# the same kernel works whether the EM field is statically prescribed or
# being evolved by Maxwell. The reverse-direction current source on Maxwell
# (Euler → Maxwell, closing the two-fluid plasma coupling) lives in
# `maxwell_current_source.mojo`.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.apps.maxwell import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state import DGState


def _lorentz_source_kernel[
    P: Int,
    QFLayout: TensorLayout, QMLayout: TensorLayout, RhsLayout: TensorLayout,
](
    q_fluid: TileTensor[WXM_DTYPE, QFLayout, MutAnyOrigin],
    q_maxwell: TileTensor[WXM_DTYPE, QMLayout, MutAnyOrigin],
    rhs_fluid: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    alpha: Real,
    num_total_nodes_i64: Int64,
):
    var num_total_nodes = Int(num_total_nodes_i64)
    comptime assert q_fluid.flat_rank == 1
    comptime assert q_maxwell.flat_rank == 1
    comptime assert rhs_fluid.flat_rank == 1

    var n = Int(global_idx.x)
    if n >= num_total_nodes:
        return

    var fb = n * EULER_NC
    var mb = n * MAXWELL_NC

    var rho = rebind[Scalar[WXM_DTYPE]](q_fluid[fb + IDX_RHO])
    var mx = rebind[Scalar[WXM_DTYPE]](q_fluid[fb + IDX_MX])
    var my = rebind[Scalar[WXM_DTYPE]](q_fluid[fb + IDX_MY])
    var mz = rebind[Scalar[WXM_DTYPE]](q_fluid[fb + IDX_MZ])

    var Ex = rebind[Scalar[WXM_DTYPE]](q_maxwell[mb + IDX_EX])
    var Ey = rebind[Scalar[WXM_DTYPE]](q_maxwell[mb + IDX_EY])
    var Ez = rebind[Scalar[WXM_DTYPE]](q_maxwell[mb + IDX_EZ])
    var Bx = rebind[Scalar[WXM_DTYPE]](q_maxwell[mb + IDX_BX])
    var By = rebind[Scalar[WXM_DTYPE]](q_maxwell[mb + IDX_BY])
    var Bz = rebind[Scalar[WXM_DTYPE]](q_maxwell[mb + IDX_BZ])

    # m × B  with  m = (mx, my, mz)  and  B = (Bx, By, Bz).
    var mxB_x: Scalar[WXM_DTYPE] = my * Bz - mz * By
    var mxB_y: Scalar[WXM_DTYPE] = mz * Bx - mx * Bz
    var mxB_z: Scalar[WXM_DTYPE] = mx * By - my * Bx

    var s_mx: Scalar[WXM_DTYPE] = alpha * (rho * Ex + mxB_x)
    var s_my: Scalar[WXM_DTYPE] = alpha * (rho * Ey + mxB_y)
    var s_mz: Scalar[WXM_DTYPE] = alpha * (rho * Ez + mxB_z)
    var s_e: Scalar[WXM_DTYPE] = alpha * (mx * Ex + my * Ey + mz * Ez)

    var r_mx = rebind[Scalar[WXM_DTYPE]](rhs_fluid[fb + IDX_MX])
    var r_my = rebind[Scalar[WXM_DTYPE]](rhs_fluid[fb + IDX_MY])
    var r_mz = rebind[Scalar[WXM_DTYPE]](rhs_fluid[fb + IDX_MZ])
    var r_e = rebind[Scalar[WXM_DTYPE]](rhs_fluid[fb + IDX_E])

    rhs_fluid[fb + IDX_MX] = rebind[rhs_fluid.ElementType](r_mx + s_mx)
    rhs_fluid[fb + IDX_MY] = rebind[rhs_fluid.ElementType](r_my + s_my)
    rhs_fluid[fb + IDX_MZ] = rebind[rhs_fluid.ElementType](r_mz + s_mz)
    rhs_fluid[fb + IDX_E] = rebind[rhs_fluid.ElementType](r_e + s_e)


def apply_lorentz_source[NDIM: Int, P: Int](
    var ctx: DeviceContext,
    mut q_fluid: DGState[NDIM, P, EULER_NC],
    mut q_maxwell: DGState[NDIM, P, MAXWELL_NC],
    mut rhs_fluid: DGState[NDIM, P, EULER_NC],
    alpha: Real,
) raises:
    """Add α·(ρE + ρu × B) and α·ρu·E to `rhs_fluid` from `q_maxwell`."""
    var num_total_nodes = q_fluid.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var fl = row_major(Int(q_fluid.buf_len))
    var ml = row_major(Int(q_maxwell.buf_len))

    comptime kernel = _lorentz_source_kernel[
        P, type_of(fl), type_of(ml), type_of(fl),
    ]
    ctx.enqueue_function[kernel](
        TileTensor(q_fluid.buf, fl),
        TileTensor(q_maxwell.buf, ml),
        TileTensor(rhs_fluid.buf, fl),
        alpha, Int64(num_total_nodes),
        grid_dim=grid, block_dim=BLOCK,
    )
