# Maxwell current-source kernel: fluid → Maxwell coupling.
#
# Per node, the fluid contributes a charge-current `J = α · (ρu)` to the
# Ampere-Maxwell equation:
#
#     ∂E/∂t += -J / ε₀  =  -α · (ρu) / ε₀
#
# Only the three E-components of the Maxwell state receive a contribution;
# B components are untouched (no magnetic current). When combined with the
# Lorentz source (`spatial/lorentz_source_2d.mojo`), this closes the
# two-fluid feedback loop and reproduces Langmuir-style plasma oscillations.
#
# Ions (in a two-species plasma) get the same kernel applied with the ion
# charge-to-mass ratio; the existing apply function is per-species so two
# back-to-back calls accumulate the contributions correctly.

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.euler import EULER_NC, IDX_MX, IDX_MY, IDX_MZ
from warpxm.apps.maxwell import MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state_2d import DGState2D


def _current_source_kernel_2d[
    P: Int,
    QFLayout: TensorLayout, RhsMLayout: TensorLayout,
](
    q_fluid: TileTensor[WXM_DTYPE, QFLayout, MutAnyOrigin],
    rhs_maxwell: TileTensor[WXM_DTYPE, RhsMLayout, MutAnyOrigin],
    alpha: Real,
    inv_eps0: Real,
    num_total_nodes: Int,
):
    comptime assert q_fluid.flat_rank == 1
    comptime assert rhs_maxwell.flat_rank == 1

    var n = Int(global_idx.x)
    if n >= num_total_nodes:
        return

    var fb = n * EULER_NC
    var mb = n * MAXWELL_NC

    var mx = rebind[Scalar[WXM_DTYPE]](q_fluid[fb + IDX_MX])
    var my = rebind[Scalar[WXM_DTYPE]](q_fluid[fb + IDX_MY])
    var mz = rebind[Scalar[WXM_DTYPE]](q_fluid[fb + IDX_MZ])

    # ∂E/∂t += -J/ε₀ = -α·(ρu)/ε₀.
    var s_ex: Scalar[WXM_DTYPE] = -alpha * inv_eps0 * mx
    var s_ey: Scalar[WXM_DTYPE] = -alpha * inv_eps0 * my
    var s_ez: Scalar[WXM_DTYPE] = -alpha * inv_eps0 * mz

    var r_ex = rebind[Scalar[WXM_DTYPE]](rhs_maxwell[mb + IDX_EX])
    var r_ey = rebind[Scalar[WXM_DTYPE]](rhs_maxwell[mb + IDX_EY])
    var r_ez = rebind[Scalar[WXM_DTYPE]](rhs_maxwell[mb + IDX_EZ])
    rhs_maxwell[mb + IDX_EX] = rebind[rhs_maxwell.ElementType](r_ex + s_ex)
    rhs_maxwell[mb + IDX_EY] = rebind[rhs_maxwell.ElementType](r_ey + s_ey)
    rhs_maxwell[mb + IDX_EZ] = rebind[rhs_maxwell.ElementType](r_ez + s_ez)


def apply_current_source_2d[P: Int](
    var ctx: DeviceContext,
    mut q_fluid: DGState2D[P, EULER_NC],
    mut rhs_maxwell: DGState2D[P, MAXWELL_NC],
    alpha: Real,
    eps0: Real,
) raises:
    """Add `-α · (ρu) / ε₀` to the E components of `rhs_maxwell`."""
    var num_total_nodes = q_fluid.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var fl = row_major(Idx(q_fluid.buf_len))
    var ml = row_major(Idx(rhs_maxwell.buf_len))

    comptime kernel = _current_source_kernel_2d[P, type_of(fl), type_of(ml)]
    ctx.enqueue_function[kernel](
        TileTensor(q_fluid.buf, fl),
        TileTensor(rhs_maxwell.buf, ml),
        alpha, Real(1.0) / eps0, num_total_nodes,
        grid_dim=grid, block_dim=BLOCK,
    )
