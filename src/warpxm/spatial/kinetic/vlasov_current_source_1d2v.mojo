# Vlasov → Maxwell current source (1D2V).
#
# Two-component analogue of `vlasov_current_source_1d.mojo`. With one
# physical-space axis (x) and two velocity axes (v1, v2), the species
# carries a real two-component current. Per x-node:
#
#     ∂E_x/∂t += -α · J_x / ε₀
#     ∂E_y/∂t += -α · J_y / ε₀
#
# where `J_x(x) = ∫∫ v1 f dv1 dv2` and `J_y(x) = ∫∫ v2 f dv1 dv2` are the
# 1D2V first moments. The 1D Maxwell solver already evolves both
# transverse components of `E` — this source just feeds them. `α = q/m`
# for the species; `1/ε₀` collapses with `λ_skin` upstream.
#
# Single fused kernel: both updates touch the same (x-node, Maxwell
# component) record, so a single thread per x-node updates both. Half
# the launch overhead vs. running the 1D version twice.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.maxwell import MAXWELL_NC, IDX_EX, IDX_EY
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state import DGState


def _vlasov_current_source_kernel_1d2v[
    P: Int,
    JLayout: TensorLayout, RhsMLayout: TensorLayout,
](
    j_x: TileTensor[WXM_DTYPE, JLayout, MutAnyOrigin],
    j_y: TileTensor[WXM_DTYPE, JLayout, MutAnyOrigin],
    rhs_maxwell: TileTensor[WXM_DTYPE, RhsMLayout, MutAnyOrigin],
    alpha: Real,
    inv_eps0: Real,
    num_total_nodes_i64: Int64,
):
    var num_total_nodes = Int(num_total_nodes_i64)
    comptime assert j_x.flat_rank == 1
    comptime assert j_y.flat_rank == 1
    comptime assert rhs_maxwell.flat_rank == 1

    var n = Int(global_idx.x)
    if n >= num_total_nodes:
        return

    var jx_n = rebind[Scalar[WXM_DTYPE]](j_x[n])
    var jy_n = rebind[Scalar[WXM_DTYPE]](j_y[n])
    var mb = n * MAXWELL_NC
    var r_ex = rebind[Scalar[WXM_DTYPE]](rhs_maxwell[mb + IDX_EX])
    var r_ey = rebind[Scalar[WXM_DTYPE]](rhs_maxwell[mb + IDX_EY])
    rhs_maxwell[mb + IDX_EX] = rebind[rhs_maxwell.ElementType](
        r_ex + (-alpha * inv_eps0) * jx_n
    )
    rhs_maxwell[mb + IDX_EY] = rebind[rhs_maxwell.ElementType](
        r_ey + (-alpha * inv_eps0) * jy_n
    )


def apply_vlasov_current_source_1d2v[P: Int](
    var ctx: DeviceContext,
    mut j_x: DGState[1, P, 1],
    mut j_y: DGState[1, P, 1],
    mut rhs_maxwell: DGState[1, P, MAXWELL_NC],
    alpha: Real,
    eps0: Real,
) raises:
    """Add `-α · (J_x, J_y) / ε₀` to `rhs_maxwell.(E_x, E_y)` from the
    1D2V Vlasov first moments. `j_x`, `j_y`, and `rhs_maxwell` share the
    same per-x-node grid (`P` and total-node count must match)."""
    var num_total_nodes = j_x.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var jl = row_major(Int(j_x.buf_len))
    var ml = row_major(Int(rhs_maxwell.buf_len))

    comptime kernel = _vlasov_current_source_kernel_1d2v[
        P, type_of(jl), type_of(ml),
    ]
    ctx.enqueue_function[kernel](
        TileTensor(j_x.buf, jl),
        TileTensor(j_y.buf, jl),
        TileTensor(rhs_maxwell.buf, ml),
        alpha, Real(1.0) / eps0, Int64(num_total_nodes),
        grid_dim=grid, block_dim=BLOCK,
    )
