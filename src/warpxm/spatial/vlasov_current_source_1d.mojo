# Vlasov → Maxwell current source (1D).
#
# Mirrors `spatial/maxwell_current_source_1d.mojo` but the "current"
# input is the species first moment `J(x) = ∫ v · f(x, v) dv` rather
# than the fluid's full momentum vector. Per x-node:
#
#     ∂E_x/∂t += -α · J / ε₀
#
# `α = (q/m)` for the species; `1/ε₀` collapses with the skin-depth
# normalisation `λ_skin` upstream — this kernel keeps them as separate
# scalars so the caller can stay close to the upstream parameterisation.
# For 1D1V only `E_x` receives a contribution (the v-axis is parallel
# to x; transverse currents are absent).

from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.maxwell import MAXWELL_NC, IDX_EX
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state_1d import DGState1D


def _vlasov_current_source_kernel_1d[
    P: Int,
    JLayout: TensorLayout, RhsMLayout: TensorLayout,
](
    j: TileTensor[WXM_DTYPE, JLayout, MutAnyOrigin],
    rhs_maxwell: TileTensor[WXM_DTYPE, RhsMLayout, MutAnyOrigin],
    alpha: Real,
    inv_eps0: Real,
    num_total_nodes: Int,
):
    comptime assert j.flat_rank == 1
    comptime assert rhs_maxwell.flat_rank == 1

    var n = Int(global_idx.x)
    if n >= num_total_nodes:
        return

    var jn = rebind[Scalar[WXM_DTYPE]](j[n])
    var mb = n * MAXWELL_NC
    var r_ex = rebind[Scalar[WXM_DTYPE]](rhs_maxwell[mb + IDX_EX])
    rhs_maxwell[mb + IDX_EX] = rebind[rhs_maxwell.ElementType](
        r_ex + (-alpha * inv_eps0) * jn
    )


def apply_vlasov_current_source_1d[P: Int](
    var ctx: DeviceContext,
    mut j: DGState1D[P, 1],
    mut rhs_maxwell: DGState1D[P, MAXWELL_NC],
    alpha: Real,
    eps0: Real,
) raises:
    """Add `-α · J / ε₀` to `rhs_maxwell.E_x` from the Vlasov first
    moment `J`. `j` and `rhs_maxwell` share the same per-x-node grid
    (`P` and total-node count must match)."""
    var num_total_nodes = j.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var jl = row_major(Idx(j.buf_len))
    var ml = row_major(Idx(rhs_maxwell.buf_len))

    comptime kernel = _vlasov_current_source_kernel_1d[
        P, type_of(jl), type_of(ml),
    ]
    ctx.enqueue_function[kernel](
        TileTensor(j.buf, jl),
        TileTensor(rhs_maxwell.buf, ml),
        alpha, Real(1.0) / eps0, num_total_nodes,
        grid_dim=grid, block_dim=BLOCK,
    )
