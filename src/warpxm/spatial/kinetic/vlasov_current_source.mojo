# Vlasov → Maxwell current source, generic over the dimensions.
#
# Adds the current of one kinetic species to Ampère's law:
#
#     ∂E_d/∂t += −α · Γ_d / ε₀        d = x, y, z up to VDIM
#
# where `Γ_d = ∫ v_d f dv` are the `MOM_FLUX_*` moments from
# `VlasovMoments` and `α = q/m` for the species. Call it once per species.
#
# The moments and the Maxwell state live on the same
# `BlockMesh[XDIM, P]`, so this is a pointwise kernel over its nodes.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, row_major

from warpxm.apps.maxwell import MAXWELL_NC, IDX_EX
from warpxm.core.types import Real
from warpxm.spatial.dg_launch import RealTensor
from warpxm.spatial.kinetic.vlasov_moments import VLASOV_MOMENT_NC, MOM_FLUX_X
from warpxm.variable.state import DGState


def _vlasov_current_source_kernel[VDIM: Int](
    moments: RealTensor,
    rhs_maxwell: RealTensor,
    coefficient: Real,
    num_total_nodes_i64: Int64,
):
    var n = Int(global_idx.x)
    if n >= Int(num_total_nodes_i64):
        return
    comptime for d in range(VDIM):
        var flux = rebind[Real](moments[n * VLASOV_MOMENT_NC + MOM_FLUX_X + d])
        var r = rebind[Real](rhs_maxwell[n * MAXWELL_NC + IDX_EX + d])
        rhs_maxwell[n * MAXWELL_NC + IDX_EX + d] = rebind[
            rhs_maxwell.ElementType
        ](r + coefficient * flux)


def apply_vlasov_current_source[XDIM: Int, VDIM: Int, P: Int](
    var ctx: DeviceContext,
    mut moments: DGState[XDIM, P, VLASOV_MOMENT_NC],
    mut rhs_maxwell: DGState[XDIM, P, MAXWELL_NC],
    alpha: Real,
    eps0: Real,
) raises:
    """Add `−α · Γ / ε₀` to the `E` components of `rhs_maxwell`, for the
    first `VDIM` components of the flux moments in `moments`."""
    comptime assert VDIM >= 1 and VDIM <= 3, "VDIM must be 1, 2 or 3"
    comptime BLOCK = 256
    var num_total_nodes = moments.total_nodes
    var mom_layout = row_major(Int(moments.buf_len))
    var rhs_layout = row_major(Int(rhs_maxwell.buf_len))
    ctx.enqueue_function[_vlasov_current_source_kernel[VDIM]](
        TileTensor(moments.buf, mom_layout),
        TileTensor(rhs_maxwell.buf, rhs_layout),
        -alpha / eps0, Int64(num_total_nodes),
        grid_dim=ceildiv(num_total_nodes, BLOCK), block_dim=BLOCK,
    )
