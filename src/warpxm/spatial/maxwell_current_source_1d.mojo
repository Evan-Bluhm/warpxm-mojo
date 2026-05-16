# 1D Maxwell current source: fluid → Maxwell coupling. Delegates to the
# 2D kernel function definition.

from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, Idx, row_major

from warpxm.apps.euler import EULER_NC
from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.core.types import Real
from warpxm.spatial.maxwell_current_source_2d import _current_source_kernel_2d
from warpxm.variable.state_1d import DGState1D


def apply_current_source_1d[P: Int](
    var ctx: DeviceContext,
    mut q_fluid: DGState1D[P, EULER_NC],
    mut rhs_maxwell: DGState1D[P, MAXWELL_NC],
    alpha: Real,
    eps0: Real,
) raises:
    """Add -α·ρu/ε₀ to `rhs_maxwell.E_*` from the fluid's momentum."""
    var num_total_nodes = q_fluid.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var fl = row_major(Idx(q_fluid.buf_len))
    var ml = row_major(Idx(rhs_maxwell.buf_len))

    comptime kernel = _current_source_kernel_2d[
        P, type_of(fl), type_of(ml),
    ]
    ctx.enqueue_function[kernel](
        TileTensor(q_fluid.buf, fl),
        TileTensor(rhs_maxwell.buf, ml),
        alpha, eps0, num_total_nodes,
        grid_dim=grid, block_dim=BLOCK,
    )
