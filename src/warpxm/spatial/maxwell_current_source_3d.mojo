# 3D Maxwell current-source kernel: fluid → Maxwell coupling.
#
# Thin wrapper around the dimension-agnostic `_current_source_kernel_2d`
# kernel; only the state types differ.

from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, Idx, row_major

from warpxm.apps.euler import EULER_NC
from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.core.types import Real
from warpxm.spatial.maxwell_current_source_2d import _current_source_kernel_2d
from warpxm.variable.state_3d import DGState3D


def apply_current_source_3d[P: Int](
    var ctx: DeviceContext,
    mut q_fluid: DGState3D[P, EULER_NC],
    mut rhs_maxwell: DGState3D[P, MAXWELL_NC],
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
