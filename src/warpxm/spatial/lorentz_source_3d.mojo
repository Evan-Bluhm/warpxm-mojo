# 3D Lorentz-force source kernel: Maxwell → 5-moment Euler coupling.
#
# Identical physics to `lorentz_source_2d.mojo`; only the wrapper changes
# to take 3D state types. The kernel itself is dimension-agnostic — it
# operates on flat per-node buffers — so we delegate to the 2D kernel
# function definition.

from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, Idx, row_major

from warpxm.apps.euler import EULER_NC
from warpxm.apps.maxwell import MAXWELL_NC
from warpxm.core.types import Real
from warpxm.spatial.lorentz_source_2d import _lorentz_source_kernel_2d
from warpxm.variable.state_3d import DGState3D


def apply_lorentz_source_3d[P: Int](
    var ctx: DeviceContext,
    mut q_fluid: DGState3D[P, EULER_NC],
    mut q_maxwell: DGState3D[P, MAXWELL_NC],
    mut rhs_fluid: DGState3D[P, EULER_NC],
    alpha: Real,
) raises:
    """Add α·(ρE + ρu × B) and α·ρu·E to `rhs_fluid` from `q_maxwell`."""
    var num_total_nodes = q_fluid.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var fl = row_major(Idx(q_fluid.buf_len))
    var ml = row_major(Idx(q_maxwell.buf_len))

    comptime kernel = _lorentz_source_kernel_2d[
        P, type_of(fl), type_of(ml), type_of(fl),
    ]
    ctx.enqueue_function[kernel](
        TileTensor(q_fluid.buf, fl),
        TileTensor(q_maxwell.buf, ml),
        TileTensor(rhs_fluid.buf, fl),
        alpha, num_total_nodes,
        grid_dim=grid, block_dim=BLOCK,
    )
