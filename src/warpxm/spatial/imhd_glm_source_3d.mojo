# 3D GLM ψ-damping wrapper. The kernel function is dimension-agnostic
# (operates on a flat per-node buffer), so we just thread a 3D state in.

from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, Idx, row_major

from warpxm.apps.imhd_glm_2d import MHD_GLM_NC
from warpxm.core.types import Real
from warpxm.spatial.imhd_glm_source_2d import _glm_damping_kernel
from warpxm.variable.state_3d import DGState3D


def apply_glm_damping_3d[P: Int, NC: Int](
    var ctx: DeviceContext,
    mut q: DGState3D[P, NC],
    mut rhs: DGState3D[P, NC],
    c_h: Real,
    c_p_sq: Real,
) raises:
    """Apply `∂ψ/∂t += -(c_h² / c_p²) · ψ` in place on `rhs`.

    `NC` is parameterised so the same wrapper works for `MHD_GLM_NC = 9`
    today and any future GLM augmentations. The kernel only touches the
    9th component (`IDX_PSI = 8`); any other NC ≥ 9 is fine.
    """
    var num_total_nodes = q.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var layout = row_major(Idx(q.buf_len))
    var decay_rate = (c_h * c_h) / c_p_sq

    comptime kernel = _glm_damping_kernel[P, type_of(layout), type_of(layout)]
    ctx.enqueue_function[kernel](
        TileTensor(q.buf, layout),
        TileTensor(rhs.buf, layout),
        decay_rate, num_total_nodes,
        grid_dim=grid, block_dim=BLOCK,
    )
