# Parabolic damping source for the GLM ψ field.
#
# The Dedner equation is `∂ψ/∂t + c_h² · ∇·B = -(c_h² / c_p²) · ψ`. The
# hyperbolic LHS lives in `dg_imhd_glm.mojo`'s flux tensor; this kernel
# applies the parabolic decay term post-spatial-RHS as a per-node update.
#
# With `c_p²` ~ 0.18·c_h·L_domain (Mignone & Tzeferacos 2010) the e-folding
# time is τ = c_p²/c_h² ~ 0.18·L_domain/c_h, i.e. a few crossing times.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.imhd_glm import MHD_GLM_NC, IDX_PSI
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.variable.state import DGState


def _glm_damping_kernel[
    P: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    decay_rate: Real,
    num_total_nodes_i64: Int64,
):
    """rhs[ψ] += -decay_rate · q[ψ], where decay_rate = c_h² / c_p²."""
    var num_total_nodes = Int(num_total_nodes_i64)
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime NC = MHD_GLM_NC

    var n = Int(global_idx.x)
    if n >= num_total_nodes:
        return
    var base = n * NC

    var psi = rebind[Scalar[WXM_DTYPE]](q[base + IDX_PSI])
    var r = rebind[Scalar[WXM_DTYPE]](rhs[base + IDX_PSI])
    rhs[base + IDX_PSI] = rebind[rhs.ElementType](r - decay_rate * psi)


def apply_glm_damping[NDIM: Int, P: Int, NC: Int](
    var ctx: DeviceContext,
    mut q: DGState[NDIM, P, NC],
    mut rhs: DGState[NDIM, P, NC],
    c_h: Real,
    c_p_sq: Real,
) raises:
    """Apply `∂ψ/∂t += -(c_h² / c_p²) · ψ` in place on `rhs`."""
    var num_total_nodes = q.total_nodes
    comptime BLOCK = 256
    var grid = ceildiv(num_total_nodes, BLOCK)
    var layout = row_major(Int(q.buf_len))
    var decay_rate = (c_h * c_h) / c_p_sq

    comptime kernel = _glm_damping_kernel[P, type_of(layout), type_of(layout)]
    ctx.enqueue_function[kernel](
        TileTensor(q.buf, layout),
        TileTensor(rhs.buf, layout),
        decay_rate, Int64(num_total_nodes),
        grid_dim=grid, block_dim=BLOCK,
    )
