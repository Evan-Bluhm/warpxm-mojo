# Moe-Rossmanith slope limiter for 2D DG fields — conserved-variable cut.
#
# Tensor-product extension of `adjusters/moe_rossmanith_1d.mojo`. Same
# algorithm; neighbors come from all four face directions (left / right /
# bottom / top). Cell averages are tensor-product GLL quadrature:
#
#     q_avg(e, c) = (1/4) · Σᵢ Σⱼ wᵢ wⱼ · q(e, i, j, c)
#
# Each element of a 2D block mesh has at most 4 neighbors; we collapse
# their bounds the same way as 1D. Boundary elements use themselves as the
# missing neighbor.
#
# Parameterised on the conserved-variable count `NC` so the same kernel
# handles Euler (NC=5), IMHD (NC=8), or any other multi-component
# conservative system.

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs, fmin, fmax
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import (
    BlockMesh2D,
    GHOST_NEG,
    GHOST_POS,
    FACE_DIR_LEFT,
    FACE_DIR_RIGHT,
    FACE_DIR_BOTTOM,
    FACE_DIR_TOP,
)
from warpxm.variable.state_2d import DGState2D


@always_inline
def _y_func(extremum: Real, q_avg: Real, element_extremum: Real) -> Real:
    return (extremum - q_avg) / (element_extremum - q_avg + Real(1.0e-8))


@always_inline
def _phi_func(y: Real) -> Real:
    return fmax(fmin(fabs(y) / Real(1.1), Real(1.0)), Real(0.0))


@always_inline
def _alpha_dx(alpha_coeff: Real, dx: Real) -> Real:
    return alpha_coeff * dx * (dx ** Real(0.5))


# ---------------------------------------------------------------------------
# Kernel A: per-element bounds in 2D.
# ---------------------------------------------------------------------------


def _bounds_kernel_2d[
    P: Int,
    NC: Int,
    QLayout: TensorLayout,
    BoundsLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    q_avg: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_min: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_max: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    num_elements: Int,
):
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1
    comptime assert q.flat_rank == 1
    comptime assert q_avg.flat_rank == 1
    comptime assert q_min.flat_rank == 1
    comptime assert q_max.flat_rank == 1

    var e = Int(global_idx.x)
    if e >= num_elements:
        return

    var weights = NodalBasis1D[P].weights()

    comptime for c in range(NC):
        var sum_val: Scalar[WXM_DTYPE] = 0.0
        var min_val: Scalar[WXM_DTYPE] = rebind[Scalar[WXM_DTYPE]](q[(e * nodes_per_elem + 0) * NC + c])
        var max_val: Scalar[WXM_DTYPE] = min_val
        comptime for jn in range(P1):
            comptime for in_ in range(P1):
                var n = jn * P1 + in_
                var v = rebind[Scalar[WXM_DTYPE]](q[(e * nodes_per_elem + n) * NC + c])
                sum_val += weights[in_] * weights[jn] * v
                if v < min_val:
                    min_val = v
                if v > max_val:
                    max_val = v
        q_avg[e * NC + c] = rebind[q_avg.ElementType](sum_val * Real(0.25))
        q_min[e * NC + c] = rebind[q_min.ElementType](min_val)
        q_max[e * NC + c] = rebind[q_max.ElementType](max_val)


# ---------------------------------------------------------------------------
# Kernel B: compute θ and apply.
# ---------------------------------------------------------------------------


def _apply_kernel_2d[
    P: Int,
    NC: Int,
    QLayout: TensorLayout,
    BoundsLayout: TensorLayout,
    FaceLayout: TensorLayout,
    EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    q_avg: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_min: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_max: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    face_neg_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    face_pos_elem: TileTensor[INDEX_DTYPE, FaceLayout, MutAnyOrigin],
    elem_face_dir: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    alpha_coeff: Real,
    dx_min: Real,
    num_elements: Int,
):
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1
    comptime assert q.flat_rank == 1
    comptime assert q_avg.flat_rank == 1
    comptime assert q_min.flat_rank == 1
    comptime assert q_max.flat_rank == 1
    comptime assert face_neg_elem.flat_rank == 1
    comptime assert face_pos_elem.flat_rank == 1
    comptime assert elem_face_dir.flat_rank == 1

    var e = Int(global_idx.x)
    if e >= num_elements:
        return

    var fL = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 4 + FACE_DIR_LEFT]))
    var fR = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 4 + FACE_DIR_RIGHT]))
    var fB = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 4 + FACE_DIR_BOTTOM]))
    var fT = Int(rebind[Scalar[INDEX_DTYPE]](elem_face_dir[e * 4 + FACE_DIR_TOP]))

    var nbr_L = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fL]))
    var nbr_R = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fR]))
    var nbr_B = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[fB]))
    var nbr_T = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[fT]))

    if nbr_L < 0:
        nbr_L = e
    if nbr_R < 0:
        nbr_R = e
    if nbr_B < 0:
        nbr_B = e
    if nbr_T < 0:
        nbr_T = e

    var alpha = _alpha_dx(alpha_coeff, dx_min)
    var theta: Scalar[WXM_DTYPE] = 1.0

    comptime for c in range(NC):
        var q_e = rebind[Scalar[WXM_DTYPE]](q_avg[e * NC + c])
        var nbr1 = rebind[Scalar[WXM_DTYPE]](q_avg[nbr_L * NC + c])
        var nbr2 = rebind[Scalar[WXM_DTYPE]](q_avg[nbr_R * NC + c])
        var nbr3 = rebind[Scalar[WXM_DTYPE]](q_avg[nbr_B * NC + c])
        var nbr4 = rebind[Scalar[WXM_DTYPE]](q_avg[nbr_T * NC + c])

        var nbr_min = fmin(fmin(nbr1, nbr2), fmin(nbr3, nbr4))
        var nbr_max = fmax(fmax(nbr1, nbr2), fmax(nbr3, nbr4))

        var upper = fmax(q_e + alpha, nbr_max)
        var lower = fmin(q_e - alpha, nbr_min)

        var qmin_c = rebind[Scalar[WXM_DTYPE]](q_min[e * NC + c])
        var qmax_c = rebind[Scalar[WXM_DTYPE]](q_max[e * NC + c])

        var y_M = _y_func(upper, q_e, qmax_c)
        var y_m = _y_func(lower, q_e, qmin_c)
        var theta_c = fmin(_phi_func(y_M), _phi_func(y_m))
        if theta_c < theta:
            theta = theta_c

    if theta < Scalar[WXM_DTYPE](1.0 - 1.0e-10):
        comptime for n in range(nodes_per_elem):
            comptime for c in range(NC):
                var idx = (e * nodes_per_elem + n) * NC + c
                var q_avg_c = rebind[Scalar[WXM_DTYPE]](q_avg[e * NC + c])
                var q_old = rebind[Scalar[WXM_DTYPE]](q[idx])
                q[idx] = rebind[q.ElementType](q_avg_c + theta * (q_old - q_avg_c))


# ---------------------------------------------------------------------------
# Host-side driver.
# ---------------------------------------------------------------------------


struct MoeRossmanith2D[P: Int, NC: Int]:
    """Device-resident Moe-Rossmanith slope limiter (conserved-only) for 2D
    DG fields with `NC` components."""

    var alpha_coeff: Real
    var num_elements: Int

    var q_avg_buf: DeviceBuffer[Real.dtype]
    var q_min_buf: DeviceBuffer[Real.dtype]
    var q_max_buf: DeviceBuffer[Real.dtype]

    def __init__(
        out self,
        var ctx: DeviceContext,
        num_elements: Int,
        alpha_coeff: Real = Real(1000.0),
    ) raises:
        self.alpha_coeff = alpha_coeff
        self.num_elements = num_elements
        var n = num_elements * Self.NC
        self.q_avg_buf = ctx.enqueue_create_buffer[Real.dtype](n)
        self.q_min_buf = ctx.enqueue_create_buffer[Real.dtype](n)
        self.q_max_buf = ctx.enqueue_create_buffer[Real.dtype](n)

    def apply(
        mut self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh2D[Self.P],
        mut q: DGState2D[Self.P, Self.NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)

        var q_layout = row_major(Idx(q.buf_len))
        var bounds_layout = row_major(Idx(self.num_elements * Self.NC))
        var face_layout = row_major(Idx(mesh.num_faces))
        var ef_layout = row_major(Idx(mesh.num_elements * 4))

        var q_view = TileTensor(q.buf, q_layout)
        var qa_view = TileTensor(self.q_avg_buf, bounds_layout)
        var qmn_view = TileTensor(self.q_min_buf, bounds_layout)
        var qmx_view = TileTensor(self.q_max_buf, bounds_layout)

        comptime k_bounds = _bounds_kernel_2d[
            Self.P,
            Self.NC,
            type_of(q_layout),
            type_of(bounds_layout),
        ]
        ctx.enqueue_function[k_bounds](
            q_view,
            qa_view,
            qmn_view,
            qmx_view,
            self.num_elements,
            grid_dim=grid,
            block_dim=BLOCK,
        )

        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var dx_min = fmin(mesh.dx, mesh.dy)

        comptime k_apply = _apply_kernel_2d[
            Self.P,
            Self.NC,
            type_of(q_layout),
            type_of(bounds_layout),
            type_of(face_layout),
            type_of(ef_layout),
        ]
        ctx.enqueue_function[k_apply](
            q_view,
            qa_view,
            qmn_view,
            qmx_view,
            fn_view,
            fp_view,
            ef_view,
            self.alpha_coeff,
            dx_min,
            self.num_elements,
            grid_dim=grid,
            block_dim=BLOCK,
        )
