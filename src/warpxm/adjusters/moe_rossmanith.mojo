# Moe-Rossmanith slope limiter for 2D + 3D DG fields — NDIM-parametric.
#
# Consolidates `moe_rossmanith_2d.mojo` and `moe_rossmanith_3d.mojo`.
# The 1D variant uses a different mesh API (separate left/right face
# arrays and Euler-only NC) and is kept separate.
#
# Algorithm: per element compute cell average + conserved-variable
# bounds (min, max) via GLL quadrature; gather all 2·NDIM face
# neighbors' averages; compute a single per-element θ ∈ [0, 1] using
# the Moe-Rossmanith bound-collapse + φ functions; linearly shrink each
# node's deviation from the cell mean by θ. Cell means are untouched.
#
# Parametric on the conserved-variable count NC so one kernel handles
# Euler (NC=5), IMHD (NC=8), GLM-MHD (NC=9), …

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs, fmin, fmax
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_2d import DGState2D
from warpxm.variable.state_3d import DGState3D


@always_inline
def _y_func(extremum: Real, q_avg: Real, element_extremum: Real) -> Real:
    return (extremum - q_avg) / (element_extremum - q_avg + Real(1.0e-8))


@always_inline
def _phi_func(y: Real) -> Real:
    return fmax(fmin(fabs(y) / Real(1.1), Real(1.0)), Real(0.0))


@always_inline
def _alpha_dx(alpha_coeff: Real, dx: Real) -> Real:
    return alpha_coeff * dx * (dx ** Real(0.5))


@always_inline
def _gll_cell_weight[P: Int, NDIM: Int](n: Int) -> Scalar[WXM_DTYPE]:
    """Tensor-product GLL cell-mean weight at flat node `n`, normalised
    by 1 / 2^NDIM (reference [-1, 1]^NDIM → unit-cell)."""
    comptime P1 = P + 1
    var weights = NodalBasis1D[P].weights()
    comptime if NDIM == 2:
        var jn = n // P1
        var in_ = n % P1
        return weights[in_] * weights[jn] * Real(0.25)
    # NDIM == 3
    var kn = n // (P1 * P1)
    var rem = n % (P1 * P1)
    var jn = rem // P1
    var in_ = rem % P1
    return weights[in_] * weights[jn] * weights[kn] * Real(0.125)


def _bounds_kernel[
    P: Int, NDIM: Int, NC: Int,
    QLayout: TensorLayout, BoundsLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    q_avg: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_min: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_max: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    num_elements: Int,
):
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 ** NDIM
    comptime assert q.flat_rank == 1
    comptime assert q_avg.flat_rank == 1
    comptime assert q_min.flat_rank == 1
    comptime assert q_max.flat_rank == 1

    var e = Int(global_idx.x)
    if e >= num_elements:
        return

    comptime for c in range(NC):
        var sum_val: Scalar[WXM_DTYPE] = 0.0
        var min_val: Scalar[WXM_DTYPE] = rebind[Scalar[WXM_DTYPE]](
            q[(e * nodes_per_elem + 0) * NC + c]
        )
        var max_val: Scalar[WXM_DTYPE] = min_val
        comptime for n in range(nodes_per_elem):
            var w = _gll_cell_weight[P, NDIM](n)
            var v = rebind[Scalar[WXM_DTYPE]](
                q[(e * nodes_per_elem + n) * NC + c]
            )
            sum_val += w * v
            if v < min_val:
                min_val = v
            if v > max_val:
                max_val = v
        q_avg[e * NC + c] = rebind[q_avg.ElementType](sum_val)
        q_min[e * NC + c] = rebind[q_min.ElementType](min_val)
        q_max[e * NC + c] = rebind[q_max.ElementType](max_val)


def _apply_kernel[
    P: Int, NDIM: Int, NC: Int,
    QLayout: TensorLayout, BoundsLayout: TensorLayout,
    FaceLayout: TensorLayout, EFLayout: TensorLayout,
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
    """Gather 2*NDIM face neighbors of element e, compute per-element θ,
    apply linear deviation shrink. Face direction d ∈ [0, 2*NDIM) maps
    even/odd → face_neg_elem / face_pos_elem (matches the BlockMesh
    convention: FACE_DIR_LEFT = 0, FACE_DIR_RIGHT = 1, FACE_DIR_BOTTOM
    = 2, FACE_DIR_TOP = 3, FACE_DIR_DOWN = 4, FACE_DIR_UP = 5).
    """
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 ** NDIM
    comptime num_neighbors = 2 * NDIM
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

    var nbrs: InlineArray[Int, num_neighbors] = InlineArray[Int, num_neighbors](fill=0)
    comptime for d in range(num_neighbors):
        var f_id = Int(rebind[Scalar[INDEX_DTYPE]](
            elem_face_dir[e * num_neighbors + d]
        ))
        var nbr: Int
        comptime if d % 2 == 0:
            nbr = Int(rebind[Scalar[INDEX_DTYPE]](face_neg_elem[f_id]))
        else:
            nbr = Int(rebind[Scalar[INDEX_DTYPE]](face_pos_elem[f_id]))
        if nbr < 0:
            nbr = e
        nbrs[d] = nbr

    var alpha = _alpha_dx(alpha_coeff, dx_min)
    var theta: Scalar[WXM_DTYPE] = 1.0

    comptime for c in range(NC):
        var q_e = rebind[Scalar[WXM_DTYPE]](q_avg[e * NC + c])
        var nbr_min: Scalar[WXM_DTYPE] = rebind[Scalar[WXM_DTYPE]](
            q_avg[nbrs[0] * NC + c]
        )
        var nbr_max = nbr_min
        comptime for d in range(1, num_neighbors):
            var v = rebind[Scalar[WXM_DTYPE]](q_avg[nbrs[d] * NC + c])
            if v < nbr_min: nbr_min = v
            if v > nbr_max: nbr_max = v

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
                q[idx] = rebind[q.ElementType](
                    q_avg_c + theta * (q_old - q_avg_c)
                )


struct MoeRossmanith2D[P: Int, NC: Int]:
    """Device-resident Moe-Rossmanith slope limiter for 2D DG fields."""

    var alpha_coeff: Real
    var num_elements: Int
    var q_avg_buf: DeviceBuffer[Real.dtype]
    var q_min_buf: DeviceBuffer[Real.dtype]
    var q_max_buf: DeviceBuffer[Real.dtype]

    def __init__(
        out self, var ctx: DeviceContext, num_elements: Int,
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

        comptime k_bounds = _bounds_kernel[
            Self.P, 2, Self.NC,
            type_of(q_layout), type_of(bounds_layout),
        ]
        ctx.enqueue_function[k_bounds](
            q_view, qa_view, qmn_view, qmx_view, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )

        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var dx_min = fmin(mesh.dx, mesh.dy)

        comptime k_apply = _apply_kernel[
            Self.P, 2, Self.NC,
            type_of(q_layout), type_of(bounds_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        ctx.enqueue_function[k_apply](
            q_view, qa_view, qmn_view, qmx_view,
            fn_view, fp_view, ef_view,
            self.alpha_coeff, dx_min, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )


struct MoeRossmanith3D[P: Int, NC: Int]:
    """Device-resident Moe-Rossmanith slope limiter for 3D DG fields."""

    var alpha_coeff: Real
    var num_elements: Int
    var q_avg_buf: DeviceBuffer[Real.dtype]
    var q_min_buf: DeviceBuffer[Real.dtype]
    var q_max_buf: DeviceBuffer[Real.dtype]

    def __init__(
        out self, var ctx: DeviceContext, num_elements: Int,
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
        mut mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, Self.NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)

        var q_layout = row_major(Idx(q.buf_len))
        var bounds_layout = row_major(Idx(self.num_elements * Self.NC))
        var face_layout = row_major(Idx(mesh.num_faces))
        var ef_layout = row_major(Idx(mesh.num_elements * 6))

        var q_view = TileTensor(q.buf, q_layout)
        var qa_view = TileTensor(self.q_avg_buf, bounds_layout)
        var qmn_view = TileTensor(self.q_min_buf, bounds_layout)
        var qmx_view = TileTensor(self.q_max_buf, bounds_layout)

        comptime k_bounds = _bounds_kernel[
            Self.P, 3, Self.NC,
            type_of(q_layout), type_of(bounds_layout),
        ]
        ctx.enqueue_function[k_bounds](
            q_view, qa_view, qmn_view, qmx_view, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )

        var fn_view = TileTensor(mesh.face_neg_elem_buf, face_layout)
        var fp_view = TileTensor(mesh.face_pos_elem_buf, face_layout)
        var ef_view = TileTensor(mesh.elem_face_dir_buf, ef_layout)
        var dx_min = fmin(fmin(mesh.dx, mesh.dy), mesh.dz)

        comptime k_apply = _apply_kernel[
            Self.P, 3, Self.NC,
            type_of(q_layout), type_of(bounds_layout),
            type_of(face_layout), type_of(ef_layout),
        ]
        ctx.enqueue_function[k_apply](
            q_view, qa_view, qmn_view, qmx_view,
            fn_view, fp_view, ef_view,
            self.alpha_coeff, dx_min, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )
