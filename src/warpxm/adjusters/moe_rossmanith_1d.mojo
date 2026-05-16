# Moe-Rossmanith slope limiter for 1D DG fields — conserved-variable cut.
#
# Mirrors `dfem/variable_adjusters/limiter/kokkos/moe_rossmanith_kokkos.{h,cc}`
# from upstream, restricted to 1D and to conserved-variable bounds only.
# The full upstream limiter also checks primitive-variable bounds (so
# pressure positivity is enforced for Euler / MHD); we leave that as a
# follow-up — the conserved-only cut is enough to tame moderate shocks like
# Sod (8:1 density, 10:1 pressure) without NaNing out.
#
# Reference: Moe, Rossmanith, Seal, "A SIMPLE AND EFFECTIVE HIGH-ORDER
# SHOCK CAPTURING LIMITER FOR DISCONTINUOUS GALERKIN METHODS" (2015),
# https://arxiv.org/abs/1507.03024
#
# Algorithm (per element e, each conserved component c):
#
#     q_avg(e, c) = Σᵢ wᵢ · q(e, i, c) / 2           (GLL quadrature; weights sum to 2)
#     q_min(e, c) = minᵢ q(e, i, c)
#     q_max(e, c) = maxᵢ q(e, i, c)
#
#     upper(e, c) = max( q_avg(e, c) + α(dx), max_{n ∈ neighbors} q_avg(n, c) )   # eq 6.4
#     lower(e, c) = min( q_avg(e, c) - α(dx), min_{n ∈ neighbors} q_avg(n, c) )   # eq 6.5
#     y_M(e, c)   = (upper(e, c) - q_avg(e, c)) / (q_max(e, c) - q_avg(e, c) + ε)
#     y_m(e, c)   = (lower(e, c) - q_avg(e, c)) / (q_min(e, c) - q_avg(e, c) + ε)
#     φ(y)        = max( min( |y| / 1.1, 1 ), 0 )                                 # clip
#     θ_c         = min( φ(y_M(e, c)), φ(y_m(e, c)) )                              # eq 6.6
#     θ(e)        = min(1, min_c θ_c)                                              # eq 6.7
#
#     q_new(e, i, c) = q_avg(e, c) + θ(e) · (q_old(e, i, c) - q_avg(e, c))         # eq 6.8
#
# with α(dx) = α_coeff · dx^{1.5} a safety margin that disables the limiter
# in smooth-flow regions. Default α_coeff = 1000 (upstream default).

from std.collections import InlineArray
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.euler import EULER_NC
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs, fmin, fmax
from warpxm.core.types import Real, WXM_DTYPE, INDEX_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D, GHOST_LEFT, GHOST_RIGHT
from warpxm.variable.state_1d import DGState1D


# ---------------------------------------------------------------------------
# Helpers (host & device).
# ---------------------------------------------------------------------------

@always_inline
def _y_func(extremum: Real, q_avg: Real, element_extremum: Real) -> Real:
    return (extremum - q_avg) / (element_extremum - q_avg + Real(1.0e-8))


@always_inline
def _phi_func(y: Real) -> Real:
    return fmax(fmin(fabs(y) / Real(1.1), Real(1.0)), Real(0.0))


@always_inline
def _alpha_dx(alpha_coeff: Real, dx: Real) -> Real:
    # α(dx) = α_coeff · dx^{1.5}; we use `dx * sqrt(dx)` which lowers cleanly.
    return alpha_coeff * dx * (dx ** Real(0.5))


# ---------------------------------------------------------------------------
# Kernel A: compute per-element cell averages, mins, and maxes of every
# conserved component. One thread per element.
# ---------------------------------------------------------------------------

def _bounds_kernel_1d[
    P: Int,
    QLayout: TensorLayout, BoundsLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    q_avg: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_min: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_max: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    num_elements: Int,
):
    comptime num_nodes_per_elem = P + 1
    comptime NC = EULER_NC
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
        var min_val: Scalar[WXM_DTYPE] = rebind[Scalar[WXM_DTYPE]](
            q[(e * num_nodes_per_elem + 0) * NC + c]
        )
        var max_val: Scalar[WXM_DTYPE] = min_val
        comptime for i in range(num_nodes_per_elem):
            var v = rebind[Scalar[WXM_DTYPE]](
                q[(e * num_nodes_per_elem + i) * NC + c]
            )
            sum_val += weights[i] * v
            if v < min_val: min_val = v
            if v > max_val: max_val = v
        # GLL weights on [-1,1] sum to 2, so the average is sum / 2.
        q_avg[e * NC + c] = rebind[q_avg.ElementType](sum_val * Real(0.5))
        q_min[e * NC + c] = rebind[q_min.ElementType](min_val)
        q_max[e * NC + c] = rebind[q_max.ElementType](max_val)


# ---------------------------------------------------------------------------
# Kernel B: compute the per-element θ and apply limiting in place.
# Neighbors are derived directly from `elem_left_face` / `elem_right_face` —
# boundary elements use themselves as the missing neighbor (effectively
# disabling the limiter at the domain edge, which is the simplest safe
# default).
# ---------------------------------------------------------------------------

def _apply_kernel_1d[
    P: Int,
    QLayout: TensorLayout, BoundsLayout: TensorLayout, EFLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    q_avg: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_min: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    q_max: TileTensor[WXM_DTYPE, BoundsLayout, MutAnyOrigin],
    elem_left_face: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    elem_right_face: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_left: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    face_right: TileTensor[INDEX_DTYPE, EFLayout, MutAnyOrigin],
    alpha_coeff: Real,
    dx: Real,
    num_elements: Int,
):
    comptime num_nodes_per_elem = P + 1
    comptime NC = EULER_NC
    comptime assert q.flat_rank == 1
    comptime assert q_avg.flat_rank == 1
    comptime assert q_min.flat_rank == 1
    comptime assert q_max.flat_rank == 1
    comptime assert elem_left_face.flat_rank == 1
    comptime assert elem_right_face.flat_rank == 1
    comptime assert face_left.flat_rank == 1
    comptime assert face_right.flat_rank == 1

    var e = Int(global_idx.x)
    if e >= num_elements:
        return

    var fL = Int(rebind[Scalar[INDEX_DTYPE]](elem_left_face[e]))
    var fR = Int(rebind[Scalar[INDEX_DTYPE]](elem_right_face[e]))
    var le = Int(rebind[Scalar[INDEX_DTYPE]](face_left[fL]))
    var re = Int(rebind[Scalar[INDEX_DTYPE]](face_right[fR]))

    # Use this element as the "neighbor" when the actual neighbor is a ghost
    # — this disables the limiter at the boundary and avoids reaching into
    # uninitialized ghost cells.
    var nbr_lo = le if le >= 0 else e
    var nbr_hi = re if re >= 0 else e

    var alpha = _alpha_dx(alpha_coeff, dx)
    var theta: Scalar[WXM_DTYPE] = 1.0

    comptime for c in range(NC):
        var q_e = rebind[Scalar[WXM_DTYPE]](q_avg[e * NC + c])
        var q_lo = rebind[Scalar[WXM_DTYPE]](q_avg[nbr_lo * NC + c])
        var q_hi = rebind[Scalar[WXM_DTYPE]](q_avg[nbr_hi * NC + c])

        var nbr_min = fmin(q_lo, q_hi)
        var nbr_max = fmax(q_lo, q_hi)

        var upper = fmax(q_e + alpha, nbr_max)
        var lower = fmin(q_e - alpha, nbr_min)

        var qmin_c = rebind[Scalar[WXM_DTYPE]](q_min[e * NC + c])
        var qmax_c = rebind[Scalar[WXM_DTYPE]](q_max[e * NC + c])

        var y_M = _y_func(upper, q_e, qmax_c)
        var y_m = _y_func(lower, q_e, qmin_c)
        var theta_c = fmin(_phi_func(y_M), _phi_func(y_m))
        if theta_c < theta: theta = theta_c

    if theta < Scalar[WXM_DTYPE](1.0 - 1.0e-10):
        comptime for i in range(num_nodes_per_elem):
            comptime for c in range(NC):
                var idx = (e * num_nodes_per_elem + i) * NC + c
                var q_avg_c = rebind[Scalar[WXM_DTYPE]](q_avg[e * NC + c])
                var q_old = rebind[Scalar[WXM_DTYPE]](q[idx])
                q[idx] = rebind[q.ElementType](
                    q_avg_c + theta * (q_old - q_avg_c)
                )


# ---------------------------------------------------------------------------
# Host-side driver.
# ---------------------------------------------------------------------------

struct MoeRossmanith1D[P: Int]:
    """Device-resident Moe-Rossmanith slope limiter (conserved-only) for 1D
    DG fields with `EULER_NC` components."""

    var alpha_coeff: Real
    var num_elements: Int

    # Scratch buffers, allocated once.
    var q_avg_buf: DeviceBuffer[Real.dtype]
    var q_min_buf: DeviceBuffer[Real.dtype]
    var q_max_buf: DeviceBuffer[Real.dtype]

    def __init__(
        out self, var ctx: DeviceContext, num_elements: Int,
        alpha_coeff: Real = Real(1000.0),
    ) raises:
        self.alpha_coeff = alpha_coeff
        self.num_elements = num_elements
        var n = num_elements * EULER_NC
        self.q_avg_buf = ctx.enqueue_create_buffer[Real.dtype](n)
        self.q_min_buf = ctx.enqueue_create_buffer[Real.dtype](n)
        self.q_max_buf = ctx.enqueue_create_buffer[Real.dtype](n)

    def apply(
        mut self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, EULER_NC],
    ) raises:
        """Run the limiter in place on `q`."""
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)

        var q_layout = row_major(Idx(q.buf_len))
        var bounds_layout = row_major(Idx(self.num_elements * EULER_NC))
        var ef_layout = row_major(Idx(mesh.num_elements))
        var face_layout = row_major(Idx(mesh.num_faces))

        var q_view = TileTensor(q.buf, q_layout)
        var qa_view = TileTensor(self.q_avg_buf, bounds_layout)
        var qmn_view = TileTensor(self.q_min_buf, bounds_layout)
        var qmx_view = TileTensor(self.q_max_buf, bounds_layout)

        comptime k_bounds = _bounds_kernel_1d[
            Self.P, type_of(q_layout), type_of(bounds_layout),
        ]
        ctx.enqueue_function[k_bounds](
            q_view, qa_view, qmn_view, qmx_view, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )

        var elf_view = TileTensor(mesh.elem_left_face_buf, ef_layout)
        var erf_view = TileTensor(mesh.elem_right_face_buf, ef_layout)
        var fl_view = TileTensor(mesh.face_left_buf, face_layout)
        var fr_view = TileTensor(mesh.face_right_buf, face_layout)

        comptime k_apply = _apply_kernel_1d[
            Self.P, type_of(q_layout), type_of(bounds_layout), type_of(ef_layout),
        ]
        ctx.enqueue_function[k_apply](
            q_view, qa_view, qmn_view, qmx_view,
            elf_view, erf_view, fl_view, fr_view,
            self.alpha_coeff, mesh.dx, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )
