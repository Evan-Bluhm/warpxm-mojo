# Vlasov BGK collision source for 1D1V.
#
# Mirrors upstream `wmapplication_bgk_collision_source.{h,cc}`.
# Implements the Bhatnagar-Gross-Krook relaxation operator
#
#     ∂f/∂t |_coll = -(f - f_M) / τ,
#
# where `f_M(x, v)` is the local Maxwellian carrying the same moments
# `(n, u, T)` as the current distribution at each x-node:
#
#     n(x)  = ∫ f dv          (zeroth moment, `ρ`)
#     u(x)  = ∫ v f dv / n    (bulk velocity, from first moment `J`)
#     T(x)  = (2 · E_kin / n) - u²
#             where E_kin(x) = ½ ∫ v² f dv (second moment)
#     f_M(x, v) = n / √(2π T) · exp( -(v - u)² / (2T) ).
#
# The kernel is a pure per-phase-node local computation (no neighbour
# loads, no diff matrix, no surface fluxes), so it parallelises one
# thread per phase node. The driver is responsible for pre-computing
# the three moments before launch — that's encapsulated by
# `VlasovBGKSource1D1V.apply()` which holds the scratch buffers and
# the `VlasovMoments1D1V` helper.

from std.gpu import block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import exp
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.vlasovmaxwell import VLASOV_NC
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, WXM_DTYPE, PI
from warpxm.mesh.phase_mesh_1d1v import PhaseMesh1D1V
from warpxm.spatial.vlasov_moments_1d1v import VlasovMoments1D1V
from warpxm.variable.phase_state_1d1v import PhaseState1D1V
from warpxm.variable.state_1d import DGState1D


# Floor on the reconstructed temperature, used when the
# instantaneous moments give `T ≤ 0` (numerical noise; physically
# impossible for a real distribution). Keeps `√(2π T)` finite and
# `exp(...)` well defined.
comptime T_MIN: Real = 1.0e-12


def _bgk_source_kernel_1d1v[
    P_x: Int, P_v: Int,
    QLayout: TensorLayout, RhsLayout: TensorLayout,
    RhoLayout: TensorLayout, JLayout: TensorLayout, ELayout: TensorLayout,
    NodeVLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rhs: TileTensor[WXM_DTYPE, RhsLayout, MutAnyOrigin],
    rho: TileTensor[WXM_DTYPE, RhoLayout, MutAnyOrigin],
    j: TileTensor[WXM_DTYPE, JLayout, MutAnyOrigin],
    e_kin: TileTensor[WXM_DTYPE, ELayout, MutAnyOrigin],
    node_v: TileTensor[WXM_DTYPE, NodeVLayout, MutAnyOrigin],
    num_v: Int,
    inv_tau: Real,
    num_phase_elements: Int,
):
    """Per-phase-node BGK source. Adds `-(f - f_M) / τ` to `rhs` at
    every phase node.

    Thread layout: `block_idx.x = phase_e`,
    `(thread_idx.x, thread_idx.y) = (ix, iv)`."""
    comptime P1x = P_x + 1
    comptime P1v = P_v + 1
    comptime nodes_per_phase = P1x * P1v
    comptime NC = VLASOV_NC
    comptime assert q.flat_rank == 1
    comptime assert rhs.flat_rank == 1
    comptime assert rho.flat_rank == 1
    comptime assert j.flat_rank == 1
    comptime assert e_kin.flat_rank == 1
    comptime assert node_v.flat_rank == 1

    var phase_e = Int(block_idx.x)
    var ix = Int(thread_idx.x)
    var iv = Int(thread_idx.y)
    if phase_e >= num_phase_elements or ix >= P1x or iv >= P1v:
        return

    var ex = phase_e // num_v
    var ev = phase_e % num_v
    var local_n = iv * P1x + ix

    # Per-x-node moments.
    var node_x = ex * P1x + ix
    var n_val: Real = rebind[Scalar[WXM_DTYPE]](rho[node_x])
    var j_val: Real = rebind[Scalar[WXM_DTYPE]](j[node_x])
    var e_val: Real = rebind[Scalar[WXM_DTYPE]](e_kin[node_x])

    # Bulk velocity + temperature reconstruction. Guard `n_val` against
    # exact zero (the kernel runs on every x-node — even those where
    # the distribution happens to be empty); fall back to a delta-like
    # Maxwellian width to keep arithmetic finite.
    var n_safe = n_val if n_val > T_MIN else T_MIN
    var u = j_val / n_safe
    var t_raw = Real(2.0) * e_val / n_safe - u * u
    var t = t_raw if t_raw > T_MIN else T_MIN

    # Local Maxwellian at this phase node.
    var v_iv: Real = rebind[Scalar[WXM_DTYPE]](node_v[ev * P1v + iv])
    var dv_shift = v_iv - u
    var f_m = n_val / sqrt(Real(2.0) * PI * t) * exp(
        -dv_shift * dv_shift / (Real(2.0) * t)
    )

    # Current f at this phase node.
    var base = (phase_e * nodes_per_phase + local_n) * NC
    var f_curr: Real = rebind[Scalar[WXM_DTYPE]](q[base])

    var src = (f_m - f_curr) * inv_tau
    var r_curr: Real = rebind[Scalar[WXM_DTYPE]](rhs[base])
    rhs[base] = rebind[rhs.ElementType](r_curr + src)


struct VlasovBGKSource1D1V[P_x: Int, P_v: Int]:
    """Host-side BGK collision source launcher. Owns scratch buffers
    for the three velocity moments and a `VlasovMoments1D1V`
    helper."""

    var moments: VlasovMoments1D1V[Self.P_x, Self.P_v]
    var rho: DGState1D[Self.P_x, 1]
    var j: DGState1D[Self.P_x, 1]
    var e_kin: DGState1D[Self.P_x, 1]

    def __init__(out self, var ctx: DeviceContext, num_x: Int) raises:
        self.moments = VlasovMoments1D1V[Self.P_x, Self.P_v]()
        self.rho = DGState1D[Self.P_x, 1](ctx, num_x)
        self.j = DGState1D[Self.P_x, 1](ctx, num_x)
        self.e_kin = DGState1D[Self.P_x, 1](ctx, num_x)

    def apply(
        mut self,
        var ctx: DeviceContext,
        mut mesh: PhaseMesh1D1V[Self.P_x, Self.P_v],
        mut q: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
        mut rhs: PhaseState1D1V[Self.P_x, Self.P_v, VLASOV_NC],
        tau: Real,
    ) raises:
        """Compute the three moments of `q`, reconstruct the local
        Maxwellian per phase node, and add `-(f - f_M) / τ` to
        `rhs`. Call this AFTER the Vlasov-Maxwell `compute_rhs` so
        the BGK source accumulates into the existing residual."""
        self.moments.zeroth_moment(ctx, mesh, q, self.rho)
        self.moments.first_moment(ctx, mesh, q, self.j)
        self.moments.second_moment(ctx, mesh, q, self.e_kin)

        comptime P1x = Self.P_x + 1
        comptime P1v = Self.P_v + 1
        var q_layout = row_major(Idx(q.buf_len))
        var rho_layout = row_major(Idx(self.rho.buf_len))
        var node_v_layout = row_major(Idx(mesh.num_v * P1v))

        var q_view = TileTensor(q.buf, q_layout)
        var rhs_view = TileTensor(rhs.buf, q_layout)
        var rho_view = TileTensor(self.rho.buf, rho_layout)
        var j_view = TileTensor(self.j.buf, rho_layout)
        var e_view = TileTensor(self.e_kin.buf, rho_layout)
        var node_v_view = TileTensor(mesh.node_v_buf, node_v_layout)

        comptime kernel = _bgk_source_kernel_1d1v[
            Self.P_x, Self.P_v,
            type_of(q_layout), type_of(q_layout),
            type_of(rho_layout), type_of(rho_layout), type_of(rho_layout),
            type_of(node_v_layout),
        ]
        ctx.enqueue_function[kernel](
            q_view, rhs_view, rho_view, j_view, e_view, node_v_view,
            mesh.num_v, Real(1.0) / tau, mesh.num_phase_elements,
            grid_dim=mesh.num_phase_elements,
            block_dim=(P1x, P1v),
        )
