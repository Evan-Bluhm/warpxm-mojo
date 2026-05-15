# Zhang-Shu positivity-preserving limiter for 1D Euler.
#
# Runs *after* the Moe-Rossmanith conserved-variable limiter and reduces
# the deviation from the cell mean (linearly, with a per-element scalar
# θ ∈ [0,1]) until every GLL node satisfies ρ ≥ ρ_eps and P ≥ P_eps.
#
# Algorithm (Zhang-Shu 2010, eq 3.14–3.16):
#
#   Let ū = cell mean (GLL quadrature). For each node n:
#     u_n(θ) = ū + θ · (u_n - ū)
#
#   Step 1 (density):
#     If ρ(u_n) < ρ_eps, pick θ_n = (ρ̄ - ρ_eps) / (ρ̄ - ρ(u_n)).
#     Take θ₁ = min over all such nodes; apply.
#
#   Step 2 (pressure):
#     After step 1, ρ ≥ ρ_eps everywhere. For each node n where
#     P(u_n) < P_eps, find largest θ ∈ [0,1] with P(u_n(θ)) ≥ P_eps —
#     P is a quadratic in θ, so we use the closed-form root.
#     Take θ₂ = min over all such nodes; apply.
#
# The cell mean ū is *not* modified, so mass / momentum / energy are
# exactly conserved. We assume the cell mean itself has admissible
# state (ρ̄ ≥ ρ_eps and P̄ ≥ P_eps); if not, the limiter clamps to ū
# and there is nothing more to do — the simulation has crashed already.

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.euler_2d import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fmin, fmax, sqrt, fabs
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.variable.state_1d import DGState1D


@always_inline
def _pressure(
    rho: Scalar[WXM_DTYPE], mx: Scalar[WXM_DTYPE],
    my: Scalar[WXM_DTYPE], mz: Scalar[WXM_DTYPE],
    E: Scalar[WXM_DTYPE], gamma: Real,
) -> Scalar[WXM_DTYPE]:
    var ke = Real(0.5) * (mx * mx + my * my + mz * mz) / rho
    return (gamma - Real(1.0)) * (E - ke)


def _positivity_kernel_1d[
    P: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rho_eps: Real, p_eps: Real, gamma: Real,
    num_elements: Int,
):
    comptime P1 = P + 1
    comptime NC = EULER_NC
    comptime assert q.flat_rank == 1

    var e = Int(global_idx.x)
    if e >= num_elements:
        return

    var weights = NodalBasis1D[P].weights()

    # --- Cell averages of all 5 conserved variables -----------------------
    var rho_bar: Scalar[WXM_DTYPE] = 0.0
    var mx_bar: Scalar[WXM_DTYPE] = 0.0
    var my_bar: Scalar[WXM_DTYPE] = 0.0
    var mz_bar: Scalar[WXM_DTYPE] = 0.0
    var E_bar: Scalar[WXM_DTYPE] = 0.0
    comptime for i in range(P1):
        var base = (e * P1 + i) * NC
        var w = weights[i] * Real(0.5)  # GLL weights on [-1,1] sum to 2
        rho_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
        mx_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MX])
        my_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MY])
        mz_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MZ])
        E_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_E])

    # --- Step 1: density positivity ---------------------------------------
    var theta_rho: Scalar[WXM_DTYPE] = 1.0
    comptime for i in range(P1):
        var base = (e * P1 + i) * NC
        var rho_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
        if rho_n < Scalar[WXM_DTYPE](rho_eps):
            var denom = rho_bar - rho_n
            # rho_bar ≥ rho_eps assumed; denom > 0 since rho_n < rho_eps ≤ rho_bar.
            var t = (rho_bar - Scalar[WXM_DTYPE](rho_eps)) / denom
            if t < theta_rho: theta_rho = t

    if theta_rho < Scalar[WXM_DTYPE](1.0):
        if theta_rho < Scalar[WXM_DTYPE](0.0): theta_rho = 0.0
        comptime for i in range(P1):
            var base = (e * P1 + i) * NC
            var d_rho = rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO]) - rho_bar
            var d_mx = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MX]) - mx_bar
            var d_my = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MY]) - my_bar
            var d_mz = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MZ]) - mz_bar
            var d_E = rebind[Scalar[WXM_DTYPE]](q[base + IDX_E]) - E_bar
            q[base + IDX_RHO] = rebind[q.ElementType](rho_bar + theta_rho * d_rho)
            q[base + IDX_MX] = rebind[q.ElementType](mx_bar + theta_rho * d_mx)
            q[base + IDX_MY] = rebind[q.ElementType](my_bar + theta_rho * d_my)
            q[base + IDX_MZ] = rebind[q.ElementType](mz_bar + theta_rho * d_mz)
            q[base + IDX_E] = rebind[q.ElementType](E_bar + theta_rho * d_E)

    # --- Step 2: pressure positivity --------------------------------------
    # For each node, P(θ) = (γ-1)·[E(θ) - ½|m(θ)|² / ρ(θ)] is rational in θ.
    # P(θ) ≥ p_eps  iff  f(θ) = ρ(θ)·E(θ) - ½|m(θ)|² - ρ(θ)·p_eps/(γ-1) ≥ 0
    # which is a quadratic in θ since ρ(θ), m(θ), E(θ) are all linear.
    var theta_p: Scalar[WXM_DTYPE] = 1.0
    var c_e = p_eps / (gamma - Real(1.0))

    comptime for i in range(P1):
        var base = (e * P1 + i) * NC
        var rho_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
        var mx_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MX])
        var my_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MY])
        var mz_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MZ])
        var E_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_E])
        var P_n = _pressure(rho_n, mx_n, my_n, mz_n, E_n, gamma)
        if P_n < Scalar[WXM_DTYPE](p_eps):
            var d_rho = rho_n - rho_bar
            var d_mx = mx_n - mx_bar
            var d_my = my_n - my_bar
            var d_mz = mz_n - mz_bar
            var d_E = E_n - E_bar
            # f(θ) = A θ² + B θ + C  with f(0) ≥ 0 and f(1) < 0.
            var A = d_rho * d_E - Real(0.5) * (d_mx * d_mx + d_my * d_my + d_mz * d_mz)
            var B = (
                rho_bar * d_E + E_bar * d_rho
                - (mx_bar * d_mx + my_bar * d_my + mz_bar * d_mz)
                - c_e * d_rho
            )
            var C = (
                rho_bar * E_bar
                - Real(0.5) * (mx_bar * mx_bar + my_bar * my_bar + mz_bar * mz_bar)
                - c_e * rho_bar
            )
            # Smallest positive root.
            var disc = B * B - Real(4.0) * A * C
            if disc < Scalar[WXM_DTYPE](0.0): disc = 0.0
            var sq = sqrt(disc)
            var root_a = (-B - sq) / (Real(2.0) * A + Real(1.0e-30))
            var root_b = (-B + sq) / (Real(2.0) * A + Real(1.0e-30))
            # Pick smallest positive root in [0, 1].
            var t: Scalar[WXM_DTYPE] = 1.0
            if root_a > Scalar[WXM_DTYPE](0.0) and root_a < t: t = root_a
            if root_b > Scalar[WXM_DTYPE](0.0) and root_b < t: t = root_b
            if t < Scalar[WXM_DTYPE](0.0): t = 0.0
            if t < theta_p: theta_p = t

    if theta_p < Scalar[WXM_DTYPE](1.0):
        if theta_p < Scalar[WXM_DTYPE](0.0): theta_p = 0.0
        comptime for i in range(P1):
            var base = (e * P1 + i) * NC
            var d_rho = rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO]) - rho_bar
            var d_mx = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MX]) - mx_bar
            var d_my = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MY]) - my_bar
            var d_mz = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MZ]) - mz_bar
            var d_E = rebind[Scalar[WXM_DTYPE]](q[base + IDX_E]) - E_bar
            q[base + IDX_RHO] = rebind[q.ElementType](rho_bar + theta_p * d_rho)
            q[base + IDX_MX] = rebind[q.ElementType](mx_bar + theta_p * d_mx)
            q[base + IDX_MY] = rebind[q.ElementType](my_bar + theta_p * d_my)
            q[base + IDX_MZ] = rebind[q.ElementType](mz_bar + theta_p * d_mz)
            q[base + IDX_E] = rebind[q.ElementType](E_bar + theta_p * d_E)


struct PositivityEuler1D[P: Int]:
    """Device-resident Zhang-Shu positivity preserver for 1D Euler.

    Reduces each element's GLL-node deviation from the cell mean until
    ρ ≥ rho_eps and P ≥ p_eps. Conservative: cell averages are untouched.
    """

    var rho_eps: Real
    var p_eps: Real
    var gamma: Real
    var num_elements: Int

    def __init__(
        out self, num_elements: Int,
        gamma: Real,
        rho_eps: Real = Real(1.0e-10),
        p_eps: Real = Real(1.0e-10),
    ):
        self.rho_eps = rho_eps
        self.p_eps = p_eps
        self.gamma = gamma
        self.num_elements = num_elements

    def apply(
        mut self,
        var ctx: DeviceContext,
        mut mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, EULER_NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)
        var q_layout = row_major(Idx(q.buf_len))
        var q_view = TileTensor(q.buf, q_layout)
        comptime k_fn = _positivity_kernel_1d[Self.P, type_of(q_layout)]
        ctx.enqueue_function[k_fn](
            q_view, self.rho_eps, self.p_eps, self.gamma, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )
