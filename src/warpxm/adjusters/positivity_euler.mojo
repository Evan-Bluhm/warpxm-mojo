# Zhang-Shu positivity-preserving limiter for 5-moment Euler, generic
# over the dimension.
#
# Runs after the Moe-Rossmanith conserved-variable limiter and reduces
# each element's deviation from the cell mean (linearly, with a per-
# element scalar θ ∈ [0, 1]) until every GLL node satisfies ρ ≥ ρ_eps
# and P ≥ P_eps. Cell averages are untouched, so mass / momentum /
# energy stay exactly conserved.
#
# Per Zhang-Shu (2010, §3.14–3.16):
#   ū = cell mean. For each node n:  u_n(θ) = ū + θ·(u_n − ū).
#   Step 1 (density): pick θ_n so ρ(u_n(θ_n)) = ρ_eps; θ_ρ = min over n.
#   Step 2 (pressure): P is quadratic in θ — closed-form root.
#
# The kernel runs flat over the `(P+1)^NDIM` nodes of an element; the only
# dimension-specific piece is the cell-mean weight (`adjusters/cell_mean.mojo`).

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.adjusters.cell_mean import gll_cell_weight
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.block import BlockMesh
from warpxm.variable.state import DGState


@always_inline
def _pressure(
    rho: Scalar[WXM_DTYPE], mx: Scalar[WXM_DTYPE],
    my: Scalar[WXM_DTYPE], mz: Scalar[WXM_DTYPE],
    E: Scalar[WXM_DTYPE], gamma: Real,
) -> Scalar[WXM_DTYPE]:
    var ke = Real(0.5) * (mx * mx + my * my + mz * mz) / rho
    return (gamma - Real(1.0)) * (E - ke)


def _positivity_kernel_euler[
    P: Int, NDIM: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rho_eps: Real, p_eps: Real, gamma: Real,
    num_elements_i64: Int64,
):
    var num_elements = Int(num_elements_i64)
    comptime P1 = P + 1
    comptime NC = EULER_NC
    comptime nodes_per_elem = P1 ** NDIM
    comptime assert q.flat_rank == 1
    comptime assert NDIM >= 1 and NDIM <= 3, "NDIM must be 1, 2, or 3"

    var e = Int(global_idx.x)
    if e >= num_elements:
        return

    # --- Cell averages of all 5 conserved variables -----------------------
    var rho_bar: Scalar[WXM_DTYPE] = 0.0
    var mx_bar: Scalar[WXM_DTYPE] = 0.0
    var my_bar: Scalar[WXM_DTYPE] = 0.0
    var mz_bar: Scalar[WXM_DTYPE] = 0.0
    var E_bar: Scalar[WXM_DTYPE] = 0.0
    comptime for n in range(nodes_per_elem):
        var base = (e * nodes_per_elem + n) * NC
        var w = gll_cell_weight[P, NDIM](n)
        rho_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
        mx_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MX])
        my_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MY])
        mz_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MZ])
        E_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_E])

    # --- Step 1: density positivity ---------------------------------------
    var theta_rho: Scalar[WXM_DTYPE] = 1.0
    comptime for n in range(nodes_per_elem):
        var base = (e * nodes_per_elem + n) * NC
        var rho_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
        if rho_n < Scalar[WXM_DTYPE](rho_eps):
            var denom = rho_bar - rho_n
            var t = (rho_bar - Scalar[WXM_DTYPE](rho_eps)) / denom
            if t < theta_rho: theta_rho = t

    if theta_rho < Scalar[WXM_DTYPE](1.0):
        if theta_rho < Scalar[WXM_DTYPE](0.0): theta_rho = 0.0
        comptime for n in range(nodes_per_elem):
            var base = (e * nodes_per_elem + n) * NC
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
    var theta_p: Scalar[WXM_DTYPE] = 1.0
    var c_e = p_eps / (gamma - Real(1.0))

    comptime for n in range(nodes_per_elem):
        var base = (e * nodes_per_elem + n) * NC
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
            # g(θ) = A θ² + B θ + C is ρ·(ρe − c_e) along the segment from
            # the mean (θ = 0) to the node (θ = 1); we want its smallest
            # root in (0, 1). The textbook (−B ± √disc) / 2A cancels
            # catastrophically when |A C| ≪ B², which is the usual case
            # (small deviation about the mean), so use the stable pair
            # q / A and C / q with q = −(B + sign(B) √disc) / 2.
            var t: Scalar[WXM_DTYPE] = 1.0
            if C <= Scalar[WXM_DTYPE](0.0):
                # The mean itself is inadmissible: nothing better is
                # available than collapsing the element onto it.
                t = 0.0
            else:
                var disc = B * B - Real(4.0) * A * C
                if disc < Scalar[WXM_DTYPE](0.0): disc = 0.0
                var sq = sqrt(disc)
                var qq = -Real(0.5) * (B - sq)
                if B >= Scalar[WXM_DTYPE](0.0): qq = -Real(0.5) * (B + sq)
                if A != Scalar[WXM_DTYPE](0.0):
                    var root_a = qq / A
                    if root_a > Scalar[WXM_DTYPE](0.0) and root_a < t: t = root_a
                if qq != Scalar[WXM_DTYPE](0.0):
                    var root_b = C / qq
                    if root_b > Scalar[WXM_DTYPE](0.0) and root_b < t: t = root_b
            if t < theta_p: theta_p = t

    if theta_p < Scalar[WXM_DTYPE](1.0):
        if theta_p < Scalar[WXM_DTYPE](0.0): theta_p = 0.0
        comptime for n in range(nodes_per_elem):
            var base = (e * nodes_per_elem + n) * NC
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


struct PositivityEuler[NDIM: Int, P: Int]:
    """Device-resident Zhang-Shu positivity preserver for Euler."""

    var rho_eps: Real
    var p_eps: Real
    var gamma: Real
    var num_elements: Int

    def __init__(
        out self, num_elements: Int, gamma: Real,
        rho_eps: Real = Real(1.0e-10), p_eps: Real = Real(1.0e-10),
    ):
        self.rho_eps = rho_eps
        self.p_eps = p_eps
        self.gamma = gamma
        self.num_elements = num_elements

    def apply(
        mut self,
        var ctx: DeviceContext,
        mesh: BlockMesh[Self.NDIM, Self.P],
        mut q: DGState[Self.NDIM, Self.P, EULER_NC],
    ) raises:
        """Restore ρ ≥ ρ_eps and P ≥ P_eps at every node of `q`, in place."""
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)
        var q_layout = row_major(Int(q.buf_len))
        comptime k_fn = _positivity_kernel_euler[
            Self.P, Self.NDIM, type_of(q_layout),
        ]
        ctx.enqueue_function[k_fn](
            TileTensor(q.buf, q_layout),
            self.rho_eps, self.p_eps, self.gamma, Int64(self.num_elements),
            grid_dim=grid, block_dim=BLOCK,
        )
