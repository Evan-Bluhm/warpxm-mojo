# Zhang-Shu positivity-preserving limiter for 5-moment Euler — single
# NDIM-parametric version covering 1D / 2D / 3D.
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
# The dim-specific bits are (1) the cell-mean weight at node n (a
# tensor-product GLL weight scaled by 1 / 2^NDIM, i.e. the [-1,1]^NDIM
# reference-element measure normalised to [0,1]^NDIM), and (2) the
# per-dim DGState type used at the host-side API. The kernel body is
# identical across dimensions and runs flat over `(P+1)^NDIM` nodes.

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, row_major

from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_1d import DGState1D
from warpxm.variable.state_2d import DGState2D
from warpxm.variable.state_3d import DGState3D


@always_inline
def _pressure(
    rho: Scalar[WXM_DTYPE], mx: Scalar[WXM_DTYPE],
    my: Scalar[WXM_DTYPE], mz: Scalar[WXM_DTYPE],
    E: Scalar[WXM_DTYPE], gamma: Real,
) -> Scalar[WXM_DTYPE]:
    var ke = Real(0.5) * (mx * mx + my * my + mz * mz) / rho
    return (gamma - Real(1.0)) * (E - ke)


@always_inline
def _gll_cell_weight[P: Int, NDIM: Int](n: Int) -> Scalar[WXM_DTYPE]:
    """Tensor-product GLL weight for node `n` on the reference
    [-1, 1]^NDIM element, normalised by 1 / 2^NDIM so the unit-cell
    integral of 1 is 1. Replicates the per-dim cell-mean weights used
    in the legacy positivity preservers exactly.
    """
    comptime P1 = P + 1
    var weights = NodalBasis1D[P].weights()
    comptime if NDIM == 1:
        return weights[n] * Real(0.5)
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
        var w = _gll_cell_weight[P, NDIM](n)
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


# ---------------------------------------------------------------------------
# Per-dim host-side launchers.
# ---------------------------------------------------------------------------

struct PositivityEuler1D[P: Int]:
    """Device-resident Zhang-Shu positivity preserver for 1D Euler."""

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
        mesh: BlockMesh1D[Self.P],
        mut q: DGState1D[Self.P, EULER_NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)
        var q_layout = row_major(Int(q.buf_len))
        var q_view = TileTensor(q.buf, q_layout)
        comptime k_fn = _positivity_kernel_euler[Self.P, 1, type_of(q_layout)]
        ctx.enqueue_function[k_fn](
            q_view, self.rho_eps, self.p_eps, self.gamma, Int64(self.num_elements),
            grid_dim=grid, block_dim=BLOCK,
        )


struct PositivityEuler2D[P: Int]:
    """Device-resident Zhang-Shu positivity preserver for 2D Euler."""

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
        mesh: BlockMesh2D[Self.P],
        mut q: DGState2D[Self.P, EULER_NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)
        var q_layout = row_major(Int(q.buf_len))
        var q_view = TileTensor(q.buf, q_layout)
        comptime k_fn = _positivity_kernel_euler[Self.P, 2, type_of(q_layout)]
        ctx.enqueue_function[k_fn](
            q_view, self.rho_eps, self.p_eps, self.gamma, Int64(self.num_elements),
            grid_dim=grid, block_dim=BLOCK,
        )


struct PositivityEuler3D[P: Int]:
    """Device-resident Zhang-Shu positivity preserver for 3D Euler."""

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
        mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, EULER_NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)
        var q_layout = row_major(Int(q.buf_len))
        var q_view = TileTensor(q.buf, q_layout)
        comptime k_fn = _positivity_kernel_euler[Self.P, 3, type_of(q_layout)]
        ctx.enqueue_function[k_fn](
            q_view, self.rho_eps, self.p_eps, self.gamma, Int64(self.num_elements),
            grid_dim=grid, block_dim=BLOCK,
        )
