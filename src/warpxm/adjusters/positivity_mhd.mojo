# Zhang-Shu positivity preserver for ideal MHD — 2D + 3D in a single
# NDIM-parametric kernel. Works for plain IMHD (NC=8) and GLM-MHD
# (NC=9 — ψ is shrunk along with the rest but doesn't affect positivity).
#
# Density step is identical to the Euler positivity preserver; the
# pressure step uses a bisection on a cubic in θ because P_gas now
# includes the magnetic-pressure term:
#
#     P_gas = (γ−1)·[E − ½|m|²/ρ − ½|B|²]
#
# so f(θ) = ρ(θ)·E(θ) − ½|m(θ)|² − ½·ρ(θ)·|B(θ)|² − ρ(θ)·c_e is *cubic*
# in θ (ρ·|B|² is the product of a linear and a quadratic in θ).
# Bisection on [0, 1] for 50 iterations gives ~2^-50 ≈ 9e-16
# precision — more than enough.

from std.collections import InlineArray
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_2d import DGState2D
from warpxm.variable.state_3d import DGState3D


@always_inline
def _mhd_f(
    rho: Scalar[WXM_DTYPE], mx: Scalar[WXM_DTYPE],
    my: Scalar[WXM_DTYPE], mz: Scalar[WXM_DTYPE],
    E: Scalar[WXM_DTYPE], Bx: Scalar[WXM_DTYPE],
    By: Scalar[WXM_DTYPE], Bz: Scalar[WXM_DTYPE],
    c_e: Real,
) -> Scalar[WXM_DTYPE]:
    """Sign-indicator for P_gas ≥ p_eps; positive ⟺ pressure OK."""
    var msq = mx * mx + my * my + mz * mz
    var bsq = Bx * Bx + By * By + Bz * Bz
    return rho * E - Real(0.5) * msq - Real(0.5) * rho * bsq - rho * c_e


@always_inline
def _gll_cell_weight[P: Int, NDIM: Int](n: Int) -> Scalar[WXM_DTYPE]:
    """Tensor-product GLL cell-mean weight (same form as
    `positivity_euler._gll_cell_weight`)."""
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


def _positivity_kernel_mhd[
    P: Int, NDIM: Int, NC: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rho_eps: Real, p_eps: Real, gamma: Real,
    num_elements: Int,
):
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 ** NDIM
    comptime assert q.flat_rank == 1
    comptime assert NDIM >= 2 and NDIM <= 3, "MHD positivity is 2D or 3D"

    var e = Int(global_idx.x)
    if e >= num_elements:
        return

    var c_e = p_eps / (gamma - Real(1.0))

    # Per-component cell means (covers ρ, m, E, B, and ψ if present).
    var bar = InlineArray[Scalar[WXM_DTYPE], NC](fill=Scalar[WXM_DTYPE](0.0))
    comptime for n in range(nodes_per_elem):
        var base = (e * nodes_per_elem + n) * NC
        var w = _gll_cell_weight[P, NDIM](n)
        comptime for c in range(NC):
            bar[c] += w * rebind[Scalar[WXM_DTYPE]](q[base + c])

    # --- Density step ------------------------------------------------------
    var theta_rho: Scalar[WXM_DTYPE] = 1.0
    comptime for n in range(nodes_per_elem):
        var base = (e * nodes_per_elem + n) * NC
        var rho_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
        if rho_n < Scalar[WXM_DTYPE](rho_eps):
            var denom = bar[IDX_RHO] - rho_n
            var t = (bar[IDX_RHO] - Scalar[WXM_DTYPE](rho_eps)) / denom
            if t < theta_rho: theta_rho = t

    if theta_rho < Scalar[WXM_DTYPE](1.0):
        if theta_rho < Scalar[WXM_DTYPE](0.0): theta_rho = 0.0
        comptime for n in range(nodes_per_elem):
            var base = (e * nodes_per_elem + n) * NC
            comptime for c in range(NC):
                var d = rebind[Scalar[WXM_DTYPE]](q[base + c])
                q[base + c] = rebind[q.ElementType](
                    bar[c] + theta_rho * (d - bar[c])
                )

    # --- Pressure step (bisection on cubic) --------------------------------
    var theta_p: Scalar[WXM_DTYPE] = 1.0

    comptime for n in range(nodes_per_elem):
        var base = (e * nodes_per_elem + n) * NC
        var rho_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
        var mx_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MX])
        var my_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MY])
        var mz_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_MZ])
        var E_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_E])
        var Bx_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_BX])
        var By_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_BY])
        var Bz_n = rebind[Scalar[WXM_DTYPE]](q[base + IDX_BZ])
        var f_one = _mhd_f(rho_n, mx_n, my_n, mz_n, E_n, Bx_n, By_n, Bz_n, c_e)
        if f_one < Scalar[WXM_DTYPE](0.0):
            var d_rho = rho_n - bar[IDX_RHO]
            var d_mx = mx_n - bar[IDX_MX]
            var d_my = my_n - bar[IDX_MY]
            var d_mz = mz_n - bar[IDX_MZ]
            var d_E = E_n - bar[IDX_E]
            var d_Bx = Bx_n - bar[IDX_BX]
            var d_By = By_n - bar[IDX_BY]
            var d_Bz = Bz_n - bar[IDX_BZ]
            var lo: Scalar[WXM_DTYPE] = 0.0
            var hi: Scalar[WXM_DTYPE] = 1.0
            comptime for _bisect_it in range(50):
                var mid = Real(0.5) * (lo + hi)
                var r_m = bar[IDX_RHO] + mid * d_rho
                var mx_m = bar[IDX_MX] + mid * d_mx
                var my_m = bar[IDX_MY] + mid * d_my
                var mz_m = bar[IDX_MZ] + mid * d_mz
                var E_m = bar[IDX_E] + mid * d_E
                var Bx_m = bar[IDX_BX] + mid * d_Bx
                var By_m = bar[IDX_BY] + mid * d_By
                var Bz_m = bar[IDX_BZ] + mid * d_Bz
                var f_m = _mhd_f(r_m, mx_m, my_m, mz_m, E_m, Bx_m, By_m, Bz_m, c_e)
                if f_m >= Scalar[WXM_DTYPE](0.0): lo = mid
                else: hi = mid
            if lo < theta_p: theta_p = lo

    if theta_p < Scalar[WXM_DTYPE](1.0):
        if theta_p < Scalar[WXM_DTYPE](0.0): theta_p = 0.0
        comptime for n in range(nodes_per_elem):
            var base = (e * nodes_per_elem + n) * NC
            comptime for c in range(NC):
                var d = rebind[Scalar[WXM_DTYPE]](q[base + c])
                q[base + c] = rebind[q.ElementType](
                    bar[c] + theta_p * (d - bar[c])
                )


struct PositivityMHD2D[P: Int, NC: Int]:
    """Device-resident Zhang-Shu positivity preserver for 2D MHD
    (NC=8 plain IMHD; NC=9 GLM-MHD)."""

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
        mut mesh: BlockMesh2D[Self.P],
        mut q: DGState2D[Self.P, Self.NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)
        var q_layout = row_major(Idx(q.buf_len))
        var q_view = TileTensor(q.buf, q_layout)
        comptime k_fn = _positivity_kernel_mhd[
            Self.P, 2, Self.NC, type_of(q_layout),
        ]
        ctx.enqueue_function[k_fn](
            q_view, self.rho_eps, self.p_eps, self.gamma, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )


struct PositivityMHD3D[P: Int, NC: Int]:
    """Device-resident Zhang-Shu positivity preserver for 3D MHD."""

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
        mut mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, Self.NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)
        var q_layout = row_major(Idx(q.buf_len))
        var q_view = TileTensor(q.buf, q_layout)
        comptime k_fn = _positivity_kernel_mhd[
            Self.P, 3, Self.NC, type_of(q_layout),
        ]
        ctx.enqueue_function[k_fn](
            q_view, self.rho_eps, self.p_eps, self.gamma, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )
