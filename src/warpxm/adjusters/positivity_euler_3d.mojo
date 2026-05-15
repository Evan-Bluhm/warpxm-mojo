# Zhang-Shu positivity preserver for 3D Euler — same algorithm as the 1D
# and 2D versions, tensor-product GLL quadrature in three dimensions.

from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv
from layout import TileTensor, TensorLayout, Idx, row_major

from warpxm.apps.euler_2d import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fmin, fmax, sqrt
from warpxm.core.types import Real, WXM_DTYPE
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.variable.state_3d import DGState3D


@always_inline
def _pressure(
    rho: Scalar[WXM_DTYPE], mx: Scalar[WXM_DTYPE],
    my: Scalar[WXM_DTYPE], mz: Scalar[WXM_DTYPE],
    E: Scalar[WXM_DTYPE], gamma: Real,
) -> Scalar[WXM_DTYPE]:
    var ke = Real(0.5) * (mx * mx + my * my + mz * mz) / rho
    return (gamma - Real(1.0)) * (E - ke)


def _positivity_kernel_3d[
    P: Int, QLayout: TensorLayout,
](
    q: TileTensor[WXM_DTYPE, QLayout, MutAnyOrigin],
    rho_eps: Real, p_eps: Real, gamma: Real,
    num_elements: Int,
):
    comptime P1 = P + 1
    comptime nodes_per_elem = P1 * P1 * P1
    comptime NC = EULER_NC
    comptime assert q.flat_rank == 1

    var e = Int(global_idx.x)
    if e >= num_elements:
        return

    var weights = NodalBasis1D[P].weights()

    # --- Cell averages -----------------------------------------------------
    var rho_bar: Scalar[WXM_DTYPE] = 0.0
    var mx_bar: Scalar[WXM_DTYPE] = 0.0
    var my_bar: Scalar[WXM_DTYPE] = 0.0
    var mz_bar: Scalar[WXM_DTYPE] = 0.0
    var E_bar: Scalar[WXM_DTYPE] = 0.0
    comptime for kn in range(P1):
        comptime for jn in range(P1):
            comptime for in_ in range(P1):
                var n = kn * P1 * P1 + jn * P1 + in_
                var base = (e * nodes_per_elem + n) * NC
                # GLL weights on [-1,1] sum to 2 per axis; 3D weight is
                # wᵢ wⱼ wₖ / 8 to integrate over [-1,1]³.
                var w = weights[in_] * weights[jn] * weights[kn] * Real(0.125)
                rho_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_RHO])
                mx_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MX])
                my_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MY])
                mz_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_MZ])
                E_bar += w * rebind[Scalar[WXM_DTYPE]](q[base + IDX_E])

    # --- Density step ------------------------------------------------------
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

    # --- Pressure step -----------------------------------------------------
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
            var disc = B * B - Real(4.0) * A * C
            if disc < Scalar[WXM_DTYPE](0.0): disc = 0.0
            var sq = sqrt(disc)
            var two_a = Real(2.0) * A + Real(1.0e-30)
            var root_a = (-B - sq) / two_a
            var root_b = (-B + sq) / two_a
            var t: Scalar[WXM_DTYPE] = 1.0
            if root_a > Scalar[WXM_DTYPE](0.0) and root_a < t: t = root_a
            if root_b > Scalar[WXM_DTYPE](0.0) and root_b < t: t = root_b
            if t < Scalar[WXM_DTYPE](0.0): t = 0.0
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


struct PositivityEuler3D[P: Int]:
    """Device-resident Zhang-Shu positivity preserver for 3D Euler."""

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
        mut mesh: BlockMesh3D[Self.P],
        mut q: DGState3D[Self.P, EULER_NC],
    ) raises:
        comptime BLOCK = 256
        var grid = ceildiv(self.num_elements, BLOCK)
        var q_layout = row_major(Idx(q.buf_len))
        var q_view = TileTensor(q.buf, q_layout)
        comptime k_fn = _positivity_kernel_3d[Self.P, type_of(q_layout)]
        ctx.enqueue_function[k_fn](
            q_view, self.rho_eps, self.p_eps, self.gamma, self.num_elements,
            grid_dim=grid, block_dim=BLOCK,
        )
