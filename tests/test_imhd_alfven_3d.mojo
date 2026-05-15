# 3D ideal-MHD circularly-polarized Alfvén wave regression.
#
# Same physics as `test_imhd_alfven_2d.mojo`, but run on a 3D mesh:
#
#     Background:   ρ = 1, P = 1, B = (B_0, 0, 0) with B_0 = 1
#     Perturbation along x̂ (uniform in y and z):
#         B_y(x, 0) =  δ·cos(kx)
#         B_z(x, 0) =  δ·sin(kx)
#         u_y(x, 0) = −δ·cos(kx) / √ρ
#         u_z(x, 0) = −δ·sin(kx) / √ρ
#
# The Alfvén speed along x̂ is c_A = B_x / √(ρ μ₀) = 1. The wave returns
# to IC at t = λ/c_A = 1.
#
# Although the physics is 1D, the 3D kernel must:
#   * compute all three flux divergences and only the x-axis should
#     contribute non-trivial terms;
#   * preserve incompressibility (|B|² constant in space and time, ρ ≈ const).

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.imhd_3d import IMHD3D
from warpxm.apps.imhd_2d import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 4
comptime NZ: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO0: Real = 1.0
comptime P0: Real = 1.0
comptime B0: Real = 1.0
comptime DELTA: Real = 0.1
comptime DT: Real = 1.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 4.0), Real(0.0), Real(1.0 / 4.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1
    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var c_v = cos(k_wave * x)
                            var s_v = sin(k_wave * x)
                            var By_ic = DELTA * c_v
                            var Bz_ic = DELTA * s_v
                            var uy_ic = -DELTA * c_v * inv_sqrt_rho
                            var uz_ic = -DELTA * s_v * inv_sqrt_rho
                            var bsq = B0 * B0 + By_ic * By_ic + Bz_ic * Bz_ic
                            var msq = RHO0 * RHO0 * (uy_ic * uy_ic + uz_ic * uz_ic)
                            var E_total = P0 / (GAMMA - Real(1.0)) + Real(0.5) * msq / RHO0 + Real(0.5) * bsq
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_NC
                            host[b + IDX_RHO] = RHO0
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = RHO0 * uy_ic
                            host[b + IDX_MZ] = RHO0 * uz_ic
                            host[b + IDX_E] = E_total
                            host[b + IDX_BX] = B0
                            host[b + IDX_BY] = By_ic
                            host[b + IDX_BZ] = Bz_ic
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = IMHD3D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver3D[P](app)
    var stepper = SSPRK3_3D[P, MHD_NC](ctx, mesh.num_elements)

    var t_final: Real = 1.0

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, MHD_NC],
        mut q_out: DGState3D[P, MHD_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState3D[P, MHD_NC]) raises:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var err_By: Real = 0.0
    var err_Bz: Real = 0.0
    var max_drho: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var xl = mesh.x_min + Real(i) * mesh.dx
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                                var c_v = cos(k_wave * x)
                                var s_v = sin(k_wave * x)
                                var By_exact = DELTA * c_v
                                var Bz_exact = DELTA * s_v
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MHD_NC
                                var d_By = h[b + IDX_BY] - By_exact
                                var d_Bz = h[b + IDX_BZ] - Bz_exact
                                var d_rho = h[b + IDX_RHO] - RHO0
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                err_By += jw * d_By * d_By
                                err_Bz += jw * d_Bz * d_Bz
                                if d_rho > max_drho: max_drho = d_rho
                                if -d_rho > max_drho: max_drho = -d_rho
    var l2_By = err_By ** Real(0.5)
    var l2_Bz = err_Bz ** Real(0.5)
    print("3D Alfvén wave — t =", t)
    print("  L2 B_y    =", l2_By)
    print("  L2 B_z    =", l2_Bz)
    print("  max |Δρ|  =", max_drho)
    assert_true(l2_By < Real(5.0e-3), "B_y L2 error too large")
    assert_true(l2_Bz < Real(5.0e-3), "B_z L2 error too large")
    assert_true(max_drho < Real(1.0e-2), "density drifted (wave should be incompressible)")
    print("PASSED")
