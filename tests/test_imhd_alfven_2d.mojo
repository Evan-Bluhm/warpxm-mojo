# Circularly-polarized Alfvén wave — canonical ideal-MHD validation.
#
# Setup (uniform in y; the 2D mesh is just a thin slab for the 2D solver):
#
#     Background:  ρ = 1,  P = 1,  B = (B_0, 0, 0) with B_0 = 1
#     Perturbation along x with wavenumber k = 2π:
#         B_y(x, 0) =  δ · cos(kx)
#         B_z(x, 0) =  δ · sin(kx)
#         u_y(x, 0) = −δ · cos(kx) / √ρ      (Walen relation, +x propagating)
#         u_z(x, 0) = −δ · sin(kx) / √ρ
#     All other components zero / background.
#
# The Alfvén speed along x̂ is c_A = B_x / √(ρ μ₀) = 1 in normalised units.
# The wave propagates at +c_A. After T = λ/c_A = 1/1 = 1 the solution
# returns to the IC.
#
# Why this is a strong IMHD check:
#   * |B|² = B_x² + B_y² + B_z² = 1 + δ² is spatially uniform AND constant
#     in time, so the total pressure P + |B|²/2 is constant — no spurious
#     compression from the perturbation.
#   * The Walen relation `δu = −δB / √(ρμ₀)` is the exact non-linear
#     Alfvén-wave eigenvector, not just a linearisation.
#   * Both perpendicular components of B and u are non-trivial → every
#     off-diagonal term of the MHD flux tensor is exercised.

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.imhd import (
    IMHD, MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
    IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_2d import DGImhdSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime RHO0: Real = 1.0
comptime P0: Real = 1.0
comptime B0: Real = 1.0
comptime DELTA: Real = 0.1
comptime DT: Real = 1.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MHD_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1
    var inv_sqrt_rho = Real(1.0) / sqrt(RHO0)
    var k_wave = TWO_PI

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
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

                    var n = jn * P1 + in_
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

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MHD_NC](ctx, mesh.num_elements)

    var t_final: Real = 1.0

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MHD_NC],
        mut q_out: DGState2D[P, MHD_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState2D[P, MHD_NC]) raises:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var err_By: Real = 0.0
    var err_Bz: Real = 0.0
    var err_uy: Real = 0.0
    var err_uz: Real = 0.0
    var max_drho: Real = 0.0
    var max_dE: Real = 0.0
    var E_expected = P0 / (GAMMA - Real(1.0)) + Real(0.5) * (B0 * B0 + DELTA * DELTA) + Real(0.5) * DELTA * DELTA
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                        var c_v = cos(k_wave * x)
                        var s_v = sin(k_wave * x)
                        var By_exact = DELTA * c_v
                        var Bz_exact = DELTA * s_v
                        var uy_exact = -DELTA * c_v * inv_sqrt_rho
                        var uz_exact = -DELTA * s_v * inv_sqrt_rho

                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_NC
                        var rho = h[b + IDX_RHO]
                        var uy = h[b + IDX_MY] / rho
                        var uz = h[b + IDX_MZ] / rho

                        var d_By = h[b + IDX_BY] - By_exact
                        var d_Bz = h[b + IDX_BZ] - Bz_exact
                        var d_uy = uy - uy_exact
                        var d_uz = uz - uz_exact
                        var d_rho = rho - RHO0
                        var d_E = h[b + IDX_E] - E_expected
                        var jw = ja * weights[in_] * weights[jn]
                        err_By += jw * d_By * d_By
                        err_Bz += jw * d_Bz * d_Bz
                        err_uy += jw * d_uy * d_uy
                        err_uz += jw * d_uz * d_uz
                        if d_rho > max_drho: max_drho = d_rho
                        if -d_rho > max_drho: max_drho = -d_rho
                        if d_E > max_dE: max_dE = d_E
                        if -d_E > max_dE: max_dE = -d_E
    var l2_By = err_By ** Real(0.5)
    var l2_Bz = err_Bz ** Real(0.5)
    var l2_uy = err_uy ** Real(0.5)
    var l2_uz = err_uz ** Real(0.5)
    print("After one Alfvén period (t = 1):")
    print("  L2 B_y   =", l2_By)
    print("  L2 B_z   =", l2_Bz)
    print("  L2 u_y   =", l2_uy)
    print("  L2 u_z   =", l2_uz)
    print("  max |Δρ| =", max_drho)
    print("  max |ΔE| =", max_dE)
    assert_true(l2_By < Real(2.0e-3), "B_y L2 error too large")
    assert_true(l2_Bz < Real(2.0e-3), "B_z L2 error too large")
    assert_true(l2_uy < Real(2.0e-3), "u_y L2 error too large")
    assert_true(l2_uz < Real(2.0e-3), "u_z L2 error too large")
    # ρ should be (almost) invariant; the wave is incompressible.
    assert_true(max_drho < Real(1.0e-3), "density drifted (Alfvén wave should be incompressible)")
    print("PASSED")
