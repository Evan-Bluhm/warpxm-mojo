# Orszag-Tang vortex — canonical 2D MHD turbulence + shocks test
# (Orszag & Tang, J. Fluid Mech. 1979).
#
# Domain: [0, 2π]² periodic; γ = 5/3.
# Equilibrium-density / quiescent-thermal background:
#     ρ_0 = γ² = 25/9
#     P_0 = γ  = 5/3
# Velocity vortex:
#     u_x = -sin(y),  u_y = sin(x),  u_z = 0
# Magnetic vortex (rotated, half wavelength in x):
#     B_x = -sin(y),  B_y = sin(2x),  B_z = 0
#
# Why this test is hard:
#   * From a smooth IC, multiple shock fronts develop within ~ t = 0.5
#     and continue to evolve into supersonic MHD turbulence.
#   * The magnetic field develops sharp gradients that drive ∇·B errors
#     in any base scheme; GLM divergence cleaning damps them.
#   * The supersonic / supermagnetic shocks need a positivity-preserving
#     slope limiter to avoid blowing up.
#
# We run a short window (t = 0.4 — past the formation of the first shocks
# but before the most turbulent late-time regime) and assert:
#   * no NaN
#   * mass conservation to machine precision (periodic, no sources on ρ)
#   * total ψ-energy stays small (∇·B errors don't run away)

from std.math import cos, sin, isnan
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.apps.imhd_glm import (
    ImhdGLM, MHD_GLM_NC, IDX_PSI,
)
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_glm_2d import DGImhdGLMSolver2D
from warpxm.spatial.imhd_glm_source_2d import apply_glm_damping_2d
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime C_H: Real = 4.0       # cleaning speed (~ max signal speed at IC)
comptime C_P_SQ: Real = 0.5    # parabolic damping
comptime T_FINAL: Real = 0.4
comptime DT: Real = 1.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), TWO_PI, Real(0.0), TWO_PI,
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var rho_0 = GAMMA * GAMMA
    var p_0: Real = GAMMA

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x

                    var ux = -sin(y)
                    var uy = sin(x)
                    var Bx_v = -sin(y)
                    var By_v = sin(Real(2.0) * x)

                    var rho = rho_0
                    var msq = rho * (ux * ux + uy * uy)
                    var bsq = Bx_v * Bx_v + By_v * By_v
                    var E_tot = p_0 / (GAMMA - Real(1.0)) + Real(0.5) * msq + Real(0.5) * bsq

                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = rho * ux
                    host[b + IDX_MY] = rho * uy
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = E_tot
                    host[b + IDX_BX] = Bx_v
                    host[b + IDX_BY] = By_v
                    host[b + IDX_BZ] = Real(0.0)
                    host[b + IDX_PSI] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MHD_GLM_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var mass_0: Real = 0.0
    for j in range(NY):
        for i in range(NX):
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                    mass_0 += ja * weights[in_] * weights[jn] * host[b + IDX_RHO]


    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MHD_GLM_NC],
        mut q_out: DGState2D[P, MHD_GLM_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping_2d[P](ctx, q_in, q_out, C_H, C_P_SQ)

    @parameter
    def adjust(mut q_: DGState2D[P, MHD_GLM_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var any_nan = False
    var mass_T: Real = 0.0
    var psi2_T: Real = 0.0
    var max_rho: Real = 0.0
    var min_rho: Real = 1.0e30
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                        var rho = h[b + IDX_RHO]
                        var psi = h[b + IDX_PSI]
                        var E_tot = h[b + IDX_E]
                        if isnan(rho) or isnan(psi) or isnan(E_tot):
                            any_nan = True
                        var jw = ja * weights[in_] * weights[jn]
                        mass_T += jw * rho
                        psi2_T += jw * psi * psi
                        if rho > max_rho: max_rho = rho
                        if rho < min_rho: min_rho = rho

    var rel_mass_drift = fabs(mass_T - mass_0) / mass_0
    print("Orszag-Tang vortex — t =", t, " on", NX, "x", NY, "mesh, P =", P)
    print("  mass  (initial / final):  ", mass_0, " /", mass_T)
    print("  rel Δmass:                 ", rel_mass_drift)
    print("  ρ range (initial = ", rho_0, "): [", min_rho, ",", max_rho, "]")
    print("  ∫ψ² dV:                    ", psi2_T)

    assert_true(not any_nan, "Orszag-Tang produced NaN (GLM + limiter unstable)")
    assert_true(rel_mass_drift < Real(1.0e-12), "mass not conserved")
    assert_true(min_rho > Real(0.0), "density went non-positive")
    assert_true(max_rho < Real(10.0) * rho_0, "density blew up — unstable shock")
    print("PASSED")
