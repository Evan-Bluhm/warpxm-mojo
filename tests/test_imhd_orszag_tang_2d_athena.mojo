# Orszag-Tang vortex — Athena code normalization.
#
# Same physical problem as `test_imhd_orszag_tang_2d.mojo` but adopts the
# unit convention used by Athena (Stone et al.) so that results can be
# compared directly with the canonical Athena reference at
#   https://www.astro.princeton.edu/~jstone/Athena/tests/orszag-tang/
# and the C source at `~/GitHub/Athena-Cversion/src/prob/orszag-tang.c`.
#
# Domain: [0, 1]² periodic; γ = 5/3.
# Constant background:
#     d_0 = 25 / (36π)             ≈ 0.22105
#     p_0 = 5  / (12π)             ≈ 0.13263
# Velocity vortex (period 1 in both directions):
#     u_x = -v_0 · sin(2π y)
#     u_y =  v_0 · sin(2π x)
# Magnetic field — derived from the vector potential
#     A_z = (B_0 / 4π) · cos(4π x) + (B_0 / 2π) · cos(2π y)
# so that ∇·B = 0 analytically:
#     B_x =  ∂A_z/∂y = -B_0 · sin(2π y)
#     B_y = -∂A_z/∂x =  B_0 · sin(4π x)
# with v_0 = 1, B_0 = 1 / √(4π).
#
# Initial wave speeds (uniform):
#     c_s    = √(γ p_0 / d_0)            = 1
#     v_a    = B_0 / √d_0                 = 3/5  = 0.6
#     v_max ≈ |u| + √(c_s² + v_a²)        ≈ 2.166
#
# Why this test is hard:
#   * From a smooth IC, multiple shock fronts develop within ~ t = 0.3
#     and the flow becomes turbulent past t ~ 0.5.
#   * The magnetic field develops sharp gradients that drive ∇·B errors
#     in any base scheme; GLM divergence cleaning damps them.
#   * Supersonic / supermagnetic shocks need positivity preservation +
#     the Moe-Rossmanith slope limiter to avoid blowing up.
#
# We integrate to t = 0.3 (past the first shock formation, before the
# fully-turbulent regime) and assert:
#   * no NaN
#   * mass conservation to machine precision (periodic, no source on ρ)
#   * density remains strictly positive
#   * gas pressure remains strictly positive (positivity preserver works)
#   * density and pressure stay within physically reasonable bounds
#   * total ψ-energy stays small (∇·B errors don't run away)

from std.math import sin, cos, isnan
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.adjusters.positivity_mhd import PositivityMHD2D
from warpxm.apps.imhd_glm import (
    ImhdGLM, MHD_GLM_NC, IDX_PSI,
)
from warpxm.apps.imhd import (
    IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real, PI, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_imhd_glm_2d import DGImhdGLMSolver2D
from warpxm.spatial.imhd_glm_source_2d import apply_glm_damping_2d
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime V0: Real = 1.0
comptime C_H: Real = 3.0       # cleaning speed (~ max signal speed at IC)
comptime C_P_SQ: Real = 0.18   # parabolic damping
comptime T_FINAL: Real = 0.3
comptime DT: Real = 5.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MHD_GLM_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var d0: Real = Real(25.0) / (Real(36.0) * PI)
    var p0: Real = Real(5.0) / (Real(12.0) * PI)
    var B0: Real = Real(1.0) / sqrt(Real(4.0) * PI)

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

                    var ux = -V0 * sin(TWO_PI * y)
                    var uy =  V0 * sin(TWO_PI * x)
                    var Bx_v = -B0 * sin(TWO_PI * y)
                    var By_v =  B0 * sin(Real(2.0) * TWO_PI * x)

                    var msq = d0 * (ux * ux + uy * uy)
                    var bsq = Bx_v * Bx_v + By_v * By_v
                    var E_tot = p0 / (GAMMA - Real(1.0)) + Real(0.5) * msq + Real(0.5) * bsq

                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                    host[b + IDX_RHO] = d0
                    host[b + IDX_MX] = d0 * ux
                    host[b + IDX_MY] = d0 * uy
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
    var positivity = PositivityMHD2D[P, MHD_GLM_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )

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
        positivity.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var any_nan = False
    var mass_T: Real = 0.0
    var psi2_T: Real = 0.0
    var max_rho: Real = -1.0e30
    var min_rho: Real =  1.0e30
    var max_p: Real = -1.0e30
    var min_p: Real =  1.0e30
    with q.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MHD_GLM_NC
                        var rho = h[b + IDX_RHO]
                        var mx = h[b + IDX_MX]
                        var my = h[b + IDX_MY]
                        var mz = h[b + IDX_MZ]
                        var E_tot = h[b + IDX_E]
                        var Bx = h[b + IDX_BX]
                        var By = h[b + IDX_BY]
                        var Bz = h[b + IDX_BZ]
                        var psi = h[b + IDX_PSI]
                        if isnan(rho) or isnan(psi) or isnan(E_tot):
                            any_nan = True
                        var ke = Real(0.5) * (mx * mx + my * my + mz * mz) / rho
                        var mag = Real(0.5) * (Bx * Bx + By * By + Bz * Bz)
                        var pp = (GAMMA - Real(1.0)) * (E_tot - ke - mag)
                        var jw = ja * weights[in_] * weights[jn]
                        mass_T += jw * rho
                        psi2_T += jw * psi * psi
                        if rho > max_rho: max_rho = rho
                        if rho < min_rho: min_rho = rho
                        if pp > max_p: max_p = pp
                        if pp < min_p: min_p = pp

    var rel_mass_drift = fabs(mass_T - mass_0) / mass_0
    print("Orszag-Tang vortex (Athena units) — t =", t, " on", NX, "x", NY,
          "mesh, P =", P)
    print("  initial:  d_0 =", d0, " p_0 =", p0, " B_0 =", B0)
    print("  mass  (initial / final):   ", mass_0, " /", mass_T)
    print("  rel Δmass:                  ", rel_mass_drift)
    print("  ρ range: [", min_rho, ",", max_rho, "]  (initial =", d0, ")")
    print("  P range: [", min_p, ",", max_p, "]  (initial =", p0, ")")
    print("  ∫ψ² dV:                     ", psi2_T)

    assert_true(not any_nan, "Orszag-Tang produced NaN (GLM + limiter unstable)")
    assert_true(rel_mass_drift < Real(1.0e-12), "mass not conserved")
    assert_true(min_rho > Real(0.0),
                "density went non-positive — positivity preserver broken")
    assert_true(min_p > Real(0.0),
                "pressure went non-positive — positivity preserver broken")
    # Density compression should stay physical: Athena's reference shows
    # ρ_max ≲ 0.5 (≈ 2.3 × d_0) at t = 0.3, well below this bound.
    assert_true(max_rho < Real(10.0) * d0, "density blew up — unstable shock")
    # GLM ψ² should be O(B²·c_p²) at most; a loose bound catches divergence.
    assert_true(psi2_T < Real(1.0e-2), "GLM ψ² grew uncontrollably")
    print("PASSED")
