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

from std.collections import Array
from std.math import isnan, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.imhd_glm import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime V0: Real = 1.0
comptime C_H: Real = 3.0       # cleaning speed (~ max signal speed at IC)
comptime C_P_SQ: Real = 0.18   # parabolic damping
comptime T_FINAL: Real = 0.3
comptime DT: Real = 5.0e-4
comptime State = DGState[2, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    var d0: Real = Real(25.0) / (Real(36.0) * PI)
    var p0: Real = Real(5.0) / (Real(12.0) * PI)
    var B0: Real = Real(1.0) / sqrt(Real(4.0) * PI)

    def initial(x: Real, y: Real) {imm} -> Array[Real, MHD_GLM_NC]:
        var ux = -V0 * sin(TWO_PI * y)
        var uy = V0 * sin(TWO_PI * x)
        var bx = -B0 * sin(TWO_PI * y)
        var by = B0 * sin(Real(2.0) * TWO_PI * x)
        var msq = d0 * (ux * ux + uy * uy)
        var bsq = bx * bx + by * by
        var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        v[IDX_RHO] = d0
        v[IDX_MX] = d0 * ux
        v[IDX_MY] = d0 * uy
        v[IDX_E] = p0 / (GAMMA - Real(1.0)) + Real(0.5) * msq + Real(0.5) * bsq
        v[IDX_BX] = bx
        v[IDX_BY] = by
        return v^
    mesh.fill(q, initial)
    var mass_0 = mesh.integrate(q, IDX_RHO)

    var app = ImhdGLM(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
        c_h=C_H, c_p_sq=C_P_SQ,
    )
    var spatial = DGImhdGLMSolver[2, P](app)
    var stepper = SSPRK3[2, P, MHD_GLM_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[2, P, MHD_GLM_NC](ctx, mesh.num_elements)
    var positivity = PositivityMHD[2, P, MHD_GLM_NC](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: State) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    var any_nan = False
    var psi2_T: Real = 0.0
    var max_rho: Real = -1.0e30
    var min_rho: Real = 1.0e30
    var max_p: Real = -1.0e30
    var min_p: Real = 1.0e30
    def scan(
        x: Real, y: Real, w: Real, v: Array[Real, MHD_GLM_NC],
    ) {mut any_nan, mut psi2_T, mut max_rho, mut min_rho, mut max_p, mut min_p}:
        var rho = v[IDX_RHO]
        var psi = v[IDX_PSI]
        if isnan(rho) or isnan(psi) or isnan(v[IDX_E]):
            any_nan = True
        var ke = Real(0.5) * (
            v[IDX_MX] * v[IDX_MX] + v[IDX_MY] * v[IDX_MY] + v[IDX_MZ] * v[IDX_MZ]
        ) / rho
        var mag = Real(0.5) * (
            v[IDX_BX] * v[IDX_BX] + v[IDX_BY] * v[IDX_BY] + v[IDX_BZ] * v[IDX_BZ]
        )
        var pp = (GAMMA - Real(1.0)) * (v[IDX_E] - ke - mag)
        psi2_T += w * psi * psi
        if rho > max_rho: max_rho = rho
        if rho < min_rho: min_rho = rho
        if pp > max_p: max_p = pp
        if pp < min_p: min_p = pp
    mesh.visit_nodes(q, scan)
    var mass_T = mesh.integrate(q, IDX_RHO)

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
    assert_true(psi2_T < Real(1.0e-3), "GLM ψ² grew uncontrollably")
    print("PASSED")
