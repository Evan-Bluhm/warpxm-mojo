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
comptime C_H: Real = 4.0       # cleaning speed (~ max signal speed at IC)
comptime C_P_SQ: Real = 0.5    # parabolic damping
comptime T_FINAL: Real = 0.4
comptime DT: Real = 1.0e-3
comptime State = DGState[2, P, MHD_GLM_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), TWO_PI, Real(0.0), TWO_PI,
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    var rho_0 = GAMMA * GAMMA
    var p_0: Real = GAMMA

    def initial(x: Real, y: Real) {imm} -> Array[Real, MHD_GLM_NC]:
        var ux = -sin(y)
        var uy = sin(x)
        var bx = -sin(y)
        var by = sin(Real(2.0) * x)
        var msq = rho_0 * (ux * ux + uy * uy)
        var bsq = bx * bx + by * by
        var v = Array[Real, MHD_GLM_NC](fill=Real(0.0))
        v[IDX_RHO] = rho_0
        v[IDX_MX] = rho_0 * ux
        v[IDX_MY] = rho_0 * uy
        v[IDX_E] = p_0 / (GAMMA - Real(1.0)) + Real(0.5) * msq + Real(0.5) * bsq
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

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)
        apply_glm_damping(ctx, q_in, q_out, C_H, C_P_SQ)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, DT, compute_rhs, adjust,
    )
    ctx.synchronize()

    var any_nan = False
    var psi2_T: Real = 0.0
    var max_rho: Real = 0.0
    var min_rho: Real = 1.0e30
    def scan(
        x: Real, y: Real, w: Real, v: Array[Real, MHD_GLM_NC],
    ) {mut any_nan, mut psi2_T, mut max_rho, mut min_rho}:
        var rho = v[IDX_RHO]
        var psi = v[IDX_PSI]
        if isnan(rho) or isnan(psi) or isnan(v[IDX_E]):
            any_nan = True
        psi2_T += w * psi * psi
        if rho > max_rho: max_rho = rho
        if rho < min_rho: min_rho = rho
    mesh.visit_nodes(q, scan)
    var mass_T = mesh.integrate(q, IDX_RHO)

    var rel_mass_drift = fabs(mass_T - mass_0) / mass_0
    print("Orszag-Tang vortex — t =", t, " on", NX, "x", NY, "mesh, P =", P)
    print("  mass  (initial / final):  ", mass_0, " /", mass_T)
    print("  rel Δmass:                 ", rel_mass_drift)
    print("  ρ range (initial = ", rho_0, "): [", min_rho, ",", max_rho, "]")
    print("  ∫ψ² dV:                    ", psi2_T)

    assert_true(not any_nan, "Orszag-Tang produced NaN (GLM + limiter unstable)")
    assert_true(rel_mass_drift < Real(1.0e-12), "mass not conserved")
    assert_true(psi2_T < Real(1.0e-3), "GLM ψ² grew uncontrollably")
    assert_true(min_rho > Real(0.0), "density went non-positive")
    assert_true(max_rho < Real(10.0) * rho_0, "density blew up — unstable shock")
    print("PASSED")
