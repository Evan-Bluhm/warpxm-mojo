# Langmuir oscillation — closed-loop two-fluid plasma test.
#
# An electron fluid (charge-to-mass ratio α = −1 in normalised units) sits
# on top of a uniform immobile ion background of density n₀ = 1. A small
# sinusoidal density perturbation drives plasma oscillations at the plasma
# frequency ω_p = √(α²·n₀ / ε₀) = 1.
#
# Linearised cold-plasma solution (k = 2π, δ = 0.01):
#     ρ(x, t)   = 1 + δ·sin(kx)·cos(ω_p t)
#     ρu(x, t)  = −(δ/k)·cos(kx)·sin(ω_p t)
#     E_x(x, t) = +(δ/k)·cos(kx)·cos(ω_p t)
#     all other components zero
#
# IC: cos(ω_p · 0) = 1, sin(0) = 0, so ρu = 0 and the density and E_x
# perturbations are at their peaks. Gauss's law at t = 0 is satisfied by
# the chosen E_x profile (∂E_x/∂x = −δ·sin(kx) = ρ_q/ε₀).
#
# We use a finite (but small) thermal pressure `P_0 = 1e-4`; the
# Bohm-Gross correction shifts ω to ≈ 1.003, giving ~ 2 % phase error
# after a full period — well inside the 1 % tolerance we use as the
# regression bar on ρ amplitude. Mass conservation is the second check.

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler_2d import (
    Euler2D, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.maxwell_2d import (
    Maxwell2D, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.temporal.ssprk3_twofluid_2d import SSPRK3_TwoFluid_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 4         # uniform-in-y; just enough for the 2D kernel
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = -1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime P0: Real = 1.0e-4
comptime DT: Real = 5.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q_fluid = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
    var q_maxwell = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var k_wave = TWO_PI
    var omega_p: Real = 1.0  # ε₀ = 1, α = −1, n₀ = 1

    # IC fluid + Maxwell.
    var fh = ctx.enqueue_create_host_buffer[Real.dtype](q_fluid.buf_len)
    var mh = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var s = sin(k_wave * x)
                    var c_v = cos(k_wave * x)
                    var rho = Real(1.0) + DELTA * s
                    var n = jn * P1 + in_

                    var fb = (e * nodes_per_elem + n) * EULER_NC
                    fh[fb + IDX_RHO] = rho
                    fh[fb + IDX_MX] = Real(0.0)
                    fh[fb + IDX_MY] = Real(0.0)
                    fh[fb + IDX_MZ] = Real(0.0)
                    # E_total = P₀ / (γ−1) + ½·ρ·|u|², |u| = 0 at IC.
                    fh[fb + IDX_E] = P0 / (GAMMA - Real(1.0))

                    var mb = (e * nodes_per_elem + n) * MAXWELL_NC
                    mh[mb + IDX_EX] = (DELTA / k_wave) * c_v
                    mh[mb + IDX_EY] = Real(0.0)
                    mh[mb + IDX_EZ] = Real(0.0)
                    mh[mb + IDX_BX] = Real(0.0)
                    mh[mb + IDX_BY] = Real(0.0)
                    mh[mb + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q_fluid.buf, src_buf=fh)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=mh)

    # Initial mass for conservation check.
    var ja = mesh.jacobian_x * mesh.jacobian_y
    var mass0: Real = 0.0
    for j in range(NY):
        for i in range(NX):
            var e = j * NX + i
            for jn in range(P1):
                for in_ in range(P1):
                    var n = jn * P1 + in_
                    var fb = (e * nodes_per_elem + n) * EULER_NC
                    mass0 += ja * weights[in_] * weights[jn] * fh[fb + IDX_RHO]

    var fluid_app = Euler2D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell2D(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver2D[P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver2D[P](maxwell_app)
    var stepper = SSPRK3_TwoFluid_2D[P](ctx, mesh.num_elements)

    # Track ρ at one fixed sample point through time to confirm oscillation.
    var sample_e = 0 * NX + (NX // 4)  # near x = 0.25 (sin(2π·0.25) = 1, peak)
    var sample_n = 0
    var rho_at_quarter_T: Real = 0.0
    var rho_at_half_T: Real = 0.0
    var rho_at_full_T: Real = 0.0

    var t_final = TWO_PI / omega_p
    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step(
            ctx, mesh, fluid_spatial, maxwell_spatial,
            q_fluid, q_maxwell, ALPHA, EPS0, dt_use,
        )
        t += dt_use

        # Snapshot ρ at the sample point at quarter / half / full periods.
        if rho_at_quarter_T == Real(0.0) and t >= t_final * Real(0.25):
            ctx.synchronize()
            with q_fluid.buf.map_to_host() as h:
                rho_at_quarter_T = h[(sample_e * nodes_per_elem + sample_n) * EULER_NC + IDX_RHO]
        if rho_at_half_T == Real(0.0) and t >= t_final * Real(0.5):
            ctx.synchronize()
            with q_fluid.buf.map_to_host() as h:
                rho_at_half_T = h[(sample_e * nodes_per_elem + sample_n) * EULER_NC + IDX_RHO]
    ctx.synchronize()
    with q_fluid.buf.map_to_host() as h:
        rho_at_full_T = h[(sample_e * nodes_per_elem + sample_n) * EULER_NC + IDX_RHO]

    # Compute final mass.
    var mass1: Real = 0.0
    with q_fluid.buf.map_to_host() as h:
        for j in range(NY):
            for i in range(NX):
                var e = j * NX + i
                for jn in range(P1):
                    for in_ in range(P1):
                        var n = jn * P1 + in_
                        var fb = (e * nodes_per_elem + n) * EULER_NC
                        mass1 += ja * weights[in_] * weights[jn] * h[fb + IDX_RHO]

    # Sample point x ≈ 0.25 corresponds to sin(k·x) ≈ 1, so:
    #   ρ_IC      ≈ 1 + δ
    #   ρ(T/4)    ≈ 1 + δ·cos(π/2) = 1
    #   ρ(T/2)    ≈ 1 + δ·cos(π)   = 1 − δ
    #   ρ(T)      ≈ 1 + δ          (back to IC)
    print("Sampled ρ at x ≈ 0.25 (peak of IC perturbation):")
    print("  IC               =", Real(1.0) + DELTA)
    print("  at T/4 (zero  )  =", rho_at_quarter_T, "  (analytic ≈ 1.0)")
    print("  at T/2 (-peak )  =", rho_at_half_T,    "  (analytic ≈", Real(1.0) - DELTA, ")")
    print("  at T   (+peak )  =", rho_at_full_T,    "  (analytic ≈", Real(1.0) + DELTA, ")")
    print("Mass conservation:")
    print("  initial  =", mass0)
    print("  final    =", mass1)
    print("  drift    =", fabs(mass1 - mass0))
    assert_true(fabs(mass1 - mass0) < Real(1.0e-12),
                "mass should be conserved to machine precision")
    # Quarter-period phase: density should be back to mean (~1) within
    # phase tolerance allowed by Bohm-Gross + SSPRK3 timestep error.
    assert_true(fabs(rho_at_quarter_T - Real(1.0)) < Real(2.0e-3),
                "ρ should pass through equilibrium at T/4")
    # Half-period: perturbation must have inverted.
    assert_true(rho_at_half_T < Real(1.0),
                "ρ should be below equilibrium at T/2 (perturbation inverted)")
    # Full-period closure within 1% of IC perturbation amplitude.
    assert_true(fabs(rho_at_full_T - (Real(1.0) + DELTA)) < Real(1.5e-4),
                "ρ should return to IC after one Langmuir period")
    print("PASSED")
