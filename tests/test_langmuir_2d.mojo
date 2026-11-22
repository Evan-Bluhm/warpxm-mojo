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

from std.collections import Array
from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.plasma import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 4         # uniform-in-y; just enough for the 2D kernel
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = -1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime P0: Real = 1.0e-4
comptime DT: Real = 5.0e-3
comptime FluidState = DGState[2, P, EULER_NC]
comptime MaxwellState = DGState[2, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q_fluid = FluidState(ctx, mesh.num_elements)
    var q_maxwell = MaxwellState(ctx, mesh.num_elements)

    var k_wave = TWO_PI
    var omega_p: Real = 1.0  # ε₀ = 1, α = −1, n₀ = 1

    # E_total = P₀ / (γ−1) + ½·ρ·|u|², |u| = 0 at IC.
    def fluid_ic(x: Real, y: Real) {imm} -> Array[Real, EULER_NC]:
        return [
            Real(1.0) + DELTA * sin(k_wave * x),
            Real(0.0), Real(0.0), Real(0.0),
            P0 / (GAMMA - Real(1.0)),
        ]
    def maxwell_ic(x: Real, y: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = (DELTA / k_wave) * cos(k_wave * x)
        return v^
    mesh.fill(q_fluid, fluid_ic)
    mesh.fill(q_maxwell, maxwell_ic)

    # Initial mass for conservation check.
    var mass0 = mesh.integrate(q_fluid, IDX_RHO)

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver[2, P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver[2, P](maxwell_app)
    var stepper = SSPRK3_TwoFluid[2, P](ctx, mesh.num_elements)

    # Track ρ at one fixed sample point through time to confirm oscillation:
    # first node of the element near x = 0.25 (sin(2π·0.25) = 1, peak).
    var sample = FluidState.index(NX // 4, 0, IDX_RHO)
    var rho_at_quarter_T: Real = 0.0
    var rho_at_half_T: Real = 0.0

    def compute_joint_rhs(
        mut qf_in: FluidState,
        mut qm_in: MaxwellState,
        mut qf_out: FluidState,
        mut qm_out: MaxwellState,
    ) raises {imm}:
        fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source(ctx, qf_in, qm_in, qf_out, ALPHA)
        apply_current_source(ctx, qf_in, qm_out, ALPHA, EPS0)

    var t_final = TWO_PI / omega_p
    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step(ctx, q_fluid, q_maxwell, dt_use, compute_joint_rhs)
        t += dt_use

        # Snapshot ρ at the sample point at quarter / half / full periods.
        if rho_at_quarter_T == Real(0.0) and t >= t_final * Real(0.25):
            ctx.synchronize()
            with q_fluid.buf.map_to_host() as h:
                rho_at_quarter_T = h[sample]
        if rho_at_half_T == Real(0.0) and t >= t_final * Real(0.5):
            ctx.synchronize()
            with q_fluid.buf.map_to_host() as h:
                rho_at_half_T = h[sample]
    ctx.synchronize()
    var rho_at_full_T: Real
    with q_fluid.buf.map_to_host() as h:
        rho_at_full_T = h[sample]

    var mass1 = mesh.integrate(q_fluid, IDX_RHO)

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
