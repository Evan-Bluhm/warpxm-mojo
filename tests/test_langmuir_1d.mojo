# 1D Langmuir oscillation — the textbook plasma test.
#
# A single mobile electron fluid against a static positive-ion
# background. Apply a small density perturbation; the resulting space
# charge drives an E_x field that oscillates the electrons at the
# plasma frequency ω_pe = √(α²·n_0/ε_0).
#
# Setup:
#   α    = -1 (electrons)
#   n_0  = 1 (background ion density)
#   ρ(x) = 1 + δ·sin(2π x)   ← small electron density perturbation
#   ρ_q(x) = α·ρ + n_0       (net charge density)
#          = -δ·sin(2π x)
#   ∂E_x/∂x = ρ_q/ε_0 = -δ·sin/ε_0
#       ⇒ E_x(x) = (δ/(k·ε_0))·cos(2π x)
#
# Plasma frequency:
#   ω_pe² = α²·n_0/ε_0 = 1
#   T = 2π/ω = 2π
#
# After one period the perturbation returns to IC. Mass conservation is
# automatic by construction; the test asserts:
#   (a) species mass drift to machine epsilon,
#   (b) ρ returns to its IC at the sample point.

from std.collections import Array
from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.plasma import *


comptime P: Int = 2
comptime NX: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = -1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime P0: Real = 1.0e-4
comptime DT: Real = 2.5e-3
comptime FluidState = DGState[1, P, EULER_NC]
comptime MaxwellState = DGState[1, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](ctx, Real(0.0), Real(1.0), NX, periodic=True)
    var q_fluid = FluidState(ctx, mesh.num_elements)
    var q_maxwell = MaxwellState(ctx, mesh.num_elements)

    var k_wave = TWO_PI
    var omega_p: Real = 1.0      # √(α²·n_0/ε_0) = 1

    def fluid_ic(x: Real) {imm} -> Array[Real, EULER_NC]:
        return [
            Real(1.0) + DELTA * sin(k_wave * x),
            Real(0.0), Real(0.0), Real(0.0),
            P0 / (GAMMA - Real(1.0)),
        ]
    # Net charge density ρ_q = -δ·sin (electrons exceed ions where ρ_e is high)
    # ∂E_x/∂x = ρ_q/ε_0 = -δ·sin/ε_0
    # E_x(x) = (δ/(k·ε_0))·cos(2π x)
    def maxwell_ic(x: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = (DELTA / (k_wave * EPS0)) * cos(k_wave * x)
        return v^
    mesh.fill(q_fluid, fluid_ic)
    mesh.fill(q_maxwell, maxwell_ic)

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver[1, P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver[1, P](maxwell_app)
    var stepper = SSPRK3_TwoFluid[1, P](ctx, mesh.num_elements)

    var mass_0 = mesh.integrate(q_fluid, IDX_RHO)

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

    # One Langmuir period. T / DT is not an integer; the run stops at the
    # last whole step before T (within half a step of it).
    var t_final = TWO_PI / omega_p
    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step(ctx, q_fluid, q_maxwell, dt_use, compute_joint_rhs)
        t += dt_use
    ctx.synchronize()

    # Sample at x ≈ 0.25 (sin(2π·0.25) = 1 — IC peak).
    var rho_T: Real
    with q_fluid.buf.map_to_host() as h:
        rho_T = h[FluidState.index(NX // 4, 0, IDX_RHO)]
    var mass_T = mesh.integrate(q_fluid, IDX_RHO)

    var dmass = fabs(mass_T - mass_0)
    print("1D Langmuir, t = 2π/ω_pe ≈", t_final, ":")
    print("  ρ at sample (IC = 1.01):", rho_T)
    print("  mass drift:               ", dmass)

    assert_true(dmass < Real(1.0e-12), "mass should be exactly conserved")
    # Closure tolerance: ρ should be within ~0.1% of IC after one period.
    assert_true(fabs(rho_T - (Real(1.0) + DELTA)) < Real(1.5e-4),
                "ρ at sample point drifted too far from IC")
    print("PASSED")
