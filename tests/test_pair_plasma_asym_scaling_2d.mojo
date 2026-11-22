# Diagnostic — empirical δ-scaling of the pair-plasma antisymmetry
# residual.
#
# Background: test_langmuir_pair_plasma_2d.mojo reports a residual of
# 1.85e-3 with δ = 0.01. Its docstring attributes this to "first-order
# splitting" (species-1 Lorentz before species-2 within each RK stage).
# But for symmetric pair plasma with mobile + opposite-charge species,
# the *continuous* dynamics is only antisymmetric at LINEAR order in δ —
# at O(δ²) the ρ-dependent Lorentz force (F = α·ρ·E) is already
# antisymmetry-breaking because species-1 has ρ = 1 + δ·sin while
# species-2 has ρ = 1 − δ·sin. That's a physical effect that no
# operator-splitting fix can eliminate.
#
# This test runs the pair-plasma stepper at three values of δ and
# reports the asymmetry residual at t = T_final. If the scaling is
# O(δ²), the residual ratio between δ=0.01 and δ=0.001 should be ≈ 100;
# if it's O(δ·dt) (an algorithmic asymmetry), the ratio should be ≈ 10.

from std.collections import Array
from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.plasma import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 2
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA_1: Real = -1.0
comptime ALPHA_2: Real = +1.0
comptime EPS0: Real = 1.0
comptime P0: Real = 1.0e-4
comptime DT: Real = 2.5e-3
comptime FluidState = DGState[2, P, EULER_NC]
comptime MaxwellState = DGState[2, P, MAXWELL_NC]


def _measure_asym(var ctx: DeviceContext, delta: Real) raises -> Real:
    """Run the pair-plasma stepper at the given δ, return the
    antisymmetry residual at t = T_final at the diagnostic sample point.
    """
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q_s1 = FluidState(ctx, mesh.num_elements)
    var q_s2 = FluidState(ctx, mesh.num_elements)
    var q_maxwell = MaxwellState(ctx, mesh.num_elements)

    var k_wave = TWO_PI
    var omega = sqrt(Real(2.0))
    var t_final = TWO_PI / omega

    # Opposite density perturbations on the two species, both at rest.
    def species_1(x: Real, y: Real) {imm} -> Array[Real, EULER_NC]:
        return [
            Real(1.0) + delta * sin(k_wave * x),
            Real(0.0), Real(0.0), Real(0.0),
            P0 / (GAMMA - Real(1.0)),
        ]
    def species_2(x: Real, y: Real) {imm} -> Array[Real, EULER_NC]:
        return [
            Real(1.0) - delta * sin(k_wave * x),
            Real(0.0), Real(0.0), Real(0.0),
            P0 / (GAMMA - Real(1.0)),
        ]
    def maxwell_ic(x: Real, y: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = (Real(2.0) * delta / k_wave) * cos(k_wave * x)
        return v^
    mesh.fill(q_s1, species_1)
    mesh.fill(q_s2, species_2)
    mesh.fill(q_maxwell, maxwell_ic)

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver[2, P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver[2, P](maxwell_app)
    var stepper = SSPRK3_Plasma[2, P](ctx, mesh.num_elements)

    def compute_joint_rhs(
        mut q1_in: FluidState,
        mut q2_in: FluidState,
        mut qm_in: MaxwellState,
        mut q1_out: FluidState,
        mut q2_out: FluidState,
        mut qm_out: MaxwellState,
    ) raises {imm}:
        fluid_spatial.compute_rhs(ctx, mesh, q1_in, q1_out)
        fluid_spatial.compute_rhs(ctx, mesh, q2_in, q2_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source(ctx, q1_in, qm_in, q1_out, ALPHA_1)
        apply_lorentz_source(ctx, q2_in, qm_in, q2_out, ALPHA_2)
        apply_current_source(ctx, q1_in, qm_out, ALPHA_1, EPS0)
        apply_current_source(ctx, q2_in, qm_out, ALPHA_2, EPS0)

    # T / DT is not an integer; the run stops at the last whole step
    # before T (within half a step of it).
    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step(ctx, q_s1, q_s2, q_maxwell, dt_use, compute_joint_rhs)
        t += dt_use
    ctx.synchronize()

    # First node of the element at x ≈ 0.25 in the bottom row.
    var sample = FluidState.index(NX // 4, 0, IDX_RHO)
    var rho1_T: Real
    var rho2_T: Real
    with q_s1.buf.map_to_host() as h:
        rho1_T = h[sample]
    with q_s2.buf.map_to_host() as h:
        rho2_T = h[sample]

    return fabs((rho1_T - Real(1.0)) - (Real(1.0) - rho2_T))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()

    var asym_1 = _measure_asym(ctx, Real(0.01))
    var asym_2 = _measure_asym(ctx, Real(0.001))
    var asym_3 = _measure_asym(ctx, Real(0.0001))

    print("Pair-plasma antisymmetry residual vs δ:")
    print("  δ = 0.01   : asym =", asym_1)
    print("  δ = 0.001  : asym =", asym_2)
    print("  δ = 0.0001 : asym =", asym_3)
    print("")
    print("Ratios (expect ~100 if O(δ²), ~10 if O(δ¹)):")
    var ratio_1 = asym_1 / asym_2
    var ratio_2 = asym_2 / asym_3
    print("  asym(0.01)  / asym(0.001) :", ratio_1)
    print("  asym(0.001) / asym(0.0001):", ratio_2)

    # Both ratios should be ~100 — the residual is the physical
    # O(δ²) Lorentz-force coupling between species, not an algorithmic
    # asymmetry. If a future change introduces a stepper-level (i.e.
    # O(δ¹·dt) per step) asymmetry, the ratios drop toward 10 and this
    # assertion catches it.
    assert_true(ratio_1 > Real(80.0) and ratio_1 < Real(120.0),
                "δ=0.01/0.001 ratio outside O(δ²) range — possible algorithmic asymmetry")
    assert_true(ratio_2 > Real(80.0) and ratio_2 < Real(120.0),
                "δ=0.001/0.0001 ratio outside O(δ²) range — possible algorithmic asymmetry")
    print("PASSED")
