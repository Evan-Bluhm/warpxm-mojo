# 3D pair-plasma Langmuir oscillation — direct analog of the 2D test.
#
# Two mobile fluid species with opposite charge and equal mass / density.
# IC: antisymmetric density perturbation in x (uniform in y, z), matching
# E_x set up by Gauss's law. Linearised cold-plasma frequency:
#     ω = √(α_1² + α_2²) = √2
# After one period T = 2π/ω the perturbation closes back to IC.

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
comptime NZ: Int = 2
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA_1: Real = -1.0
comptime ALPHA_2: Real = +1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime P0: Real = 1.0e-4
comptime DT: Real = 2.5e-3
comptime FluidState = DGState[3, P, EULER_NC]
comptime MaxwellState = DGState[3, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q_s1 = FluidState(ctx, mesh.num_elements)
    var q_s2 = FluidState(ctx, mesh.num_elements)
    var q_maxwell = MaxwellState(ctx, mesh.num_elements)

    var k_wave = TWO_PI
    var omega = sqrt(Real(2.0))
    var t_final = TWO_PI / omega

    # Opposite density perturbations on the two species, both at rest.
    def species_1(x: Real, y: Real, z: Real) {imm} -> Array[Real, EULER_NC]:
        return [
            Real(1.0) + DELTA * sin(k_wave * x),
            Real(0.0), Real(0.0), Real(0.0),
            P0 / (GAMMA - Real(1.0)),
        ]
    def species_2(x: Real, y: Real, z: Real) {imm} -> Array[Real, EULER_NC]:
        return [
            Real(1.0) - DELTA * sin(k_wave * x),
            Real(0.0), Real(0.0), Real(0.0),
            P0 / (GAMMA - Real(1.0)),
        ]
    def maxwell_ic(x: Real, y: Real, z: Real) {imm} -> Array[Real, MAXWELL_NC]:
        var v = Array[Real, MAXWELL_NC](fill=Real(0.0))
        v[IDX_EX] = (Real(2.0) * DELTA / k_wave) * cos(k_wave * x)
        return v^
    mesh.fill(q_s1, species_1)
    mesh.fill(q_s2, species_2)
    mesh.fill(q_maxwell, maxwell_ic)

    var mass1_0 = mesh.integrate(q_s1, IDX_RHO)
    var mass2_0 = mesh.integrate(q_s2, IDX_RHO)

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver[3, P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver[3, P](maxwell_app)
    var stepper = SSPRK3_Plasma[3, P](ctx, mesh.num_elements)

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

    # First node of the element at x ≈ 0.25 in the first (y, z) row.
    var sample = FluidState.index(NX // 4, 0, IDX_RHO)
    var rho1_T: Real
    var rho2_T: Real
    with q_s1.buf.map_to_host() as h:
        rho1_T = h[sample]
    with q_s2.buf.map_to_host() as h:
        rho2_T = h[sample]
    var mass1_T = mesh.integrate(q_s1, IDX_RHO)
    var mass2_T = mesh.integrate(q_s2, IDX_RHO)

    var dmass1 = fabs(mass1_T - mass1_0)
    var dmass2 = fabs(mass2_T - mass2_0)
    print("3D pair-plasma Langmuir — t =", t_final)
    print("  species 1 ρ at sample (IC = 1.01):", rho1_T)
    print("  species 2 ρ at sample (IC = 0.99):", rho2_T)
    print("  species 1 mass drift:", dmass1)
    print("  species 2 mass drift:", dmass2)

    assert_true(dmass1 < Real(1.0e-12), "species 1 mass should be exactly conserved")
    assert_true(dmass2 < Real(1.0e-12), "species 2 mass should be exactly conserved")
    # Tolerances mirror the 2D pair-plasma test — the first-order splitting
    # leaves O(dt·δ) asymmetry between species.
    var amp_drift_1 = fabs(rho1_T - (Real(1.0) + DELTA))
    var amp_drift_2 = fabs(rho2_T - (Real(1.0) - DELTA))
    assert_true(amp_drift_1 < Real(1.5e-3), "species 1 amplitude drift")
    assert_true(amp_drift_2 < Real(1.5e-3), "species 2 amplitude drift")
    print("PASSED")
