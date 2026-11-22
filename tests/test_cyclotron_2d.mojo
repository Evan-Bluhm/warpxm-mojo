# Cyclotron motion in a uniform B field.
#
# IC: a uniform fluid drifts in the +x direction at speed 1, in a uniform
# magnetic field B = B_0 ẑ. The Maxwell field is initialised so the current
# source on Maxwell vanishes (which it does anyway, since today's stepper
# doesn't apply a current source) AND so its own RHS is zero: spatially
# uniform E and B have zero curl, so neither Maxwell ∂E/∂t nor ∂B/∂t fire
# during the run.
#
#     ρ = 1,  ρu = (1, 0, 0),  P = 1,  E = 0,  B = (0, 0, B_0)
#     α = 1   (charge-to-mass ratio)
#
# Lorentz force on the fluid: F = α·(E + u×B) per unit volume.
# With u = (1,0,0), B = (0,0,B_0): u × B = (0, -B_0, 0). So acceleration is
# initially in the -y direction. The fluid undergoes uniform circular
# motion at cyclotron frequency ω_c = α·B_0.
#
# Test: B_0 = 1 ⇒ ω_c = 1, period T = 2π. After one full period (t = 2π)
# the momentum should be back to (1, 0, 0). Density / energy / B field
# should be unchanged. After t = π/2 (quarter period): momentum (0, -1, 0).

from std.collections import Array
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.plasma import *


comptime P: Int = 2
comptime NX: Int = 4
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = 1.0
comptime B0: Real = 1.0
comptime Fluid = DGState[2, P, EULER_NC]
comptime Field = DGState[2, P, MAXWELL_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q_fluid = Fluid(ctx, mesh.num_elements)
    var q_maxwell = Field(ctx, mesh.num_elements)

    var rho0: Real = 1.0
    var u0: Real = 1.0
    var p0: Real = 1.0
    var ke0 = Real(0.5) * rho0 * u0 * u0
    var e0 = p0 / (GAMMA - Real(1.0)) + ke0

    def fluid_ic(x: Real, y: Real) {imm} -> Array[Real, EULER_NC]:
        return [rho0, rho0 * u0, Real(0.0), Real(0.0), e0]
    mesh.fill(q_fluid, fluid_ic)

    def field_ic(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
        return [Real(0.0), Real(0.0), Real(0.0), Real(0.0), Real(0.0), B0]
    mesh.fill(q_maxwell, field_ic)

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver[2, P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver[2, P](maxwell_app)
    var stepper = SSPRK3_TwoFluid[2, P](ctx, mesh.num_elements)

    # Integrate one full cyclotron period T = 2π / ω_c = 2π.
    var t_final = TWO_PI
    # Decouple the back-reaction by making ε₀ effectively infinite — the
    # cyclotron test is about Lorentz only; the current source would let
    # E grow from the fluid's drift and break the "Maxwell stays put"
    # part of the test.
    var eps0_decoupled: Real = 1.0e30

    def compute_joint_rhs(
        mut qf_in: Fluid, mut qm_in: Field, mut qf_out: Fluid, mut qm_out: Field,
    ) raises {imm}:
        fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source(ctx, qf_in, qm_in, qf_out, ALPHA)
        apply_current_source(ctx, qf_in, qm_out, ALPHA, eps0_decoupled)

    # dt divides the period exactly, so the run ends at t = 2π and the
    # closure error below is the scheme's, not a truncated last step.
    var dt: Real = t_final / Real(1257)
    _ = stepper.run_until(
        ctx, q_fluid, q_maxwell, Real(0.0), t_final, dt, compute_joint_rhs,
    )
    ctx.synchronize()

    # After one period, ρu should be back to (1, 0, 0). Density / energy
    # / B-field components should be unchanged.
    var max_d_mx: Real = 0.0
    var max_d_my: Real = 0.0
    var max_d_rho: Real = 0.0
    var max_d_e: Real = 0.0
    var max_d_Bz: Real = 0.0
    def fluid_drift(
        x: Real, y: Real, w: Real, v: Array[Real, EULER_NC],
    ) {mut max_d_mx, mut max_d_my, mut max_d_rho, mut max_d_e, imm}:
        max_d_mx = fmax(max_d_mx, fabs(v[IDX_MX] - u0))
        max_d_my = fmax(max_d_my, fabs(v[IDX_MY]))
        max_d_rho = fmax(max_d_rho, fabs(v[IDX_RHO] - rho0))
        max_d_e = fmax(max_d_e, fabs(v[IDX_E] - e0))
    mesh.visit_nodes(q_fluid, fluid_drift)
    def field_drift(
        x: Real, y: Real, w: Real, v: Array[Real, MAXWELL_NC],
    ) {mut max_d_Bz}:
        max_d_Bz = fmax(max_d_Bz, fabs(v[IDX_BZ] - B0))
    mesh.visit_nodes(q_maxwell, field_drift)

    print("After one cyclotron period (t = 2π):")
    print("  max |Δ(ρu_x)| =", max_d_mx)
    print("  max |Δ(ρu_y)| =", max_d_my)
    print("  max |Δρ|      =", max_d_rho)
    print("  max |ΔE|      =", max_d_e)
    print("  max |ΔB_z|    =", max_d_Bz)
    # SSPRK3 is third-order in time; 1257 steps of dt = 2π/1257 ≈ 5e-3 close
    # the orbit to ~3e-8, so 1e-6 leaves a 30× margin while still catching a
    # wrong gyro-frequency or a first-order source coupling.
    assert_true(max_d_mx < Real(1.0e-6), "ρu_x didn't close the cyclotron loop")
    assert_true(max_d_my < Real(1.0e-6), "ρu_y residual after one period")
    assert_true(max_d_rho < Real(1.0e-12), "density drifted during cyclotron motion")
    assert_true(max_d_e < Real(1.0e-12), "total energy drifted (magnetic force should do no work)")
    assert_true(max_d_Bz < Real(1.0e-12), "B_z should be untouched by spatial solver")
    print("PASSED")
