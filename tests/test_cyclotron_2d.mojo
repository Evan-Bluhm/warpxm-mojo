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

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.spatial.lorentz_source_2d import apply_lorentz_source_2d
from warpxm.spatial.maxwell_current_source_2d import apply_current_source_2d
from warpxm.temporal.ssprk3_twofluid_2d import SSPRK3_TwoFluid_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 4
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = 1.0
comptime B0: Real = 1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q_fluid = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
    var q_maxwell = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var nodes_per_elem = (P + 1) * (P + 1)
    var rho0: Real = 1.0
    var u0: Real = 1.0
    var p0: Real = 1.0
    var ke0 = Real(0.5) * rho0 * u0 * u0
    var e0 = p0 / (GAMMA - Real(1.0)) + ke0

    var f_host = ctx.enqueue_create_host_buffer[Real.dtype](q_fluid.buf_len)
    for k in range(mesh.num_elements * nodes_per_elem):
        var b = k * EULER_NC
        f_host[b + IDX_RHO] = rho0
        f_host[b + IDX_MX] = rho0 * u0
        f_host[b + IDX_MY] = Real(0.0)
        f_host[b + IDX_MZ] = Real(0.0)
        f_host[b + IDX_E] = e0
    ctx.enqueue_copy(dst_buf=q_fluid.buf, src_buf=f_host)

    var m_host = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for k in range(mesh.num_elements * nodes_per_elem):
        var b = k * MAXWELL_NC
        m_host[b + IDX_EX] = Real(0.0)
        m_host[b + IDX_EY] = Real(0.0)
        m_host[b + IDX_EZ] = Real(0.0)
        m_host[b + IDX_BX] = Real(0.0)
        m_host[b + IDX_BY] = Real(0.0)
        m_host[b + IDX_BZ] = B0
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=m_host)

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver2D[P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver2D[P](maxwell_app)
    var stepper = SSPRK3_TwoFluid_2D[P](ctx, mesh.num_elements)

    # Integrate one full cyclotron period T = 2π / ω_c = 2π.
    var t_final = TWO_PI
    # Decouple the back-reaction by making ε₀ effectively infinite — the
    # cyclotron test is about Lorentz only; the current source would let
    # E grow from the fluid's drift and break the "Maxwell stays put"
    # part of the test.
    var eps0_decoupled: Real = 1.0e30

    @parameter
    def compute_joint_rhs(
        mut qf_in: DGState2D[P, EULER_NC],
        mut qm_in: DGState2D[P, MAXWELL_NC],
        mut qf_out: DGState2D[P, EULER_NC],
        mut qm_out: DGState2D[P, MAXWELL_NC],
    ) raises:
        fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source_2d[P](ctx, qf_in, qm_in, qf_out, ALPHA)
        apply_current_source_2d[P](ctx, qf_in, qm_out, ALPHA, eps0_decoupled)

    var dt: Real = Real(1.0e-3)
    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step[compute_joint_rhs](ctx, q_fluid, q_maxwell, dt_use)
        t += dt_use
    ctx.synchronize()

    # After one period, ρu should be back to (1, 0, 0). Density / energy
    # / B-field components should be unchanged.
    var max_d_mx: Real = 0.0
    var max_d_my: Real = 0.0
    var max_d_rho: Real = 0.0
    var max_d_e: Real = 0.0
    var max_d_Bz: Real = 0.0
    with q_fluid.buf.map_to_host() as h:
        for k in range(mesh.num_elements * nodes_per_elem):
            var b = k * EULER_NC
            var d_mx = fabs(h[b + IDX_MX] - u0)
            var d_my = fabs(h[b + IDX_MY])
            var d_rho = fabs(h[b + IDX_RHO] - rho0)
            var d_e = fabs(h[b + IDX_E] - e0)
            if d_mx > max_d_mx: max_d_mx = d_mx
            if d_my > max_d_my: max_d_my = d_my
            if d_rho > max_d_rho: max_d_rho = d_rho
            if d_e > max_d_e: max_d_e = d_e
    with q_maxwell.buf.map_to_host() as h:
        for k in range(mesh.num_elements * nodes_per_elem):
            var b = k * MAXWELL_NC
            var d = fabs(h[b + IDX_BZ] - B0)
            if d > max_d_Bz: max_d_Bz = d

    print("After one cyclotron period (t = 2π):")
    print("  max |Δ(ρu_x)| =", max_d_mx)
    print("  max |Δ(ρu_y)| =", max_d_my)
    print("  max |Δρ|      =", max_d_rho)
    print("  max |ΔE|      =", max_d_e)
    print("  max |ΔB_z|    =", max_d_Bz)
    # SSPRK3 is third-order in time, dt = 1e-3 over T = 2π ⇒ ~6300 steps.
    # Per-step error ~ dt^4 ≈ 1e-12; accumulated ~ dt^3 ≈ 1e-9.
    assert_true(max_d_mx < Real(1.0e-3), "ρu_x didn't close the cyclotron loop")
    assert_true(max_d_my < Real(1.0e-3), "ρu_y residual after one period")
    assert_true(max_d_rho < Real(1.0e-12), "density drifted during cyclotron motion")
    assert_true(max_d_e < Real(1.0e-12), "total energy drifted (magnetic force should do no work)")
    assert_true(max_d_Bz < Real(1.0e-12), "B_z should be untouched by spatial solver")
    print("PASSED")
