# 3D cyclotron motion in a uniform B field.
#
# Direct 3D analogue of `test_cyclotron_2d.mojo`. A uniform fluid drifts
# at +x in a uniform B = B_0 ẑ field. Lorentz force u × B = (0, -B_0, 0)
# initially, so the fluid begins accelerating in -y. Circular motion in
# the xy-plane at ω_c = α·B_0.
#
# Maxwell sits still: uniform fields ⇒ ∇×E = ∇×B = 0, and we decouple
# the current source by passing ε₀ = 10³⁰ so J/ε₀ ≈ 0.
#
# After one full period T = 2π:
#   * ρu_x closes back to its IC value (1.0)
#   * ρu_y closes back to 0 (modulo SSPRK3 phase error)
#   * Density / energy / B unchanged at machine precision

from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.euler import Euler
from warpxm.apps.maxwell import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.apps.maxwell import Maxwell
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.spatial.dg_maxwell_3d import DGMaxwellSolver3D
from warpxm.spatial.lorentz_source_3d import apply_lorentz_source_3d
from warpxm.spatial.maxwell_current_source_3d import apply_current_source_3d
from warpxm.temporal.ssprk3_twofluid_3d import SSPRK3_TwoFluid_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 4
comptime NY: Int = 4
comptime NZ: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = 1.0
comptime B0: Real = 1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q_fluid = DGState3D[P, EULER_NC](ctx, mesh.num_elements)
    var q_maxwell = DGState3D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var nodes_per_elem = (P + 1) * (P + 1) * (P + 1)
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
    var fluid_spatial = DGEulerSolver3D[P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver3D[P](maxwell_app)
    var stepper = SSPRK3_TwoFluid_3D[P](ctx, mesh.num_elements)

    var t_final = TWO_PI
    # dt divides the period exactly, so the run ends at t = 2π and the
    # closure error below is the scheme's, not a truncated last step.
    var dt: Real = t_final / Real(1257)
    # ε₀ = 1e30 effectively disables the current back-reaction so the
    # cyclotron test isolates the Lorentz response (matches the 2D test).
    var eps0_decoupled: Real = 1.0e30

    def compute_joint_rhs(
        mut qf_in: DGState3D[P, EULER_NC],
        mut qm_in: DGState3D[P, MAXWELL_NC],
        mut qf_out: DGState3D[P, EULER_NC],
        mut qm_out: DGState3D[P, MAXWELL_NC],
    ) raises {imm}:
        fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source_3d[P](ctx, qf_in, qm_in, qf_out, ALPHA)
        apply_current_source_3d[P](ctx, qf_in, qm_out, ALPHA, eps0_decoupled)

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q_fluid, q_maxwell, dt_use, compute_joint_rhs)
        t += dt_use
    ctx.synchronize()

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

    print("After one 3D cyclotron period (t = 2π):")
    print("  max |Δ(ρu_x)| =", max_d_mx)
    print("  max |Δ(ρu_y)| =", max_d_my)
    print("  max |Δρ|      =", max_d_rho)
    print("  max |ΔE|      =", max_d_e)
    print("  max |ΔB_z|    =", max_d_Bz)

    assert_true(max_d_mx < Real(1.0e-6), "ρu_x didn't close the cyclotron loop")
    assert_true(max_d_my < Real(1.0e-6), "ρu_y residual after one period")
    assert_true(max_d_rho < Real(1.0e-12), "density drifted during cyclotron motion")
    assert_true(max_d_e < Real(1.0e-12), "total energy drifted (magnetic force should do no work)")
    assert_true(max_d_Bz < Real(1.0e-12), "B_z should be untouched")
    print("PASSED")
