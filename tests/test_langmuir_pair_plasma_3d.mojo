# 3D pair-plasma Langmuir oscillation — direct analog of the 2D test.
#
# Two mobile fluid species with opposite charge and equal mass / density.
# IC: antisymmetric density perturbation in x (uniform in y, z), matching
# E_x set up by Gauss's law. Linearised cold-plasma frequency:
#     ω = √(α_1² + α_2²) = √2
# After one period T = 2π/ω the perturbation closes back to IC.

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.euler import Euler
from warpxm.apps.maxwell import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.apps.maxwell import Maxwell
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.spatial.dg_maxwell_3d import DGMaxwellSolver3D
from warpxm.spatial.lorentz_source_3d import apply_lorentz_source_3d
from warpxm.spatial.maxwell_current_source_3d import apply_current_source_3d
from warpxm.temporal.ssprk3_plasma_3d import SSPRK3_Plasma_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 4
comptime NZ: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA_1: Real = -1.0
comptime ALPHA_2: Real = +1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime P0: Real = 1.0e-4
comptime DT: Real = 1.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q_s1 = DGState3D[P, EULER_NC](ctx, mesh.num_elements)
    var q_s2 = DGState3D[P, EULER_NC](ctx, mesh.num_elements)
    var q_maxwell = DGState3D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    var k_wave = TWO_PI
    var omega = sqrt(Real(2.0))
    var t_final = TWO_PI / omega

    var f1h = ctx.enqueue_create_host_buffer[Real.dtype](q_s1.buf_len)
    var f2h = ctx.enqueue_create_host_buffer[Real.dtype](q_s2.buf_len)
    var mh = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var s = sin(k_wave * x)
                            var c_v = cos(k_wave * x)
                            var n = kn * P1 * P1 + jn * P1 + in_

                            var f1_base = (e * nodes_per_elem + n) * EULER_NC
                            f1h[f1_base + IDX_RHO] = Real(1.0) + DELTA * s
                            f1h[f1_base + IDX_MX] = Real(0.0)
                            f1h[f1_base + IDX_MY] = Real(0.0)
                            f1h[f1_base + IDX_MZ] = Real(0.0)
                            f1h[f1_base + IDX_E] = P0 / (GAMMA - Real(1.0))

                            var f2_base = (e * nodes_per_elem + n) * EULER_NC
                            f2h[f2_base + IDX_RHO] = Real(1.0) - DELTA * s
                            f2h[f2_base + IDX_MX] = Real(0.0)
                            f2h[f2_base + IDX_MY] = Real(0.0)
                            f2h[f2_base + IDX_MZ] = Real(0.0)
                            f2h[f2_base + IDX_E] = P0 / (GAMMA - Real(1.0))

                            var mb = (e * nodes_per_elem + n) * MAXWELL_NC
                            mh[mb + IDX_EX] = (Real(2.0) * DELTA / k_wave) * c_v
                            mh[mb + IDX_EY] = Real(0.0)
                            mh[mb + IDX_EZ] = Real(0.0)
                            mh[mb + IDX_BX] = Real(0.0)
                            mh[mb + IDX_BY] = Real(0.0)
                            mh[mb + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q_s1.buf, src_buf=f1h)
    ctx.enqueue_copy(dst_buf=q_s2.buf, src_buf=f2h)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=mh)

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var mass1_0: Real = 0.0
    var mass2_0: Real = 0.0
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var fb = (e * nodes_per_elem + n) * EULER_NC
                            var jw = jv * weights[in_] * weights[jn] * weights[kn]
                            mass1_0 += jw * f1h[fb + IDX_RHO]
                            mass2_0 += jw * f2h[fb + IDX_RHO]

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver3D[P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver3D[P](maxwell_app)
    var stepper = SSPRK3_Plasma_3D[P](ctx, mesh.num_elements)

    @parameter
    def compute_joint_rhs(
        mut q1_in: DGState3D[P, EULER_NC],
        mut q2_in: DGState3D[P, EULER_NC],
        mut qm_in: DGState3D[P, MAXWELL_NC],
        mut q1_out: DGState3D[P, EULER_NC],
        mut q2_out: DGState3D[P, EULER_NC],
        mut qm_out: DGState3D[P, MAXWELL_NC],
    ) raises:
        fluid_spatial.compute_rhs(ctx, mesh, q1_in, q1_out)
        fluid_spatial.compute_rhs(ctx, mesh, q2_in, q2_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source_3d[P](ctx, q1_in, qm_in, q1_out, ALPHA_1)
        apply_lorentz_source_3d[P](ctx, q2_in, qm_in, q2_out, ALPHA_2)
        apply_current_source_3d[P](ctx, q1_in, qm_out, ALPHA_1, EPS0)
        apply_current_source_3d[P](ctx, q2_in, qm_out, ALPHA_2, EPS0)

    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_joint_rhs](ctx, q_s1, q_s2, q_maxwell, dt_use)
        t += dt_use
    ctx.synchronize()

    var sample_e = 0 * NX * NY + 0 * NX + (NX // 4)
    var sample_n = 0
    var b_sample = (sample_e * nodes_per_elem + sample_n) * EULER_NC

    var rho1_T: Real = 0.0
    var rho2_T: Real = 0.0
    var mass1_T: Real = 0.0
    var mass2_T: Real = 0.0
    with q_s1.buf.map_to_host() as h:
        rho1_T = h[b_sample + IDX_RHO]
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var fb = (e * nodes_per_elem + n) * EULER_NC
                                mass1_T += jv * weights[in_] * weights[jn] * weights[kn] * h[fb + IDX_RHO]
    with q_s2.buf.map_to_host() as h:
        rho2_T = h[b_sample + IDX_RHO]
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var fb = (e * nodes_per_elem + n) * EULER_NC
                                mass2_T += jv * weights[in_] * weights[jn] * weights[kn] * h[fb + IDX_RHO]

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
