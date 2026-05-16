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
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.spatial.lorentz_source_2d import apply_lorentz_source_2d
from warpxm.spatial.maxwell_current_source_2d import apply_current_source_2d
from warpxm.temporal.ssprk3_plasma_2d import SSPRK3_Plasma_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 4
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA_1: Real = -1.0
comptime ALPHA_2: Real = +1.0
comptime EPS0: Real = 1.0
comptime P0: Real = 1.0e-4
comptime DT: Real = 1.0e-3


def _measure_asym(var ctx: DeviceContext, delta: Real) raises -> Real:
    """Run the pair-plasma stepper at the given δ, return the
    antisymmetry residual at t = T_final at the diagnostic sample point.
    """
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0 / 8.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q_s1 = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
    var q_s2 = DGState2D[P, EULER_NC](ctx, mesh.num_elements)
    var q_maxwell = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1
    var k_wave = TWO_PI
    var omega = sqrt(Real(2.0))
    var t_final = TWO_PI / omega

    var f1h = ctx.enqueue_create_host_buffer[Real.dtype](q_s1.buf_len)
    var f2h = ctx.enqueue_create_host_buffer[Real.dtype](q_s2.buf_len)
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
                    var n = jn * P1 + in_

                    var f1_base = (e * nodes_per_elem + n) * EULER_NC
                    f1h[f1_base + IDX_RHO] = Real(1.0) + delta * s
                    f1h[f1_base + IDX_MX] = Real(0.0)
                    f1h[f1_base + IDX_MY] = Real(0.0)
                    f1h[f1_base + IDX_MZ] = Real(0.0)
                    f1h[f1_base + IDX_E] = P0 / (GAMMA - Real(1.0))

                    var f2_base = (e * nodes_per_elem + n) * EULER_NC
                    f2h[f2_base + IDX_RHO] = Real(1.0) - delta * s
                    f2h[f2_base + IDX_MX] = Real(0.0)
                    f2h[f2_base + IDX_MY] = Real(0.0)
                    f2h[f2_base + IDX_MZ] = Real(0.0)
                    f2h[f2_base + IDX_E] = P0 / (GAMMA - Real(1.0))

                    var mb = (e * nodes_per_elem + n) * MAXWELL_NC
                    mh[mb + IDX_EX] = (Real(2.0) * delta / k_wave) * c_v
                    mh[mb + IDX_EY] = Real(0.0)
                    mh[mb + IDX_EZ] = Real(0.0)
                    mh[mb + IDX_BX] = Real(0.0)
                    mh[mb + IDX_BY] = Real(0.0)
                    mh[mb + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q_s1.buf, src_buf=f1h)
    ctx.enqueue_copy(dst_buf=q_s2.buf, src_buf=f2h)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=mh)

    var fluid_app = Euler2D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell2D(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver2D[P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver2D[P](maxwell_app)
    var stepper = SSPRK3_Plasma_2D[P](ctx, mesh.num_elements)

    @parameter
    def compute_joint_rhs(
        mut q1_in: DGState2D[P, EULER_NC],
        mut q2_in: DGState2D[P, EULER_NC],
        mut qm_in: DGState2D[P, MAXWELL_NC],
        mut q1_out: DGState2D[P, EULER_NC],
        mut q2_out: DGState2D[P, EULER_NC],
        mut qm_out: DGState2D[P, MAXWELL_NC],
    ) raises:
        fluid_spatial.compute_rhs(ctx, mesh, q1_in, q1_out)
        fluid_spatial.compute_rhs(ctx, mesh, q2_in, q2_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source_2d[P](ctx, q1_in, qm_in, q1_out, ALPHA_1)
        apply_lorentz_source_2d[P](ctx, q2_in, qm_in, q2_out, ALPHA_2)
        apply_current_source_2d[P](ctx, q1_in, qm_out, ALPHA_1, EPS0)
        apply_current_source_2d[P](ctx, q2_in, qm_out, ALPHA_2, EPS0)

    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_joint_rhs](ctx, q_s1, q_s2, q_maxwell, dt_use)
        t += dt_use
    ctx.synchronize()

    var sample_e = 0 * NX + (NX // 4)
    var b_sample = (sample_e * nodes_per_elem + 0) * EULER_NC

    var rho1_T: Real
    var rho2_T: Real
    with q_s1.buf.map_to_host() as h:
        rho1_T = h[b_sample + IDX_RHO]
    with q_s2.buf.map_to_host() as h:
        rho2_T = h[b_sample + IDX_RHO]

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
