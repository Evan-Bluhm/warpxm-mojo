# 1D symmetric pair-plasma Langmuir oscillation.
#
# Two mobile fluid species with opposite charge, equal mass and equal
# background density. The 1D version of test_langmuir_pair_plasma_2d.
#
# Setup:
#   species 1: α_1 = -1, ρ_1(x, 0) = 1 + δ·sin(kx), u_1 = 0
#   species 2: α_2 = +1, ρ_2(x, 0) = 1 - δ·sin(kx), u_2 = 0
#   ε₀ = 1
#
#   Net charge ρ_q = α_1·ρ_1 + α_2·ρ_2 = -2δ·sin(kx)
#   ∂E_x/∂x = ρ_q/ε_0 → E_x(x) = (2δ/k)·cos(kx)
#
# Plasma frequency:
#   ω² = α_1²·n_1 + α_2²·n_2 = 2 → ω = √2
#   T = 2π/√2
#
# After one period the perturbations return to IC. Mass conservation
# per species to machine epsilon.

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
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs, sqrt
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.spatial.lorentz_source_1d import apply_lorentz_source_1d
from warpxm.spatial.maxwell_current_source_1d import apply_current_source_1d
from warpxm.temporal.ssprk3_plasma_1d import SSPRK3_Plasma_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NX: Int = 32
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
    var mesh = BlockMesh1D[P](ctx, Real(0.0), Real(1.0), NX, periodic=True)
    var q_s1 = DGState1D[P, EULER_NC](ctx, mesh.num_elements)
    var q_s2 = DGState1D[P, EULER_NC](ctx, mesh.num_elements)
    var q_maxwell = DGState1D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var k_wave = TWO_PI
    var omega = sqrt(Real(2.0))
    var t_final = TWO_PI / omega

    var f1h = ctx.enqueue_create_host_buffer[Real.dtype](q_s1.buf_len)
    var f2h = ctx.enqueue_create_host_buffer[Real.dtype](q_s2.buf_len)
    var mh = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var s = sin(k_wave * x)
            var c_v = cos(k_wave * x)

            var f1b = (e * nodes_per_elem + i) * EULER_NC
            f1h[f1b + IDX_RHO] = Real(1.0) + DELTA * s
            f1h[f1b + IDX_MX] = Real(0.0)
            f1h[f1b + IDX_MY] = Real(0.0)
            f1h[f1b + IDX_MZ] = Real(0.0)
            f1h[f1b + IDX_E] = P0 / (GAMMA - Real(1.0))

            var f2b = (e * nodes_per_elem + i) * EULER_NC
            f2h[f2b + IDX_RHO] = Real(1.0) - DELTA * s
            f2h[f2b + IDX_MX] = Real(0.0)
            f2h[f2b + IDX_MY] = Real(0.0)
            f2h[f2b + IDX_MZ] = Real(0.0)
            f2h[f2b + IDX_E] = P0 / (GAMMA - Real(1.0))

            var mb = (e * nodes_per_elem + i) * MAXWELL_NC
            mh[mb + IDX_EX] = (Real(2.0) * DELTA / k_wave) * c_v
            mh[mb + IDX_EY] = Real(0.0)
            mh[mb + IDX_EZ] = Real(0.0)
            mh[mb + IDX_BX] = Real(0.0)
            mh[mb + IDX_BY] = Real(0.0)
            mh[mb + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q_s1.buf, src_buf=f1h)
    ctx.enqueue_copy(dst_buf=q_s2.buf, src_buf=f2h)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=mh)

    var mass1_0: Real = 0.0
    var mass2_0: Real = 0.0
    for e in range(mesh.num_elements):
        for i in range(nodes_per_elem):
            var b = (e * nodes_per_elem + i) * EULER_NC
            var jw = mesh.jacobian * weights[i]
            mass1_0 += jw * f1h[b + IDX_RHO]
            mass2_0 += jw * f2h[b + IDX_RHO]

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver1D[P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver1D[P](maxwell_app)
    var stepper = SSPRK3_Plasma_1D[P](ctx, mesh.num_elements)

    @parameter
    def compute_joint_rhs(
        mut q1_in: DGState1D[P, EULER_NC],
        mut q2_in: DGState1D[P, EULER_NC],
        mut qm_in: DGState1D[P, MAXWELL_NC],
        mut q1_out: DGState1D[P, EULER_NC],
        mut q2_out: DGState1D[P, EULER_NC],
        mut qm_out: DGState1D[P, MAXWELL_NC],
    ) raises:
        fluid_spatial.compute_rhs(ctx, mesh, q1_in, q1_out)
        fluid_spatial.compute_rhs(ctx, mesh, q2_in, q2_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source_1d[P](ctx, q1_in, qm_in, q1_out, ALPHA_1)
        apply_lorentz_source_1d[P](ctx, q2_in, qm_in, q2_out, ALPHA_2)
        apply_current_source_1d[P](ctx, q1_in, qm_out, ALPHA_1, EPS0)
        apply_current_source_1d[P](ctx, q2_in, qm_out, ALPHA_2, EPS0)

    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_joint_rhs](ctx, q_s1, q_s2, q_maxwell, dt_use)
        t += dt_use
    ctx.synchronize()

    var sample_e = NX // 4
    var b_sample = (sample_e * nodes_per_elem + 0) * EULER_NC

    var rho1_T: Real
    var rho2_T: Real
    var mass1_T: Real = 0.0
    var mass2_T: Real = 0.0
    with q_s1.buf.map_to_host() as h:
        rho1_T = h[b_sample + IDX_RHO]
        for e in range(mesh.num_elements):
            for i in range(nodes_per_elem):
                var b = (e * nodes_per_elem + i) * EULER_NC
                mass1_T += mesh.jacobian * weights[i] * h[b + IDX_RHO]
    with q_s2.buf.map_to_host() as h:
        rho2_T = h[b_sample + IDX_RHO]
        for e in range(mesh.num_elements):
            for i in range(nodes_per_elem):
                var b = (e * nodes_per_elem + i) * EULER_NC
                mass2_T += mesh.jacobian * weights[i] * h[b + IDX_RHO]

    var dmass1 = fabs(mass1_T - mass1_0)
    var dmass2 = fabs(mass2_T - mass2_0)
    var asym = fabs((rho1_T - Real(1.0)) - (Real(1.0) - rho2_T))
    print("1D pair-plasma, t = 2π/√2 ≈", t_final, ":")
    print("  ρ_1 at sample (IC = 1.01):", rho1_T)
    print("  ρ_2 at sample (IC = 0.99):", rho2_T)
    print("  species 1 mass drift:", dmass1)
    print("  species 2 mass drift:", dmass2)
    print("  antisymmetry residual (|δρ_1 − (−δρ_2)|):", asym)

    assert_true(dmass1 < Real(1.0e-12), "species 1 mass should be exactly conserved")
    assert_true(dmass2 < Real(1.0e-12), "species 2 mass should be exactly conserved")
    # Both ρ values should be within ~0.15 % of IC after one period.
    assert_true(fabs(rho1_T - (Real(1.0) + DELTA)) < Real(1.5e-3),
                "species 1 amplitude drift exceeded tolerance")
    assert_true(fabs(rho2_T - (Real(1.0) - DELTA)) < Real(1.5e-3),
                "species 2 amplitude drift exceeded tolerance")
    # The antisymmetry residual is O(δ²) (physical) — same as in 2D.
    assert_true(asym < Real(3.0e-3),
                "pair plasma antisymmetry residual too large")
    print("PASSED")
