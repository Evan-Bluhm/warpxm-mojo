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

from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.spatial.dg_maxwell_1d import DGMaxwellSolver1D
from warpxm.spatial.lorentz_source_1d import apply_lorentz_source_1d
from warpxm.spatial.maxwell_current_source_1d import apply_current_source_1d
from warpxm.temporal.ssprk3_twofluid_1d import SSPRK3_TwoFluid_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NX: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime ALPHA: Real = -1.0
comptime EPS0: Real = 1.0
comptime DELTA: Real = 0.01
comptime P0: Real = 1.0e-4
comptime DT: Real = 2.5e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](ctx, Real(0.0), Real(1.0), NX, periodic=True)
    var q_fluid = DGState1D[P, EULER_NC](ctx, mesh.num_elements)
    var q_maxwell = DGState1D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var k_wave = TWO_PI
    var omega_p: Real = 1.0      # √(α²·n_0/ε_0) = 1

    var fh = ctx.enqueue_create_host_buffer[Real.dtype](q_fluid.buf_len)
    var mh = ctx.enqueue_create_host_buffer[Real.dtype](q_maxwell.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var s = sin(k_wave * x)
            var c_v = cos(k_wave * x)

            var fb = (e * nodes_per_elem + i) * EULER_NC
            fh[fb + IDX_RHO] = Real(1.0) + DELTA * s
            fh[fb + IDX_MX] = Real(0.0)
            fh[fb + IDX_MY] = Real(0.0)
            fh[fb + IDX_MZ] = Real(0.0)
            fh[fb + IDX_E] = P0 / (GAMMA - Real(1.0))

            var mb = (e * nodes_per_elem + i) * MAXWELL_NC
            # Net charge density ρ_q = -δ·sin (electrons exceed ions where ρ_e is high)
            # ∂E_x/∂x = ρ_q/ε_0 = -δ·sin/ε_0
            # E_x(x) = (δ/(k·ε_0))·cos(2π x)
            mh[mb + IDX_EX] = (DELTA / (k_wave * EPS0)) * c_v
            mh[mb + IDX_EY] = Real(0.0)
            mh[mb + IDX_EZ] = Real(0.0)
            mh[mb + IDX_BX] = Real(0.0)
            mh[mb + IDX_BY] = Real(0.0)
            mh[mb + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q_fluid.buf, src_buf=fh)
    ctx.enqueue_copy(dst_buf=q_maxwell.buf, src_buf=mh)

    var fluid_app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-15), cfl_max=Real(0.5),
    )
    var maxwell_app = Maxwell(c_light=Real(1.0), cfl_max=Real(0.5))
    var fluid_spatial = DGEulerSolver1D[P](fluid_app)
    var maxwell_spatial = DGMaxwellSolver1D[P](maxwell_app)
    var stepper = SSPRK3_TwoFluid_1D[P](ctx, mesh.num_elements)

    # Initial mass.
    var mass_0: Real = 0.0
    for e in range(mesh.num_elements):
        for i in range(nodes_per_elem):
            var b = (e * nodes_per_elem + i) * EULER_NC
            mass_0 += mesh.jacobian * weights[i] * fh[b + IDX_RHO]

    def compute_joint_rhs(
        mut qf_in: DGState1D[P, EULER_NC],
        mut qm_in: DGState1D[P, MAXWELL_NC],
        mut qf_out: DGState1D[P, EULER_NC],
        mut qm_out: DGState1D[P, MAXWELL_NC],
    ) raises {imm}:
        fluid_spatial.compute_rhs(ctx, mesh, qf_in, qf_out)
        maxwell_spatial.compute_rhs(ctx, mesh, qm_in, qm_out)
        apply_lorentz_source_1d[P](ctx, qf_in, qm_in, qf_out, ALPHA)
        apply_current_source_1d[P](ctx, qf_in, qm_out, ALPHA, EPS0)

    var t_final = TWO_PI / omega_p     # one Langmuir period
    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step(ctx, q_fluid, q_maxwell, dt_use, compute_joint_rhs)
        t += dt_use
    ctx.synchronize()

    # Sample at x ≈ 0.25 (sin(2π·0.25) = 1 — IC peak).
    var sample_e = NX // 4
    var b_sample = (sample_e * nodes_per_elem + 0) * EULER_NC

    var rho_T: Real
    var mass_T: Real = 0.0
    with q_fluid.buf.map_to_host() as h:
        rho_T = h[b_sample + IDX_RHO]
        for e in range(mesh.num_elements):
            for i in range(nodes_per_elem):
                var b = (e * nodes_per_elem + i) * EULER_NC
                mass_T += mesh.jacobian * weights[i] * h[b + IDX_RHO]

    var dmass = fabs(mass_T - mass_0)
    print("1D Langmuir, t = 2π/ω_pe ≈", t_final, ":")
    print("  ρ at sample (IC = 1.01):", rho_T)
    print("  mass drift:               ", dmass)

    assert_true(dmass < Real(1.0e-12), "mass should be exactly conserved")
    # Closure tolerance: ρ should be within ~0.1% of IC after one period.
    assert_true(fabs(rho_T - (Real(1.0) + DELTA)) < Real(1.5e-4),
                "ρ at sample point drifted too far from IC")
    print("PASSED")
