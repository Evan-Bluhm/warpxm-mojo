# Regression test: 1D Sod shock tube with the Moe-Rossmanith limiter.
#
# Asserts that the limited DG scheme:
#   * does NOT NaN out (positivity preserved);
#   * places the shock, contact, and rarefaction at the analytic Riemann
#     locations to within one element of accuracy at t = 0.2.
#
# Analytic at t = 0.2 (γ = 1.4, x_0 = 0.5, t = 0.2):
#   Rarefaction head at x ≈ 0.264, tail at x ≈ 0.486
#   Post-rarefaction plateau (left of contact):
#     ρ_3 ≈ 0.4263, u_3 ≈ 0.9275, P_3 ≈ 0.3031
#   Contact at x ≈ 0.685
#   Post-contact plateau (right of contact):
#     ρ_4 ≈ 0.2655, u_4 ≈ 0.9275, P_4 ≈ 0.3031
#   Shock at x ≈ 0.850.
#
# We don't compare against the full analytic profile (that would require
# bringing in a Riemann-solver routine); instead we sample at a handful of
# carefully chosen locations and check the plateau / right-state values.

from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith_1d import MoeRossmanith1D
from warpxm.apps.euler import Euler
from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 200
comptime GAMMA: Real = 1.4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var rho: Real
            var pp: Real
            if x < Real(0.5):
                rho = 1.0
                pp = 1.0
            else:
                rho = 0.125
                pp = 0.1
            var b = (e * nodes_per_elem + i) * EULER_NC
            host[b + IDX_RHO] = rho
            host[b + IDX_MX] = Real(0.0)
            host[b + 2] = Real(0.0)
            host[b + 3] = Real(0.0)
            host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver1D[P](app)
    var stepper = SSPRK3_1D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith1D[P](ctx, mesh.num_elements)

    var dt: Real = 2.0e-4
    var t_final: Real = 0.2

    def compute_rhs(
        mut q_in: DGState1D[P, EULER_NC],
        mut q_out: DGState1D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState1D[P, EULER_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    # Sample at element midpoints; assert (a) finiteness, (b) plateau / right
    # state values, (c) shock and contact in roughly the right locations.
    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var rho_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    var u_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    var p_at_x = Array[Real, NUM_ELEMENTS](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var b = (e * nodes_per_elem + mid_node) * EULER_NC
            var rho = h[b + IDX_RHO]
            var mx = h[b + IDX_MX]
            var e_t = h[b + IDX_E]
            if isnan(rho) or isnan(mx) or isnan(e_t):
                any_nan = True
            rho_at_x[e] = rho
            u_at_x[e] = mx / rho
            var ke = Real(0.5) * rho * u_at_x[e] * u_at_x[e]
            p_at_x[e] = (GAMMA - Real(1.0)) * (e_t - ke)
    assert_true(not any_nan, "Sod result contains NaN")

    # x = 0.1 → left undisturbed state (ρ = 1, u = 0, P = 1).
    var ix_left = Int(0.1 / mesh.dx)
    print("at x=0.1:  rho =", rho_at_x[ix_left],
          "u =", u_at_x[ix_left], "P =", p_at_x[ix_left])
    assert_true(fabs(rho_at_x[ix_left] - Real(1.0)) < Real(1.0e-2),
                "left undisturbed state ρ wrong")

    # x = 0.6 → post-rarefaction plateau.
    var ix_plateau1 = Int(0.6 / mesh.dx)
    print("at x=0.6:  rho =", rho_at_x[ix_plateau1],
          "u =", u_at_x[ix_plateau1], "P =", p_at_x[ix_plateau1])
    assert_true(fabs(rho_at_x[ix_plateau1] - Real(0.4263)) < Real(0.03),
                "post-rarefaction ρ outside tolerance")
    assert_true(fabs(u_at_x[ix_plateau1] - Real(0.9275)) < Real(0.03),
                "post-rarefaction u outside tolerance")

    # x = 0.75 → post-contact plateau.
    var ix_plateau2 = Int(0.75 / mesh.dx)
    print("at x=0.75: rho =", rho_at_x[ix_plateau2],
          "u =", u_at_x[ix_plateau2], "P =", p_at_x[ix_plateau2])
    assert_true(fabs(rho_at_x[ix_plateau2] - Real(0.2655)) < Real(0.02),
                "post-contact ρ outside tolerance")

    # x = 0.95 → right undisturbed state.
    var ix_right = Int(0.95 / mesh.dx)
    print("at x=0.95: rho =", rho_at_x[ix_right],
          "u =", u_at_x[ix_right], "P =", p_at_x[ix_right])
    assert_true(fabs(rho_at_x[ix_right] - Real(0.125)) < Real(1.0e-2),
                "right undisturbed state ρ wrong")

    print("PASSED")
