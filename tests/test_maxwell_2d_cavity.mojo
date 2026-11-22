# 2D rectangular cavity TM₁₁ mode regression test.
#
# Domain `[0, 1]²` with PEC walls on all four sides. The exact TM₁₁ standing-
# wave solution is
#
#     E_z(x, y, t) = sin(π x) · sin(π y) · cos(ω t)
#     B_x(x, y, t) =  (π / ω) · sin(π x) · cos(π y) · sin(ω t)
#     B_y(x, y, t) = -(π / ω) · cos(π x) · sin(π y) · sin(ω t)
#
# with `ω = c · π · √2`. After one full period `T = 2π / ω = √2 / c` the
# fields return to the IC.
#
# At the PEC walls every boundary condition is satisfied identically:
#   * tangential E (here E_z and either E_x or E_y) is zero by the sin terms
#     vanishing at x ∈ {0, 1} or y ∈ {0, 1};
#   * normal B (B_x at x-walls, B_y at y-walls) is also zero because the
#     other sin factor pins it.
#
# We pick t_final = √2 (= one period at c = 1) and check the L2 error vs IC.
#
# A second run covers `BC_PMC_WALL` with the dual problem (E -> cB,
# cB -> -E): the same profile on B_z between PMC walls, where E_z, B_x and
# B_y are the components that must stay zero.

from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_PEC_WALL, BC_PMC_WALL
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_maxwell_2d import DGMaxwellSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime C_LIGHT: Real = 1.0


def _run_cavity(var ctx: DeviceContext, pmc: Bool) raises:
    """One period of the cavity mode. `pmc=False`: PEC walls, E_z mode.
    `pmc=True`: the dual problem (E -> cB, cB -> -E) — PMC walls with the
    same profile on B_z, so E_z, B_x, B_y are the unexcited components."""
    var bc = BC_PMC_WALL if pmc else BC_PEC_WALL
    var i_mode = IDX_BZ if pmc else IDX_EZ
    var i_zero0 = IDX_BX if pmc else IDX_EX
    var i_zero1 = IDX_BY if pmc else IDX_EY
    var i_zero2 = IDX_EZ if pmc else IDX_BZ
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=bc, bc_right=bc,
        bc_bottom=bc, bc_top=bc,
    )
    var q = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    # IC: only E_z, at peak of the cos(ωt) factor.
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MAXWELL_NC
                    for c in range(MAXWELL_NC):
                        host[b + c] = Real(0.0)
                    host[b + i_mode] = sin(PI * x) * sin(PI * y)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    # One period: T = 2π/ω with ω = c·π·√2 ⇒ T = √2/c.
    var t_final = sqrt(Real(2.0)) / C_LIGHT
    # dt divides the period exactly (≈ 5e-4, the step the plane-wave tests
    # use on the same mesh), so the run ends exactly at t = T.
    var dt: Real = t_final / Real(2829)

    def compute_rhs(
        mut q_in: DGState2D[P, MAXWELL_NC],
        mut q_out: DGState2D[P, MAXWELL_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState2D[P, MAXWELL_NC]) raises {imm}:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
    ctx.synchronize()

    var ja = mesh.jacobian_x * mesh.jacobian_y
    var err_ez: Real = 0.0
    var max_other: Real = 0.0
    with q.buf.map_to_host() as h:
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = j * NX + i
                for jn in range(P1):
                    var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                    for in_ in range(P1):
                        var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MAXWELL_NC
                        var exact_ez = sin(PI * x) * sin(PI * y)
                        var d_ez = h[b + i_mode] - exact_ez
                        var jw = ja * weights[in_] * weights[jn]
                        err_ez += jw * d_ez * d_ez
                        # The three components outside the mode are never excited.
                        for idx in [i_zero0, i_zero1, i_zero2]:
                            var v = h[b + idx]
                            if v > max_other: max_other = v
                            if -v > max_other: max_other = -v
    var l2_ez = err_ez ** Real(0.5)
    print("PMC walls, B_z mode:" if pmc else "PEC walls, E_z mode:")
    print("  L2 mode error after one period:", l2_ez)
    print("  max |unexcited components| (should be tiny):", max_other)
    assert_true(l2_ez < Real(2.5e-5),
                "mode L2 error too large — wall reflection likely off")
    assert_true(max_other < Real(1.0e-10),
                "unexcited components grew — wall BC mixing modes")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _run_cavity(ctx, False)
    _run_cavity(ctx, True)
    print("PASSED")
