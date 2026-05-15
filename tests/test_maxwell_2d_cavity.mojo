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

from std.math import cos, sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell_2d import (
    Maxwell2D, MAXWELL_NC,
    IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_PEC_WALL
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


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,
        bc_bottom=BC_PEC_WALL, bc_top=BC_PEC_WALL,
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
                    host[b + IDX_EX] = Real(0.0)
                    host[b + IDX_EY] = Real(0.0)
                    host[b + IDX_EZ] = sin(PI * x) * sin(PI * y)
                    host[b + IDX_BX] = Real(0.0)
                    host[b + IDX_BY] = Real(0.0)
                    host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Maxwell2D(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    # One period: T = 2π/ω with ω = c·π·√2 ⇒ T = √2/c.
    var t_final = sqrt(Real(2.0)) / C_LIGHT
    var dt: Real = Real(2.0e-4)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, MAXWELL_NC],
        mut q_out: DGState2D[P, MAXWELL_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState2D[P, MAXWELL_NC]) raises:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
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
                        var d_ez = h[b + IDX_EZ] - exact_ez
                        var jw = ja * weights[in_] * weights[jn]
                        err_ez += jw * d_ez * d_ez
                        # E_x, E_y, B_z should be identically zero throughout
                        # the cavity mode — they're never excited.
                        var ax = h[b + IDX_EX]
                        var ay = h[b + IDX_EY]
                        var bz = h[b + IDX_BZ]
                        if ax > max_other: max_other = ax
                        if -ax > max_other: max_other = -ax
                        if ay > max_other: max_other = ay
                        if -ay > max_other: max_other = -ay
                        if bz > max_other: max_other = bz
                        if -bz > max_other: max_other = -bz
    var l2_ez = err_ez ** Real(0.5)
    print("L2 E_z after one period:", l2_ez)
    print("max |E_x, E_y, B_z| (should be tiny):", max_other)
    assert_true(l2_ez < Real(5.0e-3),
                "E_z L2 error too large — PEC wall reflection likely off")
    assert_true(max_other < Real(5.0e-3),
                "non-TM₁₁ components excited — wall BC mixing modes")
    print("PASSED")
