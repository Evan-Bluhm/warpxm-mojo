# 2D Maxwell plane-wave regression test.
#
# IC: a TM-mode wave with E along ẑ, propagating in (1,1)/√2 direction at
# wavelength λ = 1/√2 (one wavelength on the diagonal of the unit cell).
#
#     E_z(x, y, 0) = sin( 2π (x + y) )
#     B_x(x, y, 0) =  (1/√2) sin( 2π (x + y) )
#     B_y(x, y, 0) = -(1/√2) sin( 2π (x + y) )
#     E_x = E_y = B_z = 0
#
# With c = 1, the period is T = λ/c = 1/√2. After exactly one period the
# solution must return to the IC. We check L2 error against the IC.
#
# This exercises every component of the Maxwell flux tensor (the wave
# couples E_z ↔ B_x ↔ B_y in the curl pattern) and validates that the DG
# kernel's volume + face flux logic handles the linear hyperbolic system
# correctly.

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import (
    Maxwell, MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, TWO_PI
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
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1

    var inv_sqrt2: Real = Real(1.0) / sqrt(Real(2.0))

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
                    var s = sin(TWO_PI * (x + y))
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * MAXWELL_NC
                    host[b + IDX_EX] = Real(0.0)
                    host[b + IDX_EY] = Real(0.0)
                    host[b + IDX_EZ] = s
                    host[b + IDX_BX] = inv_sqrt2 * s
                    host[b + IDX_BY] = -inv_sqrt2 * s
                    host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver2D[P](app)
    var stepper = SSPRK3_2D[P, MAXWELL_NC](ctx, mesh.num_elements)

    # Period: T = λ / c, with λ = 1/√2 → T = 1/√2 ≈ 0.7071.
    var t_final: Real = inv_sqrt2
    var dt: Real = Real(5.0e-4)

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

    # L2 errors per component.
    var err_ez: Real = 0.0
    var err_bx: Real = 0.0
    var err_by: Real = 0.0
    var ja = mesh.jacobian_x * mesh.jacobian_y
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
                        var s = sin(TWO_PI * (x + y))
                        var n = jn * P1 + in_
                        var b = (e * nodes_per_elem + n) * MAXWELL_NC
                        var d_ez = h[b + IDX_EZ] - s
                        var d_bx = h[b + IDX_BX] - inv_sqrt2 * s
                        var d_by = h[b + IDX_BY] - (-inv_sqrt2 * s)
                        var jw = ja * weights[in_] * weights[jn]
                        err_ez += jw * d_ez * d_ez
                        err_bx += jw * d_bx * d_bx
                        err_by += jw * d_by * d_by
    var l2_ez = err_ez ** Real(0.5)
    var l2_bx = err_bx ** Real(0.5)
    var l2_by = err_by ** Real(0.5)
    print("L2 E_z =", l2_ez)
    print("L2 B_x =", l2_bx)
    print("L2 B_y =", l2_by)
    assert_true(l2_ez < Real(5.0e-3), "E_z error too large")
    assert_true(l2_bx < Real(5.0e-3), "B_x error too large")
    assert_true(l2_by < Real(5.0e-3), "B_y error too large")
    print("PASSED")
