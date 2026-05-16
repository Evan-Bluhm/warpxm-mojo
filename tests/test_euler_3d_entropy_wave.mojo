# 3D 5-moment Euler entropy-wave regression.
#
# IC (triply-periodic, smooth):
#     ρ(x, y, z, 0) = 1 + ε · sin(2π (x + y + z))
#     u = (1, 1, 1)   (uniform flow)
#     P = 1           (constant pressure)
#     E = P/(γ-1) + ½ρ|u|²
#
# This is a pure entropy / contact wave: pressure and velocity stay
# constant in space and time, density advects with the flow. After
# t = 1 (one period in each axis) the IC returns exactly.

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.euler import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.euler import Euler
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime GAMMA: Real = 5.0 / 3.0
comptime EPS: Real = 0.01
comptime UX: Real = 1.0
comptime UY: Real = 1.0
comptime UZ: Real = 1.0
comptime DT: Real = 1.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1
    var pp: Real = 1.0

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        var zl = mesh.z_min + Real(k) * mesh.dz
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    var z = zl + (Real(1.0) + ref_nodes[kn]) * mesh.jacobian_z
                    for jn in range(P1):
                        var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var rho = Real(1.0) + EPS * sin(TWO_PI * (x + y + z))
                            var ke = Real(0.5) * rho * (UX * UX + UY * UY + UZ * UZ)
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = rho * UX
                            host[b + IDX_MY] = rho * UY
                            host[b + IDX_MZ] = rho * UZ
                            host[b + IDX_E] = pp / (GAMMA - Real(1.0)) + ke
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver3D[P](app)
    var stepper = SSPRK3_3D[P, EULER_NC](ctx, mesh.num_elements)

    var t_final: Real = 1.0

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, EULER_NC],
        mut q_out: DGState3D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState3D[P, EULER_NC]) raises:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var rho_err: Real = 0.0
    var mass_drift: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            var zl = mesh.z_min + Real(k) * mesh.dz
            for j in range(NY):
                var yl = mesh.y_min + Real(j) * mesh.dy
                for i in range(NX):
                    var xl = mesh.x_min + Real(i) * mesh.dx
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        var z = zl + (Real(1.0) + ref_nodes[kn]) * mesh.jacobian_z
                        for jn in range(P1):
                            var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                            for in_ in range(P1):
                                var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                                var exact = Real(1.0) + EPS * sin(TWO_PI * (x + y + z))
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * EULER_NC
                                var d = h[b + IDX_RHO] - exact
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                rho_err += jw * d * d
                                mass_drift += jw * (h[b + IDX_RHO] - Real(1.0))
    var l2 = rho_err ** Real(0.5)
    print("3D Euler entropy wave — t =", t)
    print("  ρ L2 error vs exact:", l2)
    print("  net mass drift:      ", mass_drift)
    assert_true(l2 < Real(5.0e-4), "ρ L2 error too large")
    assert_true(fabs(mass_drift) < Real(1.0e-12), "mass not conserved")
    print("PASSED")
