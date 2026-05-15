# 3D Maxwell plane-wave regression.
#
# IC: a TM wave propagating along x̂ with E in ẑ, B in ŷ:
#     E_z(x, y, z, 0) =  sin(2π x)
#     B_y(x, y, z, 0) = -sin(2π x) / c
#     all other components = 0
#
# With c = 1 the wave traverses one wavelength in T = 1 and returns to IC.
# Although the physics is 1D, we run on a 3D mesh — the 3D kernel must
# handle the extra ζ-axis cleanly:
#   * the y and z flux divergences should produce zero net contribution
#     (fields are uniform in those directions)
#   * E_x, E_y, B_x, B_z must remain zero throughout

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell_3d import Maxwell3D
from warpxm.apps.maxwell_2d import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_maxwell_3d import DGMaxwellSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 8
comptime NZ: Int = 8
comptime C_LIGHT: Real = 1.0
comptime DT: Real = 5.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = DGState3D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var s = sin(TWO_PI * x)
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MAXWELL_NC
                            host[b + IDX_EX] = Real(0.0)
                            host[b + IDX_EY] = Real(0.0)
                            host[b + IDX_EZ] = s
                            host[b + IDX_BX] = Real(0.0)
                            host[b + IDX_BY] = -s / C_LIGHT
                            host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Maxwell3D(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver3D[P](app)
    var stepper = SSPRK3_3D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var t_final: Real = 1.0 / C_LIGHT

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, MAXWELL_NC],
        mut q_out: DGState3D[P, MAXWELL_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState3D[P, MAXWELL_NC]) raises:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < t_final else (t_final - t)
        stepper.step[compute_rhs, no_adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var err_Ez: Real = 0.0
    var err_By: Real = 0.0
    var max_other: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                for i in range(NX):
                    var xl = mesh.x_min + Real(i) * mesh.dx
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            for in_ in range(P1):
                                var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                                var s = sin(TWO_PI * x)
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MAXWELL_NC
                                var d_Ez = h[b + IDX_EZ] - s
                                var d_By = h[b + IDX_BY] - (-s / C_LIGHT)
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                err_Ez += jw * d_Ez * d_Ez
                                err_By += jw * d_By * d_By
                                # E_x, E_y, B_x, B_z must remain zero.
                                for idx in [IDX_EX, IDX_EY, IDX_BX, IDX_BZ]:
                                    var v = h[b + idx]
                                    if v > max_other: max_other = v
                                    if -v > max_other: max_other = -v
    var l2_Ez = err_Ez ** Real(0.5)
    var l2_By = err_By ** Real(0.5)
    print("3D Maxwell plane wave — t =", t)
    print("  L2 E_z error:", l2_Ez)
    print("  L2 B_y error:", l2_By)
    print("  max |E_x, E_y, B_x, B_z| (should be ~0):", max_other)
    assert_true(l2_Ez < Real(1.0e-2), "E_z error too large")
    assert_true(l2_By < Real(1.0e-2), "B_y error too large")
    assert_true(max_other < Real(1.0e-10),
                "transverse components grew — 3D kernel has spurious coupling")
    print("PASSED")
