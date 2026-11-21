# 3D rectangular cavity TM mode regression.
#
# Same TM₁₁ standing-wave family as the 2D test, but on a 3D mesh bounded
# by PEC walls on all six sides. We pick a z-uniform mode (the simplest
# analytically valid eigenmode of a [0,1]³ PEC cavity that doesn't require
# more elaborate construction):
#
#     E_z(x, y, z, t) = sin(π x) · sin(π y) · cos(ω t)
#     B_x = +(π / ω) · sin(π x) · cos(π y) · sin(ω t)
#     B_y = -(π / ω) · cos(π x) · sin(π y) · sin(ω t)
#     (E_x = E_y = B_z = 0 throughout)
#
# with `ω = c · π · √2`. After one full period `T = 2π/ω = √2 / c` the
# fields return to the IC.
#
# PEC boundary conditions are satisfied by construction:
#   * x = 0 or 1: tangential E (here E_z, E_y) vanishes via the sin(πx)
#     factor; normal B (B_x) vanishes via the same sin(πx) factor
#   * y = 0 or 1: analogous via sin(πy)
#   * z = 0 or 1: tangential E (E_x, E_y) is zero throughout the cavity,
#     normal B (B_z) is zero throughout — all trivially compatible
#
# This setup exercises every PEC wall axis (the x and y walls actively,
# the z walls trivially) and confirms the 3D Maxwell kernel doesn't leak
# energy into E_x / E_y / B_z modes through the PEC ghost-state synthesis.

from std.math import cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.apps.maxwell import Maxwell
from warpxm.apps.maxwell import (
    MAXWELL_NC, IDX_EX, IDX_EY, IDX_EZ, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_PEC_WALL
from warpxm.core.math_compat import sqrt
from warpxm.core.types import Real, PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_maxwell_3d import DGMaxwellSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 4
comptime C_LIGHT: Real = 1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=False, periodic_y=False, periodic_z=False,
        bc_left=BC_PEC_WALL, bc_right=BC_PEC_WALL,
        bc_bottom=BC_PEC_WALL, bc_top=BC_PEC_WALL,
        bc_down=BC_PEC_WALL, bc_up=BC_PEC_WALL,
    )
    var q = DGState3D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var weights = NodalBasis1D[P].weights()
    var nodes_per_elem = P1 * P1 * P1

    # IC: only E_z, at peak of the cos(ωt) factor; B = 0.
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for k in range(NZ):
        for j in range(NY):
            var yl = mesh.y_min + Real(j) * mesh.dy
            for i in range(NX):
                var xl = mesh.x_min + Real(i) * mesh.dx
                var e = k * NX * NY + j * NX + i
                for kn in range(P1):
                    for jn in range(P1):
                        var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                        for in_ in range(P1):
                            var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MAXWELL_NC
                            host[b + IDX_EX] = Real(0.0)
                            host[b + IDX_EY] = Real(0.0)
                            host[b + IDX_EZ] = sin(PI * x) * sin(PI * y)
                            host[b + IDX_BX] = Real(0.0)
                            host[b + IDX_BY] = Real(0.0)
                            host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Maxwell(c_light=C_LIGHT, cfl_max=Real(0.5))
    var spatial = DGMaxwellSolver3D[P](app)
    var stepper = SSPRK3_3D[P, MAXWELL_NC](ctx, mesh.num_elements)

    var t_final = sqrt(Real(2.0)) / C_LIGHT
    var dt: Real = Real(2.0e-4)

    def compute_rhs(
        mut q_in: DGState3D[P, MAXWELL_NC],
        mut q_out: DGState3D[P, MAXWELL_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState3D[P, MAXWELL_NC]) raises {imm}:
        pass

    var t: Real = 0.0
    while t < t_final - Real(0.5) * dt:
        var dt_use = dt if (t + dt) < t_final else (t_final - t)
        stepper.step(ctx, q, dt_use, compute_rhs, no_adjust)
        t += dt_use
    ctx.synchronize()

    var jv = mesh.jacobian_x * mesh.jacobian_y * mesh.jacobian_z
    var err_ez: Real = 0.0
    var max_other: Real = 0.0
    with q.buf.map_to_host() as h:
        for k in range(NZ):
            for j in range(NY):
                var yl = mesh.y_min + Real(j) * mesh.dy
                for i in range(NX):
                    var xl = mesh.x_min + Real(i) * mesh.dx
                    var e = k * NX * NY + j * NX + i
                    for kn in range(P1):
                        for jn in range(P1):
                            var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                            for in_ in range(P1):
                                var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                                var n = kn * P1 * P1 + jn * P1 + in_
                                var b = (e * nodes_per_elem + n) * MAXWELL_NC
                                var exact_ez = sin(PI * x) * sin(PI * y)
                                var d_ez = h[b + IDX_EZ] - exact_ez
                                var jw = jv * weights[in_] * weights[jn] * weights[kn]
                                err_ez += jw * d_ez * d_ez
                                # E_x, E_y, B_z must remain identically zero.
                                for idx in [IDX_EX, IDX_EY, IDX_BZ]:
                                    var v = h[b + idx]
                                    if v > max_other: max_other = v
                                    if -v > max_other: max_other = -v
    var l2_ez = err_ez ** Real(0.5)
    print("3D Maxwell cavity TM mode — t =", t)
    print("  L2 E_z error vs IC:                                  ", l2_ez)
    print("  max |E_x, E_y, B_z| (must be ~0 by symmetry):        ", max_other)
    assert_true(l2_ez < Real(5.0e-3), "E_z L2 error too large — PEC reflection wrong")
    assert_true(max_other < Real(1.0e-10),
                "spurious E_x / E_y / B_z excited — wall BC mixing modes")
    print("PASSED")
