# 3D Sod shock tube — 1D physics extruded along y and z.
#
# Initial discontinuity at x = 0.5; uniform in y and z. Sides perpendicular
# to x are copyout outflows; top/bottom and down/up are free-slip walls.
# Validates the 3D non-periodic BC plumbing on a problem with a known
# analytic structure.
#
# Expected outcome at t = 0.2:
#   * No NaN
#   * Result uniform in y and z at every x
#   * x-profile matches the 1D Sod analytic solution (within shock-capturing
#     tolerance) at four sample locations

from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.apps.euler import (
    EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.apps.euler import Euler
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT, BC_WALL_FREESLIP
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 100
comptime NY: Int = 4
comptime NZ: Int = 4
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime DT: Real = 5.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1), Real(0.0), Real(0.1),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=False, periodic_y=False, periodic_z=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
        bc_down=BC_WALL_FREESLIP, bc_up=BC_WALL_FREESLIP,
    )
    var q = DGState3D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
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
                            var rho: Real
                            var pp: Real
                            if x < Real(0.5):
                                rho = 1.0; pp = 1.0
                            else:
                                rho = 0.125; pp = 0.1
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver3D[P](app)
    var stepper = SSPRK3_3D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, EULER_NC](ctx, mesh.num_elements)


    def compute_rhs(
        mut q_in: DGState3D[P, EULER_NC],
        mut q_out: DGState3D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState3D[P, EULER_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    # Pull the mid-row (j=0, k=0) midpoint sample at every x.
    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var max_y_drift: Real = 0.0
    var max_z_drift: Real = 0.0
    var rho_row = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        # First pass: row 0 in z=0 plane.
        for i in range(NX):
            var e = 0 * NX * NY + 0 * NX + i
            var b = (e * nodes_per_elem + mid_node) * EULER_NC
            var rho_ji = h[b + IDX_RHO]
            if isnan(rho_ji): any_nan = True
            rho_row[i] = rho_ji
        # Compare every other (j, k) sample to the row-0 z-0 value at the same x.
        for k in range(NZ):
            for j in range(NY):
                if j == 0 and k == 0: continue
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    var b = (e * nodes_per_elem + mid_node) * EULER_NC
                    var d = fabs(h[b + IDX_RHO] - rho_row[i])
                    if j != 0 and d > max_y_drift: max_y_drift = d
                    if k != 0 and d > max_z_drift: max_z_drift = d

    assert_true(not any_nan, "3D Sod NaN'd")
    print("max y-drift in ρ:", max_y_drift)
    print("max z-drift in ρ:", max_z_drift)
    assert_true(max_y_drift < Real(1.0e-10), "result not uniform in y")
    assert_true(max_z_drift < Real(1.0e-10), "result not uniform in z")

    var ix01 = Int(0.05 / mesh.dx)
    var ix06 = Int(0.6 / mesh.dx)
    var ix075 = Int(0.75 / mesh.dx)
    var ix095 = Int(0.95 / mesh.dx)
    print("ρ at x≈0.05 (left state ≈ 1.0):  ", rho_row[ix01])
    print("ρ at x≈0.6 (post-rarefaction ≈ 0.4263):", rho_row[ix06])
    print("ρ at x≈0.75 (post-contact ≈ 0.2655):  ", rho_row[ix075])
    print("ρ at x≈0.95 (right state ≈ 0.125):     ", rho_row[ix095])

    assert_true(fabs(rho_row[ix01] - Real(1.0)) < Real(2.0e-2), "left state off")
    assert_true(fabs(rho_row[ix06] - Real(0.4263)) < Real(0.04), "post-rarefaction off")
    assert_true(fabs(rho_row[ix075] - Real(0.2655)) < Real(0.04), "post-contact off")
    assert_true(fabs(rho_row[ix095] - Real(0.125)) < Real(2.0e-2), "right state off")
    print("PASSED")
