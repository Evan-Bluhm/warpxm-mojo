# 3D Brio-Wu MHD shock tube — 1D physics extruded along y and z.
#
# Same Riemann problem as `test_imhd_briowu_2d.mojo` lifted into 3D: an
# x-discontinuity at x = 0.5, periodic in y, periodic in z (free choice
# since the IC is uniform in those axes), copyout on x. γ = 2.
#
# Left state  (x < 0.5):  ρ = 1,     P = 1,    B_y = +1
# Right state (x ≥ 0.5):  ρ = 0.125, P = 0.1,  B_y = -1
# Both states:             u = 0, B_x = 0.75, B_z = 0.
#
# Validation:
#   * no NaN
#   * solution remains uniform in y and z (1D extruded)
#   * far-field x states preserved (shock hasn't reached them at t = 0.1)

from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.apps.imhd import IMHD
from warpxm.apps.imhd import (
    MHD_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E, IDX_BX, IDX_BY, IDX_BZ,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_imhd_3d import DGImhdSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 4
comptime NZ: Int = 4
comptime GAMMA: Real = 2.0
comptime T_FINAL: Real = 0.1
comptime DT: Real = 1.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1), Real(0.0), Real(0.1),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=False, periodic_y=True, periodic_z=True,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = DGState3D[P, MHD_NC](ctx, mesh.num_elements)

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
                            var By: Real
                            if x < Real(0.5):
                                rho = 1.0; pp = 1.0; By = 1.0
                            else:
                                rho = 0.125; pp = 0.1; By = -1.0
                            var Bx_v: Real = 0.75
                            var bsq = Bx_v * Bx_v + By * By
                            var E_tot = pp / (GAMMA - Real(1.0)) + Real(0.5) * bsq
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * MHD_NC
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = Real(0.0)
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = E_tot
                            host[b + IDX_BX] = Bx_v
                            host[b + IDX_BY] = By
                            host[b + IDX_BZ] = Real(0.0)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver3D[P](app)
    var stepper = SSPRK3_3D[P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith3D[P, MHD_NC](ctx, mesh.num_elements)


    def compute_rhs(
        mut q_in: DGState3D[P, MHD_NC],
        mut q_out: DGState3D[P, MHD_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState3D[P, MHD_NC]) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var max_y_drift: Real = 0.0
    var max_z_drift: Real = 0.0
    var rho_row = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        # First pass: row j=0, k=0.
        for i in range(NX):
            var e = 0 * NX * NY + 0 * NX + i
            var b = (e * nodes_per_elem + mid_node) * MHD_NC
            var rho_v = h[b + IDX_RHO]
            if isnan(rho_v): any_nan = True
            rho_row[i] = rho_v
        for k in range(NZ):
            for j in range(NY):
                if k == 0 and j == 0: continue
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    var b = (e * nodes_per_elem + mid_node) * MHD_NC
                    var d = fabs(h[b + IDX_RHO] - rho_row[i])
                    if j != 0 and d > max_y_drift: max_y_drift = d
                    if k != 0 and d > max_z_drift: max_z_drift = d

    assert_true(not any_nan, "3D Brio-Wu NaN'd")
    print("max y-drift in ρ (uniformity check):", max_y_drift)
    print("max z-drift in ρ:                   ", max_z_drift)
    assert_true(max_y_drift < Real(1.0e-10), "result not uniform in y")
    assert_true(max_z_drift < Real(1.0e-10), "result not uniform in z")

    # Far-field state sanity checks. At t = 0.1 the shock fronts haven't
    # propagated far enough to disturb x ≈ 0.05 or x ≈ 0.95.
    var ix005 = Int(0.05 / mesh.dx)
    var ix095 = Int(0.95 / mesh.dx)
    print("ρ at x≈0.05 (left state ≈ 1.0):  ", rho_row[ix005])
    print("ρ at x≈0.95 (right state ≈ 0.125):", rho_row[ix095])
    assert_true(fabs(rho_row[ix005] - Real(1.0)) < Real(2.0e-2), "left far-field off")
    assert_true(fabs(rho_row[ix095] - Real(0.125)) < Real(2.0e-2), "right far-field off")
    print("PASSED")
