# 3D Einfeldt 1-2-3 problem — 1D 1-2-3 extruded along y and z.
#
# Same IC as the 1D and 2D versions, on a [0,1] × [0,0.1] × [0,0.1] mesh
# with copyout in x and periodic in y, z. Validates 3D positivity
# preservation against vacuum-forming rarefactions; result must remain
# uniform in y and z.

from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith3D
from warpxm.adjusters.positivity_euler import PositivityEuler3D
from warpxm.apps.euler import Euler
from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_euler_3d import DGEulerSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 2
comptime NZ: Int = 2
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.15
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
                            var rho: Real = 1.0
                            var u: Real
                            var pp: Real = 0.4
                            if x < Real(0.5):
                                u = -2.0
                            else:
                                u = +2.0
                            var ke = Real(0.5) * rho * u * u
                            var E_tot = pp / (GAMMA - Real(1.0)) + ke
                            var n = kn * P1 * P1 + jn * P1 + in_
                            var b = (e * nodes_per_elem + n) * EULER_NC
                            host[b + IDX_RHO] = rho
                            host[b + IDX_MX] = rho * u
                            host[b + IDX_MY] = Real(0.0)
                            host[b + IDX_MZ] = Real(0.0)
                            host[b + IDX_E] = E_tot
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver3D[P](app)
    var limiter = MoeRossmanith3D[P, EULER_NC](ctx, mesh.num_elements)
    var positivity = PositivityEuler3D[P](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var stepper = SSPRK3_3D[P, EULER_NC](ctx, mesh.num_elements)


    def compute_rhs(
        mut q_in: DGState3D[P, EULER_NC],
        mut q_out: DGState3D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: DGState3D[P, EULER_NC]) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q_)
        positivity.apply(ctx, mesh, q_)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step(ctx, q, dt_use, compute_rhs, adjust)
        t += dt_use
    ctx.synchronize()

    var mid_node = nodes_per_elem // 2
    var any_nan = False
    var min_rho: Real = 1.0e30
    var min_p: Real = 1.0e30
    var max_y_drift: Real = 0.0
    var max_z_drift: Real = 0.0
    var rho_row = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        for i in range(NX):
            var e = 0 * NX * NY + 0 * NX + i
            var b = (e * nodes_per_elem + mid_node) * EULER_NC
            var rho = h[b + IDX_RHO]
            if isnan(rho): any_nan = True
            rho_row[i] = rho
            var mx = h[b + IDX_MX]
            var E = h[b + IDX_E]
            var u = mx / rho
            var ke = Real(0.5) * rho * u * u
            var pp = (GAMMA - Real(1.0)) * (E - ke)
            if rho < min_rho: min_rho = rho
            if pp < min_p: min_p = pp
        for k in range(NZ):
            for j in range(NY):
                if k == 0 and j == 0: continue
                for i in range(NX):
                    var e = k * NX * NY + j * NX + i
                    var b = (e * nodes_per_elem + mid_node) * EULER_NC
                    var d = fabs(h[b + IDX_RHO] - rho_row[i])
                    if j != 0 and d > max_y_drift: max_y_drift = d
                    if k != 0 and d > max_z_drift: max_z_drift = d

    print("3D Einfeldt 1-2-3 problem — t =", t)
    print("  min ρ:                  ", min_rho)
    print("  min P:                  ", min_p)
    print("  max y-drift in ρ:        ", max_y_drift)
    print("  max z-drift in ρ:        ", max_z_drift)

    assert_true(not any_nan, "3D 1-2-3 NaN'd")
    assert_true(min_rho > Real(0.0), "negative ρ survived")
    assert_true(min_p > Real(0.0), "negative P survived")
    assert_true(max_y_drift < Real(1.0e-10), "not uniform in y")
    assert_true(max_z_drift < Real(1.0e-10), "not uniform in z")
    print("PASSED")
