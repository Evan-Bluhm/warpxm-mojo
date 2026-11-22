# 2D Einfeldt 1-2-3 problem — 1D 1-2-3 extruded along y.
#
# Same IC as `test_euler_123_1d.mojo` made uniform in y, on a periodic-y
# domain. Validates that the 2D Zhang-Shu positivity preserver keeps
# the near-vacuum centre alive (analogous to the 1D test) and that the
# solution stays uniform in y.
#
# IC (Einfeldt et al. 1991):
#   Left  (x < 0.5): ρ = 1, u = -2, P = 0.4
#   Right (x ≥ 0.5): ρ = 1, u = +2, P = 0.4
#   γ = 1.4, t_final = 0.15

from std.math import isnan
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.adjusters.moe_rossmanith import MoeRossmanith2D
from warpxm.adjusters.positivity_euler import PositivityEuler2D
from warpxm.apps.euler import (
    Euler, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.math_compat import fabs
from warpxm.core.types import Real
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 2
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.15
comptime DT: Real = 1.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=True,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = DGState2D[P, EULER_NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
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
                    var n = jn * P1 + in_
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
    var spatial = DGEulerSolver2D[P](app)
    var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)
    var positivity = PositivityEuler2D[P](
        num_elements=mesh.num_elements, gamma=GAMMA,
        rho_eps=Real(1.0e-10), p_eps=Real(1.0e-10),
    )
    var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q: DGState2D[P, EULER_NC]) raises {mut limiter, mut positivity, imm}:
        limiter.apply(ctx, mesh, q)
        positivity.apply(ctx, mesh, q)

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
    var max_rho: Real = -1.0e30
    var max_y_drift: Real = 0.0
    var rho_row = Array[Real, NX](fill=Real(0.0))
    with q.buf.map_to_host() as h:
        # j=0 row first.
        for i in range(NX):
            var e = 0 * NX + i
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
            if rho > max_rho: max_rho = rho
            if pp < min_p: min_p = pp
        # Check y-uniformity.
        for j in range(NY):
            if j == 0: continue
            for i in range(NX):
                var e = j * NX + i
                var b = (e * nodes_per_elem + mid_node) * EULER_NC
                var d = fabs(h[b + IDX_RHO] - rho_row[i])
                if d > max_y_drift: max_y_drift = d

    print("2D Einfeldt 1-2-3 problem — t =", t)
    print("  min ρ:                  ", min_rho)
    print("  max ρ:                  ", max_rho)
    print("  min P:                  ", min_p)
    print("  max y-drift in ρ:        ", max_y_drift)

    assert_true(not any_nan, "2D 1-2-3 NaN'd — positivity preserver failed")
    assert_true(min_rho > Real(0.0), "negative ρ survived")
    assert_true(min_p > Real(0.0), "negative P survived")
    assert_true(max_y_drift < Real(1.0e-10), "result not uniform in y")
    print("PASSED")
