# 1D Sod shock tube — canonical CFD test problem.
#
# Initial discontinuity at x = 0.5 separates two uniform states:
#
#     Left  (x < 0.5):  ρ = 1.0,     u = 0,  P = 1.0
#     Right (x ≥ 0.5):  ρ = 0.125,   u = 0,  P = 0.1
#
# γ = 1.4 (diatomic). The exact Riemann solution (Toro Ch. 4) has, from left
# to right:
#
#     1) Left-moving rarefaction fan
#     2) Constant left state
#     3) Contact discontinuity (jump in ρ, continuous P and u)
#     4) Constant right state
#     5) Right-moving shock
#
# Now using the Moe-Rossmanith slope limiter (conserved-variable cut) to
# damp Gibbs oscillations near the discontinuities. The post-shock plateau,
# the contact, and the rarefaction structure should appear at the correct
# locations at `t = 0.2`.

from std.math import exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.apps.euler import Euler
from warpxm.apps.euler import EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT
from warpxm.core.types import Real
from warpxm.mesh.block_1d import BlockMesh1D
from warpxm.adjusters.moe_rossmanith_1d import MoeRossmanith1D
from warpxm.spatial.dg_euler_1d import DGEulerSolver1D
from warpxm.temporal.ssprk3_1d import SSPRK3_1D
from warpxm.variable.state_1d import DGState1D


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 200
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime DT: Real = 2.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh1D[P](
        ctx,
        Real(0.0),
        Real(1.0),
        NUM_ELEMENTS,
        periodic=False,
        bc_left=BC_COPYOUT,
        bc_right=BC_COPYOUT,
    )
    var q = DGState1D[P, EULER_NC](ctx, mesh.num_elements)

    var nodes_per_elem = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()

    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for e in range(mesh.num_elements):
        var xl = mesh.x_min + Real(e) * mesh.dx
        for i in range(nodes_per_elem):
            var x = xl + (Real(1.0) + ref_nodes[i]) * mesh.jacobian
            var rho: Real
            var pp: Real
            if x < Real(0.5):
                rho = 1.0
                pp = 1.0
            else:
                rho = 0.125
                pp = 0.1
            var b = (e * nodes_per_elem + i) * EULER_NC
            host[b + IDX_RHO] = rho
            host[b + IDX_MX] = Real(0.0)
            host[b + IDX_MY] = Real(0.0)
            host[b + IDX_MZ] = Real(0.0)
            host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler(
        gas_gamma=GAMMA,
        min_density=Real(1e-12),
        min_pressure=Real(1e-12),
        cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver1D[P](app)
    var stepper = SSPRK3_1D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith1D[P](ctx, mesh.num_elements)


    @parameter
    def compute_rhs(
        mut q_in: DGState1D[P, EULER_NC],
        mut q_out: DGState1D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q_: DGState1D[P, EULER_NC]) raises:
        limiter.apply(ctx, mesh, q_)

    var t: Real = 0.0
    var step: Int = 0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, adjust](ctx, q, dt_use)
        t += dt_use
        step += 1
    ctx.synchronize()

    # Print rho, u, P at element centroids for a quick text view.
    print("# Sod shock tube at t =", t, " after", step, "steps")
    print("# x, rho, u, P")
    var mid_node = NodalBasis1D[P].num_nodes // 2
    with q.buf.map_to_host() as h:
        for e in range(mesh.num_elements):
            var xl = mesh.x_min + Real(e) * mesh.dx
            var x = xl + (Real(1.0) + ref_nodes[mid_node]) * mesh.jacobian
            var b = (e * nodes_per_elem + mid_node) * EULER_NC
            var rho = h[b + IDX_RHO]
            var mx = h[b + IDX_MX]
            var e_t = h[b + IDX_E]
            var u = mx / rho
            var ke = Real(0.5) * rho * u * u
            var p_local = (GAMMA - Real(1.0)) * (e_t - ke)
            print(x, ",", rho, ",", u, ",", p_local)
