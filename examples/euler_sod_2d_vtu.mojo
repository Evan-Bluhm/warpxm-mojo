# 2D Sod shock tube with VTU output.
#
# Same setup as `tests/test_euler_sod_2d.mojo` but dumps a `.vtu` snapshot of
# the t = 0.2 state to /tmp/sod_2d.vtu so the result can be visualised in
# ParaView.

from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from warpxm.adjusters.moe_rossmanith_2d import MoeRossmanith2D
from warpxm.apps.euler_2d import (
    Euler2D, EULER_NC, IDX_RHO, IDX_MX, IDX_MY, IDX_MZ, IDX_E,
)
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.bc import BC_COPYOUT, BC_WALL_FREESLIP
from warpxm.core.types import Real
from warpxm.driver.vtu_2d import dump_euler_2d_frame
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_euler_2d import DGEulerSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NX: Int = 200
comptime NY: Int = 8
comptime GAMMA: Real = 1.4
comptime T_FINAL: Real = 0.2
comptime DT: Real = 2.0e-4
comptime OUTPUT_PATH: String = "/tmp/sod_2d.vtu"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(0.1),
        nx=NX, ny=NY, periodic_x=False, periodic_y=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
        bc_bottom=BC_WALL_FREESLIP, bc_top=BC_WALL_FREESLIP,
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
                    var rho: Real
                    var pp: Real
                    if x < Real(0.5):
                        rho = 1.0; pp = 1.0
                    else:
                        rho = 0.125; pp = 0.1
                    var n = jn * P1 + in_
                    var b = (e * nodes_per_elem + n) * EULER_NC
                    host[b + IDX_RHO] = rho
                    host[b + IDX_MX] = Real(0.0)
                    host[b + IDX_MY] = Real(0.0)
                    host[b + IDX_MZ] = Real(0.0)
                    host[b + IDX_E] = pp / (GAMMA - Real(1.0))
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Euler2D(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver2D[P](app)
    var stepper = SSPRK3_2D[P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith2D[P, EULER_NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState2D[P, EULER_NC],
        mut q_out: DGState2D[P, EULER_NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def adjust(mut q: DGState2D[P, EULER_NC]) raises:
        limiter.apply(ctx, mesh, q)

    var t: Real = 0.0
    while t < T_FINAL - Real(0.5) * DT:
        var dt_use = DT if (t + DT) < T_FINAL else (T_FINAL - t)
        stepper.step[compute_rhs, adjust](ctx, q, dt_use)
        t += dt_use
    ctx.synchronize()

    dump_euler_2d_frame[P](OUTPUT_PATH, ctx, mesh, q, GAMMA)
    print("Wrote", OUTPUT_PATH)
