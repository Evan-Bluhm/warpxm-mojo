# 3D advection bench driver. See `bench_advection_2d.mojo` for the
# measurement-window contract.

from std.math import sin
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.time import perf_counter_ns

from warpxm.apps.advection import Advection
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_3d import BlockMesh3D
from warpxm.spatial.dg_3d import DGSpatialSolver3D
from warpxm.temporal.ssprk3_3d import SSPRK3_3D
from warpxm.variable.state_3d import DGState3D


comptime P: Int = 2
comptime NC: Int = 1
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime DT: Real = 1.0e-3
comptime WARMUP: Int = 10
comptime N_STEPS: Int = 100


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh3D[P](
        ctx,
        Real(-0.5), Real(0.5),
        Real(-0.5), Real(0.5),
        Real(-0.5), Real(0.5),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = DGState3D[P, NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1 * P1
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
                            var x = (
                                xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                            )
                            var n = (kn * P1 + jn) * P1 + in_
                            host[(e * nodes_per_elem + n) * NC + 0] = (
                                sin(TWO_PI * x) * sin(TWO_PI * y) * sin(TWO_PI * z)
                            )
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Advection[NC](
        velocity_x=Real(1.0), velocity_y=Real(1.0), velocity_z=Real(1.0),
        cfl_max=Real(1.0),
    )
    var spatial = DGSpatialSolver3D[P, NC](app)
    var stepper = SSPRK3_3D[P, NC](ctx, mesh.num_elements)

    @parameter
    def compute_rhs(
        mut q_in: DGState3D[P, NC], mut q_out: DGState3D[P, NC],
    ) raises:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    @parameter
    def no_adjust(mut q_: DGState3D[P, NC]) raises:
        pass

    for _ in range(WARMUP):
        stepper.step[compute_rhs, no_adjust](ctx, q, DT)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        stepper.step[compute_rhs, no_adjust](ctx, q, DT)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var elapsed_s = Real(t1 - t0) / Real(1.0e9)
    var steps_per_s = Real(N_STEPS) / elapsed_s
    print(
        "BENCH config=advection_3d steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(steps_per_s)
    )
