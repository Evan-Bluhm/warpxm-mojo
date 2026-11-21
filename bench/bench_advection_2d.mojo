# 2D advection bench driver. WARMUP + N_STEPS SSPRK3 steps with wall-clock
# timing between the two `ctx.synchronize()` calls. Prints a single
# `BENCH config=... steps=... elapsed_s=... steps_per_s=...` line that
# `scripts/bench.sh` parses into the CSV.
#
# Build / run:
#     pixi run mojo run -I src bench/bench_advection_2d.mojo

from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from warpxm.apps.advection import Advection
from warpxm.basis.nodal_basis_1d import NodalBasis1D
from warpxm.core.types import Real, TWO_PI
from warpxm.mesh.block_2d import BlockMesh2D
from warpxm.spatial.dg_2d import DGSpatialSolver2D
from warpxm.temporal.ssprk3_2d import SSPRK3_2D
from warpxm.variable.state_2d import DGState2D


comptime P: Int = 2
comptime NC: Int = 1
comptime NX: Int = 128
comptime NY: Int = 128
comptime DT: Real = 5.0e-4
comptime WARMUP: Int = 10
comptime N_STEPS: Int = 100


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh2D[P](
        ctx, Real(-0.5), Real(0.5), Real(-0.5), Real(0.5),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = DGState2D[P, NC](ctx, mesh.num_elements)

    var P1 = NodalBasis1D[P].num_nodes
    var ref_nodes = NodalBasis1D[P].nodes()
    var nodes_per_elem = P1 * P1
    var host = ctx.enqueue_create_host_buffer[Real.dtype](q.buf_len)
    for j in range(NY):
        var yl = mesh.y_min + Real(j) * mesh.dy
        for i in range(NX):
            var xl = mesh.x_min + Real(i) * mesh.dx
            var e = j * NX + i
            for jn in range(P1):
                var y = yl + (Real(1.0) + ref_nodes[jn]) * mesh.jacobian_y
                for in_ in range(P1):
                    var x = xl + (Real(1.0) + ref_nodes[in_]) * mesh.jacobian_x
                    var n = jn * P1 + in_
                    host[(e * nodes_per_elem + n) * NC + 0] = sin(TWO_PI * x) * sin(TWO_PI * y)
    ctx.enqueue_copy(dst_buf=q.buf, src_buf=host)

    var app = Advection[NC](
        velocity_x=Real(1.0), velocity_y=Real(1.0), cfl_max=Real(1.0),
    )
    var spatial = DGSpatialSolver2D[P, NC](app)
    var stepper = SSPRK3_2D[P, NC](ctx, mesh.num_elements)

    def compute_rhs(
        mut q_in: DGState2D[P, NC], mut q_out: DGState2D[P, NC],
    ) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: DGState2D[P, NC]) raises {imm}:
        pass

    for _ in range(WARMUP):
        stepper.step(ctx, q, DT, compute_rhs, no_adjust)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        stepper.step(ctx, q, DT, compute_rhs, no_adjust)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var elapsed_s = Real(t1 - t0) / Real(1.0e9)
    var steps_per_s = Real(N_STEPS) / elapsed_s
    print(
        "BENCH config=advection_2d steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(steps_per_s)
    )
