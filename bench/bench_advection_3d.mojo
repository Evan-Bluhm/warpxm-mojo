# 3D advection bench driver. See `bench_advection_2d.mojo` for the
# measurement-window contract.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from warpxm.prelude import *
from warpxm.physics.advection import *


comptime P: Int = 2
comptime NC: Int = 1
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime DT: Real = 1.0e-3
comptime WARMUP: Int = 10
comptime N_STEPS: Int = 100
comptime State = DGState[3, P, NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx,
        Real(-0.5), Real(0.5),
        Real(-0.5), Real(0.5),
        Real(-0.5), Real(0.5),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, NC]:
        return [sin(TWO_PI * x) * sin(TWO_PI * y) * sin(TWO_PI * z)]
    mesh.fill(q, initial)

    var app = Advection[NC](
        velocity_x=Real(1.0), velocity_y=Real(1.0), velocity_z=Real(1.0),
        cfl_max=Real(1.0),
    )
    var spatial = DGAdvectionSolver[3, P, NC](app)
    var stepper = SSPRK3[3, P, NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def no_adjust(mut q_: State) raises {imm}:
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
        "BENCH config=advection_3d steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(steps_per_s)
    )
