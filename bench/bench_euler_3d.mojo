# 3D Euler bench driver.

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from warpxm.prelude import *
from warpxm.physics.euler import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime GAMMA: Real = 1.4
comptime DT: Real = 1.0e-3
comptime WARMUP: Int = 10
comptime N_STEPS: Int = 100
comptime State = DGState[3, P, EULER_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx,
        Real(0.0), Real(1.0),
        Real(0.0), Real(1.0),
        Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
        periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, EULER_NC]:
        var r2 = (
            (x - Real(0.5)) ** 2
            + (y - Real(0.5)) ** 2
            + (z - Real(0.5)) ** 2
        )
        var rho = Real(1.0) + Real(0.2) * exp(-r2 / Real(0.02))
        var p_loc: Real = 1.0
        return [rho, Real(0.0), Real(0.0), Real(0.0), p_loc / (GAMMA - Real(1.0))]
    mesh.fill(q, initial)

    var app = Euler(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGEulerSolver[3, P](app)
    var stepper = SSPRK3[3, P, EULER_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, EULER_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    for _ in range(WARMUP):
        stepper.step(ctx, q, DT, compute_rhs, adjust)
    ctx.synchronize()

    var t0 = perf_counter_ns()
    for _ in range(N_STEPS):
        stepper.step(ctx, q, DT, compute_rhs, adjust)
    ctx.synchronize()
    var t1 = perf_counter_ns()

    var elapsed_s = Real(t1 - t0) / Real(1.0e9)
    var steps_per_s = Real(N_STEPS) / elapsed_s
    print(
        "BENCH config=euler_3d steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(steps_per_s)
    )
