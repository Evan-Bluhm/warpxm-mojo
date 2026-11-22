# 2D IMHD bench driver. Rusanov-flux IMHD with Moe-Rossmanith limiter.

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 128
comptime NY: Int = 128
comptime GAMMA: Real = 5.0 / 3.0
comptime B0: Real = 1.0
comptime DT: Real = 5.0e-4
comptime WARMUP: Int = 10
comptime N_STEPS: Int = 100
comptime State = DGState[2, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, MHD_NC]:
        var r2 = (x - Real(0.5)) ** 2 + (y - Real(0.5)) ** 2
        var rho = Real(1.0) + Real(0.2) * exp(-r2 / Real(0.01))
        var p_loc: Real = 1.0
        var bsq = B0 * B0
        var E_tot = p_loc / (GAMMA - Real(1.0)) + Real(0.5) * bsq
        return [
            rho, Real(0.0), Real(0.0), Real(0.0), E_tot,
            B0, Real(0.0), Real(0.0),
        ]
    mesh.fill(q, initial)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver[2, P](app)
    var stepper = SSPRK3[2, P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[2, P, MHD_NC](ctx, mesh.num_elements)

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
        "BENCH config=imhd_2d steps=" + String(N_STEPS)
        + " elapsed_s=" + String(elapsed_s)
        + " steps_per_s=" + String(steps_per_s)
    )
