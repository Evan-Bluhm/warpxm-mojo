# Steady-state stepping driver for nsys / ncu profiling of the 3D MHD
# hot path. Gaussian pressure pulse in a uniform field (cheap to
# construct) — what matters is the per-step DG cost, not the physics setup.

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.imhd import *


comptime P: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime NZ: Int = 32
comptime GAMMA: Real = 5.0 / 3.0
comptime SIGMA: Real = 0.1
comptime P_AMB: Real = 1.0
comptime B0: Real = 1.0
comptime DT: Real = 5.0e-4
comptime WARMUP: Int = 5
comptime N_STEPS: Int = 30
comptime State = DGState[3, P, MHD_NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ, periodic_x=True, periodic_y=True, periodic_z=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, MHD_NC]:
        var r2 = (x - Real(0.5)) ** 2 + (y - Real(0.5)) ** 2 + (z - Real(0.5)) ** 2
        var P_loc = P_AMB + Real(5.0) * exp(-r2 / (SIGMA * SIGMA))
        var E_tot = P_loc / (GAMMA - Real(1.0)) + Real(0.5) * B0 * B0
        return [
            Real(1.0), Real(0.0), Real(0.0), Real(0.0), E_tot,
            B0, Real(0.0), Real(0.0),
        ]
    mesh.fill(q, initial)

    var app = IMHD(
        gas_gamma=GAMMA, min_density=Real(1e-12),
        min_pressure=Real(1e-12), cfl_max=Real(0.5),
    )
    var spatial = DGImhdSolver[3, P](app)
    var stepper = SSPRK3[3, P, MHD_NC](ctx, mesh.num_elements)
    var limiter = MoeRossmanith[3, P, MHD_NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    def adjust(mut q_: State) raises {mut limiter, imm}:
        limiter.apply(ctx, mesh, q_)

    for _ in range(WARMUP):
        stepper.step(ctx, q, DT, compute_rhs, adjust)
    ctx.synchronize()
    print("warmup_done")

    for _ in range(N_STEPS):
        stepper.step(ctx, q, DT, compute_rhs, adjust)
    ctx.synchronize()
    print("measure_done", N_STEPS, "steps NX=", NX, "NY=", NY, "NZ=", NZ, "P=", P)
