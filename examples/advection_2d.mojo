# 2D scalar advection on a doubly-periodic block mesh.
#
# Initial condition: q(x, y, 0) = sin(2π x) · cos(2π y) on [-0.5, 0.5]².
# Velocity (v_x, v_y) = (1, 1). Exact solution at t = 1 (one period in each
# axis) returns to the IC.
#
#     pixi run example advection_2d

from std.collections import Array
from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *


comptime P: Int = 2
comptime NC: Int = 1
comptime NX: Int = 32
comptime NY: Int = 32
comptime CFL: Real = 0.3
comptime T_FINAL: Real = 1.0
comptime State = DGState[2, P, NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(-0.5), Real(0.5), Real(-0.5), Real(0.5),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, NC]:
        return [sin(TWO_PI * x) * cos(TWO_PI * y)]
    mesh.fill(q, initial)

    var app = Advection[NC](
        velocity_x=Real(1.0), velocity_y=Real(1.0), cfl_max=CFL,
    )
    var spatial = DGAdvectionSolver[2, P, NC](app)
    var stepper = SSPRK3[2, P, NC](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    # The velocity is uniform, so the CFL step is fixed for the whole run.
    var dt = spatial.suggest_dt(mesh)
    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, dt, compute_rhs, no_op_adjust[2, P, NC],
    )

    def exact(x: Real, y: Real) -> Real:
        return sin(TWO_PI * x) * cos(TWO_PI * y)
    print("Integrated to t =", t, "with dt =", dt)
    print("L2 error after one period:", mesh.l2_error(q, 0, exact))
