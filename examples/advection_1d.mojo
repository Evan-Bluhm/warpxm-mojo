# 1D scalar advection on a periodic block mesh.
#
# Mirrors `examples/dg/advection/advection.py` from upstream WARPXM, restricted
# to one dimension and to a periodic domain with SSPRK3 time stepping. The
# initial condition is a single sine period on x ∈ [-0.5, 0.5]; with velocity
# v = 1 the exact solution returns to the IC at t = 1.
#
#     pixi run example advection_1d
#
# This is a compiled input file — there is no runtime input parsing. Edit the
# constants near the top to change the run.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *


comptime P: Int = 2                  # polynomial order (3 GLL nodes / elem)
comptime NUM_ELEMENTS: Int = 64
comptime X_MIN: Real = -0.5
comptime X_MAX: Real = 0.5
comptime VELOCITY: Real = 1.0
comptime CFL: Real = 0.3
comptime T_FINAL: Real = 1.0         # one full period
comptime NUM_PRINT_NODES: Int = 8    # how many nodes to dump at start/end
comptime State = DGState[1, P, 1]


def dump_first_few(label: String, mut q: State) raises:
    """Map q to host and print the first NUM_PRINT_NODES values."""
    with q.buf.map_to_host() as h:
        root_print(label)
        var n = NUM_PRINT_NODES if NUM_PRINT_NODES < q.total_nodes else q.total_nodes
        for k in range(n):
            root_print("  q[", k, "] =", h[k])


def main() raises:
    comptime assert has_accelerator(), "warpxm-mojo requires a GPU"

    var ctx = rank_device_context()
    var mesh = BlockMesh[1, P](ctx, X_MIN, X_MAX, NUM_ELEMENTS, periodic=True)
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real) -> Array[Real, 1]:
        return [sin(TWO_PI * x)]
    mesh.fill(q, initial)
    dump_first_few("Initial q (first nodes):", q)

    var app = Advection[1](velocity_x=VELOCITY, cfl_max=CFL)
    var spatial = DGAdvectionSolver[1, P, 1](app)
    var stepper = SSPRK3[1, P, 1](ctx, mesh.num_elements, mesh.num_ghost)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    # The velocity is uniform, so the CFL step is fixed for the whole run.
    var dt = spatial.suggest_dt(mesh)
    var t = stepper.run_until(
        ctx, q, Real(0.0), T_FINAL, dt, compute_rhs, no_op_adjust[1, P, 1],
    )

    root_print("\nIntegrated to t =", t, "with dt =", dt)
    dump_first_few("Final q (first nodes):", q)

    # After one period on a periodic domain the exact solution is the IC.
    def exact(x: Real) -> Real:
        return sin(TWO_PI * x)
    root_print("L2 error vs IC after one period:", mesh.l2_error(q, 0, exact))
