# Multi-rank regression: 2D periodic advection gives bitwise the same
# field for any number of ranks.
#
# The mesh is 18 x 14 so that 3 and 4 ranks cut it into unequal boxes, and
# periodic in both directions so that every partition face is exchanged,
# including a rank that is its own neighbour's neighbour on both sides
# (2 ranks along an axis). `scripts/run_mpi_tests.sh` runs this at several
# rank counts and compares the `CHECKSUM` lines.

from std.collections import Array
from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *

comptime P: Int = 2
comptime State = DGState[2, P, 1]


def _wave(x: Real, y: Real) -> Real:
    return sin(TWO_PI * x) * cos(TWO_PI * y)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=18, ny=14, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)

    def initial(x: Real, y: Real) -> Array[Real, 1]:
        return [_wave(x, y)]
    mesh.fill(q, initial)

    var spatial = DGAdvectionSolver[2, P, 1](
        Advection[1](velocity_x=Real(1.0), velocity_y=Real(0.5), cfl_max=Real(1.0))
    )
    var stepper = SSPRK3[2, P, 1](ctx, mesh.num_elements, mesh.num_ghost)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    # Two periods in x, one in y: back to the initial condition.
    _ = stepper.run_until(
        ctx, q, Real(0.0), Real(2.0), Real(2.0e-3), compute_rhs,
        no_op_adjust[2, P, 1],
    )
    var err = mesh.l2_error(q, 0, _wave)
    var sum = mesh.checksum(q)
    if is_root():
        print("ranks =", num_ranks(), " L2 error =", err)
        print("CHECKSUM", sum)
    if not (err < Real(2.0e-3)):
        raise Error("L2 error too large")
    if is_root():
        print("PASSED")
