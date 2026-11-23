# Multi-rank regression: Dormand-Prince 4(5) on a 2D Maxwell plane wave.
#
# Fixed steps first: bitwise independent of the number of ranks
# (`CHECKSUM`). Then adaptive steps: the error norm is a floating-point sum
# over ranks, so the accepted step sizes agree between rank counts only to
# round-off and the adaptive result is checked against the exact solution
# instead.

from std.collections import Array
from std.math import sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.maxwell import *

comptime P: Int = 2
comptime State = DGState[2, P, MAXWELL_NC]
comptime INV_SQRT2: Real = 0.70710678118654752440


def _fill(mesh: BlockMesh[2, P], mut q: State) raises:
    def initial(x: Real, y: Real) -> Array[Real, MAXWELL_NC]:
        var s = sin(TWO_PI * (x + y))
        return [
            Real(0.0), Real(0.0), s, INV_SQRT2 * s, -INV_SQRT2 * s, Real(0.0),
        ]
    mesh.fill(q, initial)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = rank_device_context()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=21, ny=20, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements, mesh.num_ghost)
    var spatial = DGMaxwellSolver[2, P](Maxwell(c_light=Real(1.0), cfl_max=Real(0.5)))
    var stepper = DormandPrince45[2, P, MAXWELL_NC](
        ctx, mesh.num_elements, mesh.num_ghost,
    )

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    # The wave travels along (1, 1) at speed 1 and has period 1 / sqrt(2).
    def exact(x: Real, y: Real) -> Real:
        return sin(TWO_PI * (x + y))

    _fill(mesh, q)
    _ = stepper.run_until(ctx, q, Real(0.0), INV_SQRT2, Real(2.0e-3), compute_rhs)
    var err_fixed = mesh.l2_error(q, IDX_EZ, exact)
    var sum = mesh.checksum(q)

    _fill(mesh, q)
    var t: Real = 0.0
    var dt: Real = 1.0e-2
    var steps = 0
    while t < INV_SQRT2 - Real(1.0e-12):
        var dt_try = dt if (t + dt) < INV_SQRT2 else (INV_SQRT2 - t)
        var ret = stepper.step_adaptive(
            ctx, q, dt_try, compute_rhs, atol=Real(1.0e-8), rtol=Real(1.0e-6),
        )
        t += ret[0]
        dt = ret[1]
        steps += 1
    var err_adaptive = mesh.l2_error(q, IDX_EZ, exact)

    if is_root():
        print("ranks =", num_ranks(), " L2 error: fixed =", err_fixed,
              " adaptive =", err_adaptive, "in", steps, "steps")
        print("CHECKSUM", sum)
    if not (err_fixed < Real(2.0e-3) and err_adaptive < Real(2.0e-3)):
        raise Error("plane wave error too large")
    if is_root():
        print("PASSED")
