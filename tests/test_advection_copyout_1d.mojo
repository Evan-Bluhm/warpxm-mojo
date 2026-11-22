# Non-periodic 1D advection with copyout BCs on both ends.
#
# Set-up:
#   * Domain x ∈ [0, 1], 64 elements at P = 2.
#   * Initial condition: a smooth Gaussian pulse centred at x = 0.25.
#   * Velocity v = +1, so the pulse drifts toward the right (outflow) boundary.
#   * Both ends use `BC_COPYOUT` (zero-gradient).
#
# Expected behaviour at t = 1:
#   * Pulse has fully exited through the right boundary.
#   * Domain is approximately flat at the IC's leftmost value (≈ 0) since
#     copyout at the left boundary keeps soft-feeding the original baseline.
#   * No spurious oscillations / reflections.
#
# The pulse has a small support (≈ 0.05 wide); at t = 1 it sits centred at
# x = 1.25, i.e. well outside the domain, so the post-test L2 norm of `q`
# should be ≪ the initial peak height.

from std.collections import Array
from std.math import exp
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.advection import *


comptime P: Int = 2
comptime NUM_ELEMENTS: Int = 64
comptime SIGMA: Real = 0.025
comptime X0: Real = 0.25
comptime State = DGState[1, P, 1]


def _gauss(x: Real) -> Array[Real, 1]:
    var z = (x - X0) / SIGMA
    return [exp(-Real(0.5) * z * z)]


def _max_abs(mesh: BlockMesh[1, P], mut q: State) raises -> Real:
    var max_abs: Real = 0.0
    def visit(x: Real, w: Real, v: Array[Real, 1]) {mut max_abs}:
        max_abs = fmax(max_abs, fabs(v[0]))
    mesh.visit_nodes(q, visit)
    return max_abs


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](
        ctx, Real(0.0), Real(1.0), NUM_ELEMENTS, periodic=False,
        bc_left=BC_COPYOUT, bc_right=BC_COPYOUT,
    )
    var q = State(ctx, mesh.num_elements)
    mesh.fill(q, _gauss)
    var initial_peak = _max_abs(mesh, q)

    var app = Advection[1](velocity_x=Real(1.0), cfl_max=Real(1.0))
    var spatial = DGAdvectionSolver[1, P, 1](app)
    var stepper = SSPRK3[1, P, 1](ctx, mesh.num_elements)
    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs, no_op_adjust[1, P, 1],
    )
    ctx.synchronize()

    # Check max |q|. The pulse should have exited; remaining "echo" should be
    # tiny (well below 1% of the initial peak).
    var max_abs = _max_abs(mesh, q)
    print("initial peak =", initial_peak)
    print("final max|q| =", max_abs)
    assert_true(
        max_abs < Real(1.0e-3) * initial_peak,
        "pulse did not exit cleanly through copyout BC",
    )
    print("PASSED")
