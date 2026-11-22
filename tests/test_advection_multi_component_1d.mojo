# Multi-component regression: advect two independent tracer fields (sin and
# cos) together at NC = 2 and verify each lands back on itself after one
# period. Catches indexing / per-component bugs that NC = 1 would silently
# miss.

from std.collections import Array
from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.advection import *


comptime P: Int = 2
comptime NC: Int = 2
comptime NUM_ELEMENTS: Int = 64
comptime State = DGState[1, P, NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[1, P](ctx, Real(-0.5), Real(0.5), NUM_ELEMENTS, periodic=True)
    var q = State(ctx, mesh.num_elements)

    # Component 0: sin(2π x). Component 1: cos(2π x).
    def initial(x: Real) -> Array[Real, NC]:
        return [sin(TWO_PI * x), cos(TWO_PI * x)]
    mesh.fill(q, initial)

    var app = Advection[NC](velocity_x=Real(1.0), cfl_max=Real(1.0))
    var spatial = DGAdvectionSolver[1, P, NC](app)
    var stepper = SSPRK3[1, P, NC](ctx, mesh.num_elements)

    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs, no_op_adjust[1, P, NC],
    )
    ctx.synchronize()

    def exact_0(x: Real) -> Real:
        return sin(TWO_PI * x)
    def exact_1(x: Real) -> Real:
        return cos(TWO_PI * x)
    var l2_0 = mesh.l2_error(q, 0, exact_0)
    var l2_1 = mesh.l2_error(q, 1, exact_1)

    print("Component 0 (sin) L2 error:", l2_0)
    print("Component 1 (cos) L2 error:", l2_1)
    assert_true(l2_0 < Real(5.0e-5), "sin component error too large")
    assert_true(l2_1 < Real(5.0e-5), "cos component error too large")
    print("PASSED")
