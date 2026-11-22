# 2D multi-component advection — direct analogue of
# test_advection_multi_component_1d.mojo. Catches NC-indexing bugs in the
# 2D DG kernel that NC=1 would silently miss.
#
# IC: two independent tracers
#   q_0(x, y, 0) = sin(2π x) · cos(2π y)
#   q_1(x, y, 0) = cos(2π x) · sin(2π y)
# advected with velocity (1, 1) on [0, 1]² periodic. After t=1 both
# tracers return to their ICs.

from std.collections import Array
from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.advection import *


comptime P: Int = 2
comptime NC: Int = 2
comptime NX: Int = 32
comptime NY: Int = 32
comptime State = DGState[2, P, NC]


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, NC]:
        return [
            sin(TWO_PI * x) * cos(TWO_PI * y),
            cos(TWO_PI * x) * sin(TWO_PI * y),
        ]
    mesh.fill(q, initial)

    var spatial = DGAdvectionSolver[2, P, NC](
        Advection[NC](velocity_x=Real(1.0), velocity_y=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = SSPRK3[2, P, NC](ctx, mesh.num_elements)

    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs, no_op_adjust[2, P, NC],
    )
    ctx.synchronize()

    def exact_0(x: Real, y: Real) -> Real:
        return sin(TWO_PI * x) * cos(TWO_PI * y)
    def exact_1(x: Real, y: Real) -> Real:
        return cos(TWO_PI * x) * sin(TWO_PI * y)
    var l2_0 = mesh.l2_error(q, 0, exact_0)
    var l2_1 = mesh.l2_error(q, 1, exact_1)

    print("Component 0 (sin·cos) L2 error:", l2_0)
    print("Component 1 (cos·sin) L2 error:", l2_1)
    assert_true(l2_0 < Real(2.0e-4), "component 0 error too large")
    assert_true(l2_1 < Real(2.0e-4), "component 1 error too large")
    print("PASSED")
