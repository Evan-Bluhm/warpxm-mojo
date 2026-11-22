# 3D multi-component advection — catches NC-indexing bugs in the 3D
# DG kernel.
#
# IC: two independent tracers
#   q_0(x, y, z, 0) = sin(2π x) · cos(2π y) · sin(2π z)
#   q_1(x, y, z, 0) = cos(2π x) · sin(2π y) · cos(2π z)
# advected with velocity (1, 1, 1) on [0, 1]³ periodic. After t=1
# both tracers return to their ICs.

from std.collections import Array
from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true

from warpxm.prelude import *
from warpxm.physics.advection import *


comptime P: Int = 2
comptime NC: Int = 2
comptime NX: Int = 16
comptime NY: Int = 16
comptime NZ: Int = 16
comptime State = DGState[3, P, NC]


def _tracer_0(x: Real, y: Real, z: Real) -> Real:
    return sin(TWO_PI * x) * cos(TWO_PI * y) * sin(TWO_PI * z)


def _tracer_1(x: Real, y: Real, z: Real) -> Real:
    return cos(TWO_PI * x) * sin(TWO_PI * y) * cos(TWO_PI * z)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var ctx = DeviceContext()
    var mesh = BlockMesh[3, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=NX, ny=NY, nz=NZ,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real, z: Real) -> Array[Real, NC]:
        return [_tracer_0(x, y, z), _tracer_1(x, y, z)]
    mesh.fill(q, initial)

    var spatial = DGAdvectionSolver[3, P, NC](
        Advection[NC](
            velocity_x=Real(1.0), velocity_y=Real(1.0), velocity_z=Real(1.0),
            cfl_max=Real(1.0),
        )
    )
    var stepper = SSPRK3[3, P, NC](ctx, mesh.num_elements)

    var dt: Real = 1.0e-3
    var t_final: Real = 1.0

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs, no_op_adjust[3, P, NC],
    )
    ctx.synchronize()

    var l2_0 = mesh.l2_error(q, 0, _tracer_0)
    var l2_1 = mesh.l2_error(q, 1, _tracer_1)

    print("Component 0 L2 error:", l2_0)
    print("Component 1 L2 error:", l2_1)
    assert_true(l2_0 < Real(5.0e-3), "component 0 error too large")
    assert_true(l2_1 < Real(5.0e-3), "component 1 error too large")
    print("PASSED")
