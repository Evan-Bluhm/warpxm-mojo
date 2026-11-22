# 2D DG convergence regression — confirm rate of approximately (P + 1)
# on smooth 2D advection. Direct analogue of `test_advection_convergence_1d`.
#
# IC: q(x, y, 0) = sin(2π x) · cos(2π y) on [0, 1]² periodic.
# Velocity (1, 1) — diagonal advection; returns to IC at t = 1.
#
# Tests P = 1, 2, 3 by halving the element count and checking
# log2(err_lo / err_hi) ≈ P + 1.

from std.collections import Array
from std.math import sin, cos
from std.sys import has_accelerator
from max.gpu.host import DeviceContext

from warpxm.prelude import *
from warpxm.physics.advection import *
from warpxm.testing.convergence import check_convergence


def _wave(x: Real, y: Real) -> Real:
    return sin(TWO_PI * x) * cos(TWO_PI * y)


def _one_run[P: Int](
    var ctx: DeviceContext, num_per_side: Int, dt: Real, t_final: Real,
) raises -> Real:
    comptime State = DGState[2, P, 1]
    var mesh = BlockMesh[2, P](
        ctx, Real(0.0), Real(1.0), Real(0.0), Real(1.0),
        nx=num_per_side, ny=num_per_side,
        periodic_x=True, periodic_y=True,
    )
    var q = State(ctx, mesh.num_elements)

    def initial(x: Real, y: Real) -> Array[Real, 1]:
        return [_wave(x, y)]
    mesh.fill(q, initial)

    var spatial = DGAdvectionSolver[2, P, 1](
        Advection[1](velocity_x=Real(1.0), velocity_y=Real(1.0), cfl_max=Real(1.0))
    )
    var stepper = SSPRK3[2, P, 1](ctx, mesh.num_elements)

    def compute_rhs(mut q_in: State, mut q_out: State) raises {imm}:
        spatial.compute_rhs(ctx, mesh, q_in, q_out)

    _ = stepper.run_until(
        ctx, q, Real(0.0), t_final, dt, compute_rhs, no_op_adjust[2, P, 1],
    )
    ctx.synchronize()
    return mesh.l2_error(q, 0, _wave)


def _check_rate[P: Int](
    var ctx: DeviceContext, expected: Real, tol_lo: Real, tol_hi: Real,
) raises:
    """Check the observed rate and bound both errors absolutely.

    `tol_lo` / `tol_hi` are about 3x the measured coarse / fine errors, so
    a defect that scales the error uniformly (and leaves the rate intact)
    still fails.
    """
    var dt: Real = 2.5e-4
    var t_final: Real = 1.0
    var err_lo = _one_run[P](ctx, 16, dt, t_final)
    var err_hi = _one_run[P](ctx, 32, dt, t_final)
    _ = check_convergence(
        "P = " + String(P) + " (N = 16, 32)",
        err_lo, err_hi, expected, tol_lo, tol_hi,
    )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    var ctx = DeviceContext()
    _check_rate[1](ctx, Real(2.0), Real(3.3e-1), Real(9.0e-2))
    _check_rate[2](ctx, Real(3.0), Real(2.5e-3), Real(3.0e-4))
    _check_rate[3](ctx, Real(4.0), Real(5.0e-5), Real(3.2e-6))
    print("PASSED")
